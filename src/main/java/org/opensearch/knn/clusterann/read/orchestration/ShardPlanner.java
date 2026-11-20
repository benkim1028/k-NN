/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.opensearch.common.Nullable;
import org.opensearch.knn.clusterann.read.Clusters;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Plans one query for a whole shard: which clusters of which segment to visit.
 *
 * <p>Every segment's centroids are ranked against the query by the planner's own keys ({@link CentroidPlanner#rank}),
 * which are computed from each segment's full-precision, input-space centroids and the raw query, so they compare
 * across segments. The shard keeps the global top {@code X} of the merged ranking and hands each segment its own share,
 * closest first. Under a filter the global prefix grows until the shard, not each segment, holds {@code k} accepted
 * vectors; when the shard holds {@code k} or fewer, every eligible cluster is kept.
 *
 * <p>{@code X} is set by a {@link Budget}. The shard's {@link PlanParams} come from the shard's total cluster count and
 * its filter selectivity, and travel with the plan so scoring-time pruning can read them.
 *
 * <p>Stateless and thread-safe.
 */
public final class ShardPlanner {

    /** Probe multiplier R never goes below, whatever the selectivity: R = ceil(max(8, m(s)) * floor(sqrt(C))). */
    static final double RULE_MIN_MULTIPLIER = 8.0;

    /** How many clusters the shard keeps: the dial between recall and cost. */
    public enum Budget {
        /**
         * Rule H, the default: {@code X = min(R, P)}, where {@code P} is what each segment's own knee-cut plan would
         * probe for this query, summed over the shard. Never probes more than per-segment planning did; the
         * per-segment knee survives only as this cost cap, until scoring-time pruning can stop scans early.
         */
        CAPPED,
        /** Rule R uncapped: {@code X = R}, no knee. Meant for when scoring-time pruning stops scans early. */
        RULE
    }

    /**
     * One segment's input to the plan.
     *
     * @param clusters the segment's clusters for the field
     * @param digest what the filter leaves of them, or null when nothing in this segment is filtered out
     */
    public record Segment(Clusters clusters, @Nullable FilterDigest digest) {
    }

    /**
     * The shard's plan.
     *
     * @param params the shard's bounds: {@code maxProbes} is R, {@code minProbes} the shipped floor at the shard's
     *     cluster count; null only when the shard has no cluster at all
     * @param probes per segment, in input order, the centroid ordinals to visit, closest first; empty for a segment
     *     the shard gave nothing
     * @param budget X: clusters the shard set out to keep before any filter hold or take-all
     * @param totalClusters clusters across every segment (C_total)
     * @param selectivity fraction of the shard's vectors the filter accepts; 1 when unfiltered
     */
    public record Plan(@Nullable PlanParams params, int[][] probes, int budget, int totalClusters, double selectivity) {
        /** Clusters the plan probes across the shard. */
        public int probeCount() {
            int total = 0;
            for (final int[] p : probes) {
                total += p.length;
            }
            return total;
        }
    }

    private ShardPlanner() {}

    /**
     * Shard bounds for a total cluster count and selectivity: {@code maxProbes = R = ceil(max(8, m(s)) * floor(sqrt
     * C))}, {@code minProbes} the floor {@link PlanParams#forSelectivity} ships at {@code C}.
     *
     * <p>R departs from the shipped curve, whose ceiling is {@code ceil(m(s) * floor(sqrt C))} (3x at s = 1): R is the
     * multiplier the shard-budget evaluation measured, so it is kept over the curve.
     */
    public static PlanParams shardParams(final int totalClusters, final double selectivity) {
        if (totalClusters < 1) {
            throw new IllegalArgumentException("totalClusters must be at least 1, got: " + totalClusters);
        }
        final int root = (int) Math.sqrt(totalClusters);
        final double multiplier = Math.max(RULE_MIN_MULTIPLIER, PlanParams.multiplier(selectivity));
        final int maxProbes = Math.max(1, Math.min((int) Math.ceil(multiplier * root), totalClusters));
        final int minProbes = Math.min(PlanParams.forSelectivity(totalClusters, selectivity).minProbes(), maxProbes);
        return new PlanParams(minProbes, maxProbes);
    }

    /**
     * Plan {@code query} over every segment of a shard.
     *
     * @param segments one entry per segment that has clusters for the field
     * @param query the query vector, in the space it arrived in; not modified
     * @param k results the caller wants; under a filter, accepted vectors the shard must hold
     * @param budget how X is chosen
     */
    public static Plan plan(final List<Segment> segments, final float[] query, final int k, final Budget budget) throws IOException {
        final int n = segments.size();
        int totalClusters = 0;
        long vectors = 0;
        long accepted = 0;
        boolean filtered = false;
        for (final Segment segment : segments) {
            totalClusters += segment.clusters().numClusters();
            vectors += segment.clusters().numVectors();
            if (segment.digest() == null) {
                accepted += segment.clusters().numVectors();
            } else {
                filtered = true;
                accepted += segment.digest().accepted();
            }
        }
        final int[][] probes = new int[n][];
        if (totalClusters == 0) {
            Arrays.fill(probes, new int[0]);
            return new Plan(null, probes, 0, 0, 1.0);
        }
        final double selectivity = filtered ? (double) accepted / Math.max(1, vectors) : 1.0;
        final PlanParams params = shardParams(totalClusters, selectivity);
        final int rule = params.maxProbes();
        final boolean takeAll = filtered && accepted <= k;

        // 1. Rank every segment. Each list is long enough to give the global top R, and under a filter holds k on its
        // own, so the merged prefix can always be extended to hold k. Under H it also covers the segment's own plan.
        final CentroidPlanner.Ranked[] ranked = new CentroidPlanner.Ranked[n];
        for (int s = 0; s < n; s++) {
            final Segment segment = segments.get(s);
            final Clusters clusters = segment.clusters();
            final FilterDigest digest = segment.digest();
            final int numClusters = clusters.numClusters();
            if (numClusters == 0 || (digest != null && digest.eligibleClusters() == 0)) {
                ranked[s] = new CentroidPlanner.Ranked(new int[0], new float[0]);
                continue;
            }
            final int candidates = digest == null ? numClusters : digest.eligibleClusters();
            int cap = takeAll ? candidates : Math.min(rule, candidates);
            if (budget == Budget.CAPPED) {
                cap = Math.max(cap, CentroidPlanner.segmentRankCap(clusters, digest, k));
            }
            ranked[s] = CentroidPlanner.rankClosestFirst(clusters, query, cap, digest, k);
        }

        // 2. X.
        int x = rule;
        if (budget == Budget.CAPPED) {
            int perSegment = 0;
            for (int s = 0; s < n; s++) {
                if (ranked[s].size() > 0) {
                    perSegment += CentroidPlanner.segmentPlanLength(segments.get(s).clusters(), ranked[s], segments.get(s).digest(), k);
                }
            }
            x = Math.min(rule, perSegment);
        }

        // 3. Merge the closest-first lists, keeping the global top X, extended until the shard holds k under a filter.
        final int[] take = mergeTake(segments, ranked, takeAll ? Integer.MAX_VALUE : x, filtered ? k : 0);
        for (int s = 0; s < n; s++) {
            probes[s] = Arrays.copyOf(ranked[s].ords(), take[s]);
        }
        return new Plan(params, probes, x, totalClusters, selectivity);
    }

    /**
     * How many of each segment's ranked clusters the merged prefix takes: at least {@code x} overall (or everything),
     * and, with {@code holdK > 0}, as many more as it takes to hold {@code holdK} accepted vectors. Ties on the key go
     * to the lower segment, then to the segment's own order, so the plan is deterministic.
     */
    private static int[] mergeTake(final List<Segment> segments, final CentroidPlanner.Ranked[] ranked, final int x, final int holdK) {
        final int n = ranked.length;
        final int[] take = new int[n];
        final PriorityQueue<Integer> heads = new PriorityQueue<>(Math.max(1, n), (a, b) -> {
            final int byKey = Float.compare(ranked[a].keys()[take[a]], ranked[b].keys()[take[b]]);
            return byKey != 0 ? byKey : Integer.compare(a, b);
        });
        for (int s = 0; s < n; s++) {
            if (ranked[s].size() > 0) {
                heads.add(s);
            }
        }
        int taken = 0;
        long held = 0;
        while (!heads.isEmpty() && (taken < x || held < holdK)) {
            final int s = heads.poll();
            final int ord = ranked[s].ords()[take[s]];
            held += heldBy(segments.get(s), ord);
            take[s]++;
            taken++;
            if (take[s] < ranked[s].size()) {
                heads.add(s);
            }
        }
        return take;
    }

    /**
     * Accepted vectors a cluster contributes to the shard's hold. A segment with nothing filtered out has no digest,
     * so its cluster counts as its full size, SOAR copies included: an upper bound on its accepted primaries.
     */
    private static long heldBy(final Segment segment, final int ord) {
        return segment.digest() == null ? segment.clusters().clusterSize(ord) : segment.digest().acceptedPerCluster()[ord];
    }
}
