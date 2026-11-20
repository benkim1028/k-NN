/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.opensearch.knn.clusterann.read.CentroidVectorValues;
import org.opensearch.knn.clusterann.read.Clusters;
import org.opensearch.knn.clusterann.stats.ClusterANNStat;

import java.io.IOException;
import java.util.Arrays;

/**
 * Chooses which clusters a query should visit.
 *
 * <p>Sweeps every centroid, keeps the {@code maxProbes} nearest non-empty clusters, and returns their ordinals
 * closest-first. Under a filter whose accepted vectors all fit in {@code k}, it keeps every cluster holding one
 * instead; under any other filter it goes past {@code maxProbes} when that is what holding {@code k} accepted vectors
 * takes. The scan relies on that order and does not restore it.
 *
 * <p>Ranks on a distance-like key (smaller is closer)
 *
 * <p>The ceiling bounds what a query may spend, unless a filter leaves it short of {@code k};
 * {@link CentroidKneePruner} cuts what it should not. Filtered plans are cut too, with a selectivity-scaled knee
 * factor, but never short of {@code k} accepted vectors.
 *
 * <p>Stateless and thread-safe: the only cursor it uses is a private copy taken per call. EXPERIMENT (do not merge):
 * plans also read the JVM-global {@link KneeExperiment} and {@link ProbeCurveExperiment} switches.
 */
public final class CentroidPlanner {

    private static final int[] NO_PROBES = new int[0];

    private CentroidPlanner() {}

    /**
     * Rank this field's centroids against the query and name the clusters to visit, considering only clusters reached
     * by the document-space filter when one is present.
     *
     * <p>A filter that is null or has null {@link AcceptDocs#bits()} takes the unfiltered path without reading
     * ordinal-to-centroid assignments. Lucene drops a filter that rewrites to match-all and hands each leaf its live
     * docs, so a match-all filter on a segment without deletions arrives here with null bits. Any other filter takes
     * the filtered path, even if it happens to accept every document.
     *
     * <p>When every accepted vector fits in {@code k}, the caller wants all of them, so every centroid holding one is
     * probed rather than only the {@code maxProbes} nearest.
     *
     * @param clusters the field's clusters
     * @param query the query vector, in the space it arrived in; not modified
     * @param params the bounds to choose within; a selective filter may raise the ceiling above {@code maxProbes}
     * @param k the number of results the caller wants
     * @param acceptDocs document-space filter, or null for an unfiltered search
     * @param vectorDocsSupplier a new iterator over the documents that have this vector field, exposing each one's
     *     vector ordinal; called only when a sparse field is filtered
     * @return centroid ordinals to probe, ordered closest-first; empty if no non-empty cluster matches
     */
    public static int[] plan(
        Clusters clusters,
        float[] query,
        PlanParams params,
        int k,
        AcceptDocs acceptDocs,
        IOSupplier<KnnVectorValues.DocIndexIterator> vectorDocsSupplier
    ) throws IOException {
        if (clusters.numClusters() == 0 || params.maxProbes() == 0) {
            return NO_PROBES;
        }
        if (acceptDocs == null || acceptDocs.bits() == null) {
            return plan(clusters, query, params);
        }

        final FilterDigest digest = FilterDigest.of(clusters, acceptDocs, vectorDocsSupplier).recordEligible();
        final FixedBitSet acceptedCentroids = digest.eligible();
        final int[] acceptedCountsPerCluster = digest.acceptedPerCluster();
        final int acceptedVectors = digest.accepted();
        final int eligibleClusters = digest.eligibleClusters();
        if (eligibleClusters == 0) {
            return NO_PROBES;
        }
        final double selectivity = (double) acceptedVectors / clusters.numVectors();
        final int budget = PlanParams.forSelectivity(clusters.numClusters(), selectivity).maxProbes();
        // Never cut when the caller wants every accepted vector.
        if (acceptedVectors <= k) {
            return closestFirst(rank(clusters, query, eligibleClusters, acceptedCentroids, acceptedCountsPerCluster, k));
        }
        final int maxProbes = Math.max(budget, params.maxProbes());
        final NeighborQueue nearest = rank(clusters, query, maxProbes, acceptedCentroids, acceptedCountsPerCluster, k);
        if (nearest.size() > maxProbes) {
            ClusterANNStat.FILTER_EXTENDED_PAST_MAX_PROBES.increment();
        }
        return closestFirstPruned(nearest, params.minProbes(), selectivity, acceptedCountsPerCluster, k);
    }

    /**
     * Rank every centroid in this field against the query and name the clusters to visit.
     *
     * <p>Costs one pass over the centroid region ({@code numClusters × dimension} floats), before any posting is
     * touched. Empty clusters are dropped during the sweep so they never take up a probe slot.
     *
     * @param clusters the field's clusters
     * @param query the query vector, in the space it arrived in; not modified
     * @param params the bounds to choose within
     * @return centroid ordinals to probe, ordered closest-first; empty if the field has no non-empty cluster
     */
    static int[] plan(Clusters clusters, float[] query, PlanParams params) throws IOException {
        if (clusters.numClusters() == 0 || params.maxProbes() == 0) {
            return NO_PROBES;
        }
        return closestFirstPruned(rank(clusters, query, params.maxProbes(), null, null, 0), params.minProbes(), 1.0, null, 0);
    }

    /**
     * This segment's share of a shard plan: its own centroids re-ranked with the planner's keys and, under a filter,
     * the filter's eligibility, keeping the first {@code segmentParams.maxProbes()}. No knee and no hold of {@code k}:
     * the shard's cut already did both. The ranking is a total order (key, then ordinal, as {@link NeighborQueue}
     * encodes it), so these are exactly the clusters the shard counted for this segment, ties included.
     *
     * <p>Scoring-time pruning hook: {@code segmentParams.minProbes()} is how many of the returned clusters fall inside
     * the shard's floor. A scan may stop early only past that many; nothing here stops it.
     *
     * @param digest the filter's digest, or null when nothing is filtered out in this segment
     * @return centroid ordinals, closest first; empty when {@code maxProbes} is 0 or nothing is eligible
     */
    public static int[] planWithin(Clusters clusters, float[] query, PlanParams segmentParams, FilterDigest digest) throws IOException {
        if (clusters.numClusters() == 0 || segmentParams.maxProbes() == 0 || (digest != null && digest.eligibleClusters() == 0)) {
            return NO_PROBES;
        }
        return closestFirst(rank(clusters, query, segmentParams.maxProbes(), digest == null ? null : digest.eligible(), null, 0));
    }

    /**
     * One field's clusters ranked against a query, closest first, with the planner's own distance-like keys (smaller
     * is closer). Keys are only comparable across segments because they are computed from each segment's
     * full-precision, input-space centroids and the raw query, never from rotated centroids or quantized scores.
     *
     * @param ords centroid ordinals, closest first
     * @param keys each ordinal's key, ascending
     */
    public record Ranked(int[] ords, float[] keys) {
        public int size() {
            return ords.length;
        }
    }

    /**
     * {@link #rank} drained closest-first. With a digest, the ranking considers only eligible clusters and keeps the
     * nearest that fill {@code cap} and hold {@code k} accepted vectors, which may be more than {@code cap}.
     *
     * @param digest the filter's digest, or null when nothing is filtered out
     */
    static Ranked rankClosestFirst(Clusters clusters, float[] query, int cap, FilterDigest digest, int k) throws IOException {
        return rankClosestFirst(clusters, query, cap, digest, k, false);
    }

    /**
     * As {@link #rankClosestFirst(Clusters, float[], int, FilterDigest, int)}; with {@code holdBySize} and no digest,
     * the ranking also holds {@code k} counting each cluster at its full size, for a segment with nothing filtered out
     * inside a shard whose cut must hold {@code k}. Still a prefix of the same order, so a re-rank keeps agreeing.
     */
    static Ranked rankClosestFirst(Clusters clusters, float[] query, int cap, FilterDigest digest, int k, boolean holdBySize)
        throws IOException {
        final NeighborQueue nearest;
        if (digest != null) {
            nearest = rank(clusters, query, Math.max(1, cap), digest.eligible(), digest.acceptedPerCluster(), k);
        } else if (holdBySize) {
            nearest = rank(clusters, query, Math.max(1, cap), null, clusters.clusterMeta().clusterSizes(), k);
        } else {
            nearest = rank(clusters, query, Math.max(1, cap), null, null, 0);
        }
        final int n = nearest.size();
        final int[] ords = new int[n];
        final float[] keys = new float[n];
        for (int i = n - 1; i >= 0; i--) {
            keys[i] = nearest.topScore();
            ords[i] = nearest.pop();
        }
        return new Ranked(ords, keys);
    }

    /**
     * The ranking heap capacity that {@link #plan(Clusters, float[], PlanParams, int, AcceptDocs, IOSupplier)} sweeps
     * this field with under {@code PlanParams.of(numClusters)}: what a ranking must cover for
     * {@link #segmentPlanLength} to read the segment's own plan off it.
     */
    static int segmentRankCap(Clusters clusters, FilterDigest digest, int k) {
        final PlanParams params = PlanParams.of(clusters.numClusters());
        if (digest == null) {
            return params.maxProbes();
        }
        if (digest.eligibleClusters() == 0) {
            return 0;
        }
        if (digest.accepted() <= k) {
            return digest.eligibleClusters();
        }
        final double selectivity = digest.selectivity(clusters.numVectors());
        return Math.max(PlanParams.forSelectivity(clusters.numClusters(), selectivity).maxProbes(), params.maxProbes());
    }

    /**
     * How many clusters this segment's own plan, under {@code PlanParams.of(numClusters)}, would probe for the query
     * that produced {@code ranked}. Read off an existing ranking, so it costs only the knee: {@code ranked} must have
     * been swept with at least {@link #segmentRankCap} (and, under a filter, holding {@code k}), which makes the
     * segment's own ranking a prefix of it.
     */
    static int segmentPlanLength(Clusters clusters, Ranked ranked, FilterDigest digest, int k) {
        final PlanParams params = PlanParams.of(clusters.numClusters());
        if (digest == null) {
            final int length = Math.min(params.maxProbes(), ranked.size());
            return CentroidKneePruner.prune(Arrays.copyOf(ranked.keys(), length), params.minProbes(), 1.0);
        }
        if (digest.eligibleClusters() == 0) {
            return 0;
        }
        if (digest.accepted() <= k) {
            return digest.eligibleClusters();
        }
        final int holding = holdingK(ranked.ords(), 0, digest.acceptedPerCluster(), k);
        final int length = Math.max(Math.min(segmentRankCap(clusters, digest, k), ranked.size()), holding);
        if (holding == length) {
            return length;
        }
        final double selectivity = digest.selectivity(clusters.numVectors());
        return Math.max(CentroidKneePruner.prune(Arrays.copyOf(ranked.keys(), length), params.minProbes(), selectivity), holding);
    }

    /**
     * Rank the candidate centroids, keeping the {@code maxProbes} nearest non-empty ones; given accepted counts, the
     * nearest that fill {@code maxProbes} and hold {@code k} accepted vectors, which may be more.
     *
     * @param eligibleCentroids ordinals to consider, or null when nothing is filtered out
     * @param acceptedCountsPerCluster accepted vectors per cluster, or null to keep at most {@code maxProbes}
     * @param k accepted vectors to hold; read only with {@code acceptedCountsPerCluster}
     */
    static NeighborQueue rank(
        Clusters clusters,
        float[] query,
        int maxProbes,
        FixedBitSet eligibleCentroids,
        int[] acceptedCountsPerCluster,
        int k
    ) throws IOException {
        final int numClusters = clusters.numClusters();
        final VectorSimilarityFunction similarity = clusters.clusterMeta().similarityFunction();
        final CentroidVectorValues centroids = clusters.centroids();
        final float queryNormSq = similarity == VectorSimilarityFunction.EUCLIDEAN ? VectorUtil.dotProduct(query, query) : 0f;

        // A cap for insertWithOverflow; only the initial capacity for add(), which grows past it.
        final NeighborQueue nearest = new NeighborQueue(maxProbes, true);
        // Accepted vectors in the clusters the heap holds.
        long held = 0;
        for (int ordinal = nextCandidate(eligibleCentroids, 0); ordinal < numClusters; ordinal = nextCandidate(
            eligibleCentroids,
            ordinal + 1
        )) {
            if (clusters.clusterSize(ordinal) != 0) {
                final float[] centroid = centroids.vectorValue(ordinal);
                final float key = distanceKey(similarity, query, queryNormSq, centroid, centroids.norm());
                if (acceptedCountsPerCluster == null) {
                    nearest.insertWithOverflow(ordinal, key);
                } else {
                    nearest.add(ordinal, key);
                    held += acceptedCountsPerCluster[ordinal];
                    // Keep the nearest clusters that fill the budget and hold k.
                    while (nearest.size() > maxProbes && held - acceptedCountsPerCluster[nearest.topNode()] >= k) {
                        held -= acceptedCountsPerCluster[nearest.pop()];
                    }
                }
            }
        }

        return nearest;
    }

    /** The first candidate ordinal at or after {@code from}: {@code from} itself when nothing is filtered out. */
    private static int nextCandidate(final FixedBitSet eligible, final int from) {
        if (eligible == null) {
            return from;
        }
        return from < eligible.length() ? eligible.nextSetBit(from) : DocIdSetIterator.NO_MORE_DOCS;
    }

    /**
     * Distance-like key from the query to a centroid, smaller meaning closer. Only EUCLIDEAN is a true distance:
     *
     * <ul>
     *   <li>EUCLIDEAN — {@code ‖q‖² − 2⟨q,c⟩ + ‖c‖²}, i.e. {@code d²}.
     *   <li>DOT_PRODUCT / MAXIMUM_INNER_PRODUCT — {@code −⟨q,c⟩}.
     *   <li>COSINE — {@code −⟨q,c⟩ / ‖c‖}; the norm matters because centroids of unit vectors are not unit norm.
     * </ul>
     */
    private static float distanceKey(
        VectorSimilarityFunction similarity,
        float[] query,
        float queryNormSq,
        float[] centroid,
        float centroidNorm
    ) {
        float dot = VectorUtil.dotProduct(query, centroid);
        return switch (similarity) {
            case EUCLIDEAN -> queryNormSq - 2 * dot + centroidNorm;
            case MAXIMUM_INNER_PRODUCT -> -dot;
            case DOT_PRODUCT -> throw new IllegalStateException("ClusterANN does not support DOT_PRODUCT; use MAXIMUM_INNER_PRODUCT");
            case COSINE -> -dot / (float) Math.sqrt(centroidNorm);
        };
    }

    /** Drain the heap into ordinals, closest-first. */
    private static int[] closestFirst(final NeighborQueue nearest) {
        final int[] probes = new int[nearest.size()];
        for (int i = probes.length - 1; i >= 0; i--) {
            probes[i] = nearest.pop();
        }
        return probes;
    }

    /**
     * Drain the heap into ordinals, closest-first, cutting at the knee unless {@link KneeExperiment} turns it off.
     *
     * @param selectivity fraction of vectors the filter accepts, which sets the knee factor; 1 when unfiltered
     * @param acceptedCountsPerCluster accepted vectors per cluster, or null when unfiltered
     * @param k results wanted; read only with {@code acceptedCountsPerCluster}
     */
    private static int[] closestFirstPruned(
        final NeighborQueue nearest,
        final int minProbes,
        final double selectivity,
        final int[] acceptedCountsPerCluster,
        final int k
    ) {
        // EXPERIMENT (do not merge): knee off plans every cluster the ranking kept: the ceiling, or the nearest that hold k.
        if (KneeExperiment.enabled() == false) {
            return closestFirst(nearest);
        }
        final int ranked = nearest.size();
        final int[] probes = new int[ranked];
        final float[] distances = new float[ranked];
        for (int i = ranked - 1; i >= 0; i--) {
            distances[i] = nearest.topScore();
            probes[i] = nearest.pop();
        }

        final int nearestHoldingK = acceptedCountsPerCluster == null ? 0 : holdingK(probes, 0, acceptedCountsPerCluster, k);
        if (nearestHoldingK == ranked) {
            return probes;
        }

        // The knee needs a floor of at least one; a 0 floor is only meaningful within a shard plan, which has no knee.
        final int keep = Math.max(CentroidKneePruner.prune(distances, Math.max(1, minProbes), selectivity), nearestHoldingK);
        if (keep < ranked) {
            // The knee pruner cut the plan short: the adaptive early-stop fired for this query.
            ClusterANNStat.CENTROID_KNEE_EARLY_STOP.increment();
        }
        return keep == ranked ? probes : Arrays.copyOf(probes, keep);
    }

    /**
     * Length of the shortest prefix of {@code probes} holding {@code k} accepted vectors, or all of them when the
     * ranked clusters hold fewer. Primary assignments only.
     */
    private static int holdingK(final int[] probes, final int keep, final int[] acceptedCountsPerCluster, final int k) {
        long held = 0;
        for (int i = 0; i < keep; i++) {
            held += acceptedCountsPerCluster[probes[i]];
        }
        int extended = keep;
        while (held < k && extended < probes.length) {
            held += acceptedCountsPerCluster[probes[extended++]];
        }
        return extended;
    }
}
