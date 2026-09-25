/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.clusterann.read.Cluster;
import org.opensearch.knn.clusterann.read.Clusters;
import org.opensearch.knn.clusterann.read.PostingScorer;
import org.opensearch.knn.clusterann.read.ScanParams;
import org.opensearch.knn.plugin.stats.ClusterANNQueryValue;

import java.io.IOException;
import java.util.BitSet;

/**
 * Scans a ranked probe list, scanning each cluster into the collector.
 *
 * <p>Owns the concerns that span clusters — SOAR dedup and filter membership — and nothing else.
 * Storage-agnostic: it only ever consumes a {@link PostingScorer}, so how postings are stored or scored
 * never reaches it. It takes the probe order as given and does not reorder it, since pruning and the
 * collector's competitive threshold both depend on visiting closest-first.
 *
 * <p>Stateless as far as a scan's result goes: every input is passed in and nothing about one call carries into the
 * next. The exception is bookkeeping — it counts the clusters it probes and scans into {@link ClusterANNQueryValue},
 * the node's counters, as it goes. Blocks and distances are counted where they happen, inside the scorer, so
 * nothing has to be threaded back out.
 */
public final class ClusterSearcher {

    interface ClusterSource {
        int numClusters();

        int numVectors();

        Cluster get(int ordinal) throws IOException;

        ClusterScan scan();
    }

    private ClusterSearcher() {}

    /**
     * Scan each cluster in {@code probes} into {@code collector}.
     *
     * @param clusters the field's clusters
     * @param probes centroid ordinals to visit, closest-first
     * @param params the query and its per-query knobs
     * @param collector where hits go, and the source of the competitive threshold that drives pruning
     * @param acceptedOrds ord-space filter, or {@code null} to accept every vector; composed here with
     *     SOAR dedup into the single membership test each cluster applies before scoring
     * @return the number of clusters actually scanned (fewer than {@code probes.length} when some are empty)
     */
    public static int search(Clusters clusters, int[] probes, ScanParams params, KnnCollector collector, Bits acceptedOrds)
        throws IOException {
        // acceptedOrds is deliberately absent: null is the match-all filter, the same meaning it carries in
        // AcceptDocs#bits and LeafReader#getLiveDocs, and it is what an unfiltered query on an undeleted segment
        // arrives with. Rejecting it here would fail the most common query shape.
        if (clusters == null || params == null || probes == null) {
            throw new IllegalArgumentException("clusters, probes and params must be non-null");
        }

        if (clusters.numClusters() == 0 || probes.length == 0) {
            // Not counted as a scan: nothing was looked at, so averaging it in would only dilute the numbers.
            return 0;
        }

        // Dedup across clusters: SOAR places a boundary vector in two of them, so the same ordinal can be reached
        // twice. Local to this scan, since it means nothing outside one query.
        BitSet visited = new BitSet(clusters.numVectors());
        Bits wanted = wanted(acceptedOrds, visited, clusters.numVectors());

        // Counted before the walk rather than after it: a scan that throws or times out still probed these clusters,
        // and leaving it out would make a failing query look cheap.
        ClusterANNQueryValue.SEGMENT_SCANS.increment();
        ClusterANNQueryValue.CLUSTERS_PROBED.incrementBy(probes.length);

        ClusterScan scan = clusters.scan();
        int scanned = 0;
        for (int probe : probes) {
            Cluster cluster = clusters.get(probe);
            if (cluster.size() != 0) {
                ClusterANNQueryValue.CLUSTERS_SCANNED.increment();
                scanCluster(scan, cluster, params, wanted, visited, collector);
                scanned++;
            }
        }
        return scanned;
    }

    public static int searchFiltered(
        Clusters clusters,
        int[] probes,
        ScanParams params,
        KnnCollector collector,
        Bits acceptedOrds,
        int[] filterMatchCounts,
        RandomVectorScorer exactScorer
    ) throws IOException {
        if (clusters == null || params == null || probes == null || filterMatchCounts == null || exactScorer == null) {
            throw new IllegalArgumentException("clusters, probes, params, filterMatchCounts and exactScorer must be non-null");
        }
        return searchFiltered(source(clusters), probes, params, collector, acceptedOrds, filterMatchCounts, exactScorer);
    }

    static int searchFiltered(
        ClusterSource clusters,
        int[] probes,
        ScanParams params,
        KnnCollector collector,
        Bits acceptedOrds,
        int[] filterMatchCounts,
        RandomVectorScorer exactScorer
    ) throws IOException {
        if (clusters == null || params == null || probes == null || filterMatchCounts == null || exactScorer == null) {
            throw new IllegalArgumentException("clusters, probes, params, filterMatchCounts and exactScorer must be non-null");
        }

        if (clusters.numClusters() == 0 || probes.length == 0) {
            return 0;
        }

        BitSet visited = new BitSet(clusters.numVectors());
        Bits wanted = wanted(acceptedOrds, visited, clusters.numVectors());

        ClusterScan scan = clusters.scan();
        int scanned = 0;
        long docsScored = 0;
        int consecutiveNonImproving = 0;
        final int k = collector.k();
        for (int i = 0; i < probes.length; i++) {
            int probe = probes[i];
            Cluster cluster = clusters.get(probe);
            if (cluster.size() != 0) {
                float thresholdBefore = collector.minCompetitiveSimilarity();
                int scored = filterMatchCounts[probe] < k
                    ? scanClusterExact(cluster, wanted, visited, collector, exactScorer)
                    : scanCluster(scan, cluster, params, wanted, visited, collector);
                scanned++;
                docsScored += scored;

                if (collector.earlyTerminated()) {
                    break;
                }

                float thresholdAfterBudget = collector.minCompetitiveSimilarity();
                if (shouldStopAfterBudget(docsScored, k, thresholdAfterBudget)) {
                    break;
                }

                if (i >= 2 && docsScored >= (long) k * 3) {
                    float thresholdAfter = collector.minCompetitiveSimilarity();
                    consecutiveNonImproving = nextConsecutiveNonImproving(consecutiveNonImproving, thresholdBefore, thresholdAfter);
                    if (shouldStopAfterContribution(i, docsScored, k, consecutiveNonImproving, thresholdAfter)) {
                        break;
                    }
                }
            }
        }
        return scanned;
    }

    static boolean shouldStopAfterBudget(long docsScored, int k, float thresholdAfter) {
        return docsScored >= (long) k * 3 && thresholdAfter != Float.NEGATIVE_INFINITY;
    }

    static int nextConsecutiveNonImproving(int consecutiveNonImproving, float thresholdBefore, float thresholdAfter) {
        boolean improving = thresholdAfter > thresholdBefore && thresholdBefore != Float.NEGATIVE_INFINITY;
        if (improving) {
            return 0;
        }
        if (thresholdBefore != Float.NEGATIVE_INFINITY || thresholdAfter != Float.NEGATIVE_INFINITY) {
            return consecutiveNonImproving + 1;
        }
        return consecutiveNonImproving;
    }

    static boolean shouldStopAfterContribution(int probeIndex, long docsScored, int k, int consecutiveNonImproving, float thresholdAfter) {
        return probeIndex >= 2 && docsScored >= (long) k * 3 && consecutiveNonImproving >= 2 && thresholdAfter != Float.NEGATIVE_INFINITY;
    }

    private static ClusterSource source(Clusters clusters) {
        return new ClusterSource() {
            @Override
            public int numClusters() {
                return clusters.numClusters();
            }

            @Override
            public int numVectors() {
                return clusters.numVectors();
            }

            @Override
            public Cluster get(int ordinal) throws IOException {
                return clusters.get(ordinal);
            }

            @Override
            public ClusterScan scan() {
                return clusters.scan();
            }
        };
    }

    /**
     * The single membership test a cluster applies before scoring: the caller's filter and this walk's dedup as one
     * {@link Bits}, so a cluster never learns there were two.
     */
    private static Bits wanted(Bits acceptedOrds, BitSet visited, int numOfVectors) {
        return new Bits() {
            @Override
            public boolean get(int ord) {
                return (acceptedOrds == null || acceptedOrds.get(ord)) && !visited.get(ord);
            }

            @Override
            public int length() {
                return acceptedOrds != null ? acceptedOrds.length() : numOfVectors;
            }
        };
    }

    /** Scan one cluster's postings into the collector. */
    private static int scanCluster(
        ClusterScan scan,
        Cluster cluster,
        ScanParams params,
        Bits wanted,
        BitSet visited,
        KnnCollector collector
    ) throws IOException {
        int scored = 0;
        PostingScorer scorer = scan.scorer(cluster, params, wanted);
        while (scorer.advance(collector.minCompetitiveSimilarity())) {
            int ord = scorer.ord();
            visited.set(ord);
            collector.incVisitedCount(1);
            collector.collect(ord, scorer.score());
            scored++;
        }
        return scored;
    }

    private static int scanClusterExact(
        Cluster cluster,
        Bits wanted,
        BitSet visited,
        KnnCollector collector,
        RandomVectorScorer exactScorer
    ) throws IOException {
        int scored = 0;
        PostingScorer scorer = cluster.exactScorer(exactScorer, wanted);
        while (scorer.advance(collector.minCompetitiveSimilarity())) {
            int ord = scorer.ord();
            visited.set(ord);
            collector.incVisitedCount(1);
            collector.collect(ord, scorer.score());
            scored++;
        }
        return scored;
    }
}
