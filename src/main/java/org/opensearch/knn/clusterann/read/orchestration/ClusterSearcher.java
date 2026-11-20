/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.clusterann.read.Cluster;
import org.opensearch.knn.clusterann.read.Clusters;
import org.opensearch.knn.clusterann.read.PostingScorer;
import org.opensearch.knn.clusterann.read.ScanParams;
import org.opensearch.knn.clusterann.stats.ClusterANNStat;

import java.io.IOException;
import java.util.BitSet;

/**
 * Scans a ranked probe list, scanning each cluster into the collector.
 *
 * <p>Owns the concerns that span clusters — SOAR dedup, filter membership and, when asked for, the early stop
 * ({@link EarlyStopParams}) — and nothing else.
 * Storage-agnostic: it only ever consumes a {@link PostingScorer}, so how postings are stored or scored
 * never reaches it. It takes the probe order as given and does not reorder it, since pruning and the
 * collector's competitive threshold both depend on visiting closest-first.
 *
 * <p>Stateless: everything is passed in, and everything it mutates is local to one call.
 */
public final class ClusterSearcher {

    private ClusterSearcher() {}

    /**
     * Scan every cluster in {@code probes} into {@code collector}, with no early stop.
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
        return search(clusters, probes, params, EarlyStopParams.OFF, collector, acceptedOrds).scanned();
    }

    /**
     * Scan the clusters in {@code probes} into {@code collector}, stopping short of the plan when {@code earlyStop}
     * says the rest are unlikely to help (see {@link EarlyStopParams} for the rule).
     *
     * <p>With {@link EarlyStopParams#OFF} this visits every probe and makes exactly the collector calls it made before
     * the stop existed, and records no early-stop stat.
     *
     * @param clusters the field's clusters
     * @param probes centroid ordinals to visit, closest-first
     * @param params the query and its per-query knobs
     * @param earlyStop when the scan may stop short of {@code probes}; {@link EarlyStopParams#OFF} for never
     * @param collector where hits go, and the source of the competitive threshold that drives pruning
     * @param acceptedOrds ord-space filter, or {@code null} to accept every vector; composed here with
     *     SOAR dedup into the single membership test each cluster applies before scoring
     * @return how many clusters were scanned, and how many probes the stop left unvisited
     */
    public static Result search(
        Clusters clusters,
        int[] probes,
        ScanParams params,
        EarlyStopParams earlyStop,
        KnnCollector collector,
        Bits acceptedOrds
    ) throws IOException {
        // acceptedOrds is deliberately absent: null is the match-all filter, the same meaning it carries in
        // AcceptDocs#bits and LeafReader#getLiveDocs, and it is what an unfiltered query on an undeleted segment
        // arrives with. Rejecting it here would fail the most common query shape.
        if (clusters == null || params == null || probes == null || earlyStop == null) {
            throw new IllegalArgumentException("clusters, probes, params and earlyStop must be non-null");
        }

        if (clusters.numClusters() == 0 || probes.length == 0) {
            return new Result(0, 0);
        }

        // Dedup across clusters: SOAR places a boundary vector in two of them, so the same ordinal can be reached
        // twice. Local to this scan, since it means nothing outside one query.
        BitSet visited = new BitSet(clusters.numVectors());
        Bits wanted = wanted(acceptedOrds, visited, clusters.numVectors());

        final boolean stopEnabled = earlyStop.enabled();
        // Under a filter, a cluster counts toward patience only if it had a candidate to score. The latch watches the
        // one membership test the cluster applies, so SOAR copies and the dedup are seen exactly as the scorer saw
        // them. Unfiltered, every non-empty cluster counts, so there is nothing to watch.
        final AcceptedLatch latch = stopEnabled && acceptedOrds != null ? new AcceptedLatch(wanted) : null;
        if (latch != null) {
            wanted = latch;
        }
        final int patience = earlyStop.patience();
        // The collector's k, the queue the inserts are counted against. For ClusterANNQuery that is its candidateK (the
        // query's k, or k x oversample when a rescore follows): ClusterANNCollectorManager is not optimistic.
        final int k = stopEnabled ? collector.k() : 0;
        final int epsilon = stopEnabled ? earlyStop.epsilonFor(k) : 0;
        final int floor = stopEnabled ? earlyStop.floorFor(clusters.numClusters()) : 0;

        ClusterScan scan = clusters.scan();
        final Tally tally = new Tally();
        int scanned = 0;
        int quiet = 0;
        int unvisited = 0;
        for (int i = 0; i < probes.length; i++) {
            Cluster cluster = clusters.get(probes[i]);
            if (cluster.size() == 0) {
                continue;
            }
            if (latch != null) {
                latch.reset();
            }
            final int inserts = scanCluster(scan, cluster, params, wanted, visited, collector, tally);
            scanned++;
            if (stopEnabled == false) {
                continue;
            }
            // Cheapest test first; the collector is asked last, and only about a cluster that could count.
            if ((latch == null || latch.accepted) && scanned >= floor && tally.collected >= k && isFull(collector)) {
                quiet = inserts < epsilon ? quiet + 1 : 0;
                if (quiet >= patience) {
                    unvisited = probes.length - i - 1;
                    if (unvisited > 0) {
                        // Not counted when the stop fired on the last probe, since nothing was left to skip.
                        ClusterANNStat.SCAN_EARLY_STOP.increment();
                    }
                    break;
                }
            }
            if (collector.earlyTerminated()) {
                unvisited = probes.length - i - 1;
                break;
            }
        }
        // Distinct ordinals the collector took this query (SOAR dedup already applied via `visited`). Not the
        // scoring count: blocks dropped on the threshold were scored in full without reaching the collector, and
        // ClusterANNStat.VECTORS_SCORED counts those.
        ClusterANNStat.VECTORS_COLLECTED_PER_QUERY.record(visited.cardinality());
        if (stopEnabled) {
            // Probes walked, empty ones included, against probes planned: 10000 when the scan ran its whole plan.
            ClusterANNStat.SCAN_EARLY_STOP_SCANNED_BPS.record(((probes.length - unvisited) * 10_000L) / probes.length);
        }
        return new Result(scanned, unvisited);
    }

    /**
     * Whether the collector holds k results, so a quiet cluster is evidence rather than a queue still filling.
     *
     * <p>{@link KnnCollector} exposes no size, and counting this scan's inserts would not be one either: a
     * {@code MultiLeafKnnCollector} reports {@code true} for a vector that entered only the shared cross-segment
     * queue. The competitive threshold is the cheapest correct test. {@code TopKnnCollector} returns
     * {@link Float#NEGATIVE_INFINITY} until its queue holds k and the queue's lowest score after, and every Lucene
     * similarity score is finite; {@code MultiLeafKnnCollector} returns it until its own segment's sub-collector
     * holds k; the decorators the reader adds ({@code OrdinalTranslatedKnnCollector}, the time-limiting one)
     * delegate. A collector that never raises its threshold therefore never lets the scan stop, which is the safe
     * way to be wrong; so does a {@code NaN} threshold, since it compares false.
     *
     * <p>That rests on one precondition: the collector's threshold stays at {@code -inf} until it holds k. A collector
     * seeded with a threshold of its own (a radius or a minimum score) would look full from the first cluster, so the
     * caller also requires this scan's own collect count to reach k.
     */
    private static boolean isFull(KnnCollector collector) {
        return collector.minCompetitiveSimilarity() > Float.NEGATIVE_INFINITY;
    }

    /** Running count of {@code collect} calls that returned {@code true} this scan, whatever their score. */
    private static final class Tally {
        private int collected;
    }

    /**
     * What a scan did.
     *
     * @param scanned non-empty clusters scanned
     * @param unvisited probes after the last one visited, left unscanned because the scan stopped early; {@code 0}
     *     when it ran to the end of the plan, as it always does with {@link EarlyStopParams#OFF}
     */
    public record Result(int scanned, int unvisited) {
    }

    /**
     * Wraps a cluster's membership test and remembers whether it let any candidate through since the last
     * {@link #reset()}. One per scan, reset per cluster, so it allocates nothing per probe.
     */
    private static final class AcceptedLatch implements Bits {

        private final Bits wanted;
        private boolean accepted;

        private AcceptedLatch(Bits wanted) {
            this.wanted = wanted;
        }

        private void reset() {
            accepted = false;
        }

        @Override
        public boolean get(int ord) {
            final boolean want = wanted.get(ord);
            accepted |= want;
            return want;
        }

        @Override
        public int length() {
            return wanted.length();
        }
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

    /**
     * Scan one cluster's postings into the collector.
     *
     * @return how many of the cluster's vectors entered the top k: {@code collect} returned {@code true} for them and
     *     they scored above the competitive threshold the collector reported before them
     */
    private static int scanCluster(
        ClusterScan scan,
        Cluster cluster,
        ScanParams params,
        Bits wanted,
        BitSet visited,
        KnnCollector collector,
        Tally tally
    ) throws IOException {
        PostingScorer scorer = scan.scorer(cluster, params, wanted);
        int inserts = 0;
        // The threshold is read once per step, before each advance, exactly as it always was; it is kept so the
        // insert below can be checked against the bar the vector had to clear.
        for (float threshold = collector.minCompetitiveSimilarity(); scorer.advance(threshold); threshold = collector
            .minCompetitiveSimilarity()) {
            int ord = scorer.ord();
            visited.set(ord);
            collector.incVisitedCount(1);
            final float score = scorer.score();
            // An insert that also beat the threshold. For TopKnnCollector the second test changes nothing but exact ties
            // at its lowest score: its threshold is -inf below k and that lowest score after. A MultiLeafKnnCollector
            // also returns true when it merely syncs with the cross-segment queue (every 256 visits) or updates its
            // non-competitive queue; those vectors did not beat the bar and do not count.
            if (collector.collect(ord, score)) {
                tally.collected++;
                if (score > threshold) {
                    inserts++;
                }
            }
        }
        return inserts;
    }
}
