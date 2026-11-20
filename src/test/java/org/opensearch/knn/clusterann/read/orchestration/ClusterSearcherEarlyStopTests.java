/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.read.Cluster;
import org.opensearch.knn.clusterann.read.Clusters;
import org.opensearch.knn.clusterann.read.PostingScorer;
import org.opensearch.knn.clusterann.read.ScanParams;
import org.opensearch.knn.clusterann.stats.ClusterANNStat;

import java.io.IOException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * The scan early stop in {@link ClusterSearcher} (JUnit 5 + Mockito): P quiet clusters in a row stop the scan, a
 * useful one resets the run, nothing counts before the floor or before the collector is full, a cluster the filter
 * emptied does not count, the collector's own early termination stops it, and with the stop off the walk is call for
 * call what it was before the stop existed.
 *
 * <p>The collector is Lucene's own {@link TopKnnCollector}, so "full" and "entered the top k" mean what they mean in
 * production. Each cluster is a list of ordinals with fixed scores; the stub scorer hands out every one the
 * membership test accepts, so what enters the top k is decided by the collector alone.
 */
class ClusterSearcherEarlyStopTests {

    private static final float[] QUERY = { 0.1f, 0.2f, 0.3f, 0.4f };

    /** Ordinal to score. Ordinals 1xx score high, 9xx score too low to enter a full top 2. */
    private static final Map<Integer, Float> SCORES = new HashMap<>();

    static {
        for (int ord = 0; ord < 1000; ord++) {
            SCORES.put(ord, ord < 900 ? 0.5f + (ord % 100) / 1000f : 0.01f + (ord % 100) / 10_000f);
        }
    }

    /** patience 2, epsilon 1, no floor: the plainest rule to reason about. */
    private static final EarlyStopParams P2 = new EarlyStopParams(2, 1, 0);

    // ---------------------------------------------------------------- the stop

    /** The top 2 fills in the first cluster; the next two put nothing in it, so the scan stops before the fourth. */
    @Test
    void testSearch_whenPatienceQuietClustersInARow_thenStops() throws IOException {
        // given
        Clusters clusters = clusters(new int[] { 100, 101 }, new int[] { 900 }, new int[] { 901 }, new int[] { 199 });
        TopKnnCollector collector = new TopKnnCollector(2, Integer.MAX_VALUE);
        long stops0 = ClusterANNStat.SCAN_EARLY_STOP.getCount();

        // when
        ClusterSearcher.Result result = ClusterSearcher.search(clusters, probes(4), ScanParams.of(QUERY), P2, collector, null);

        // then
        assertEquals(3, result.scanned(), "two quiet clusters after the fill, then stop");
        assertEquals(1, result.unvisited(), "the fourth probe is never reached");
        assertEquals(4, collector.visitedCount(), "the fourth cluster scored nothing");
        assertEquals(1, ClusterANNStat.SCAN_EARLY_STOP.getCount() - stops0, "the stop is counted once");
    }

    /** A useful cluster between quiet ones restarts the count: the stop needs P quiet clusters in a row. */
    @Test
    void testSearch_whenAUsefulClusterInterrupts_thenTheQuietRunResets() throws IOException {
        // given — quiet, useful (199 beats the full top 2), quiet, quiet, then one never reached
        Clusters clusters = clusters(
            new int[] { 100, 101 },
            new int[] { 900 },
            new int[] { 199 },
            new int[] { 901 },
            new int[] { 902 },
            new int[] { 198 }
        );
        TopKnnCollector collector = new TopKnnCollector(2, Integer.MAX_VALUE);

        // when
        ClusterSearcher.Result result = ClusterSearcher.search(clusters, probes(6), ScanParams.of(QUERY), P2, collector, null);

        // then
        assertEquals(5, result.scanned(), "without the reset the scan would have stopped after the fourth cluster");
        assertEquals(1, result.unvisited());
    }

    /** epsilon is a threshold, not "any": a cluster adding fewer than epsilon to the top k is still quiet. */
    @Test
    void testSearch_whenAClusterAddsFewerThanEpsilon_thenItIsStillQuiet() throws IOException {
        // given — each later cluster puts exactly one vector into the top 2, below epsilon 2
        Clusters clusters = clusters(new int[] { 100, 101 }, new int[] { 150 }, new int[] { 160 }, new int[] { 170 });
        TopKnnCollector collector = new TopKnnCollector(2, Integer.MAX_VALUE);

        // when
        ClusterSearcher.Result result = ClusterSearcher.search(
            clusters,
            probes(4),
            ScanParams.of(QUERY),
            new EarlyStopParams(2, 2, 0),
            collector,
            null
        );

        // then
        assertEquals(3, result.scanned());
    }

    /** No cluster counts before the floor: with floor 4 the earliest stop is after cluster floor + P - 1 = 5. */
    @Test
    void testSearch_beforeTheFloor_thenNoClusterCounts() throws IOException {
        // given — the top 2 fills at once and everything after it is quiet
        Clusters clusters = clusters(
            new int[] { 100, 101 },
            new int[] { 900 },
            new int[] { 901 },
            new int[] { 902 },
            new int[] { 903 },
            new int[] { 904 },
            new int[] { 905 }
        );
        TopKnnCollector collector = new TopKnnCollector(2, Integer.MAX_VALUE);

        // when
        ClusterSearcher.Result result = ClusterSearcher.search(
            clusters,
            probes(7),
            ScanParams.of(QUERY),
            new EarlyStopParams(2, 1, 4),
            collector,
            null
        );

        // then
        assertEquals(5, result.scanned(), "clusters 2 and 3 are quiet but below the floor");
        assertEquals(2, result.unvisited());
    }

    /** Until the collector holds k, a cluster cannot count however little it adds, since everything is an insert. */
    @Test
    void testSearch_beforeTheCollectorIsFull_thenNoClusterCounts() throws IOException {
        // given — one vector per cluster, below epsilon 2, with k = 10 never reached
        Clusters clusters = clusters(new int[] { 900 }, new int[] { 901 }, new int[] { 902 }, new int[] { 903 });
        EarlyStopParams impatient = new EarlyStopParams(1, 2, 0);

        // when
        ClusterSearcher.Result notFull = ClusterSearcher.search(
            clusters,
            probes(4),
            ScanParams.of(QUERY),
            impatient,
            new TopKnnCollector(10, Integer.MAX_VALUE),
            null
        );
        ClusterSearcher.Result full = ClusterSearcher.search(
            clusters,
            probes(4),
            ScanParams.of(QUERY),
            impatient,
            new TopKnnCollector(2, Integer.MAX_VALUE),
            null
        );

        // then
        assertEquals(4, notFull.scanned(), "a collector short of k never lets the scan stop");
        assertEquals(2, full.scanned(), "the same clusters stop as soon as the collector is full");
    }

    /**
     * A collector that reports {@code true} for a vector that did not enter its top k (as MultiLeafKnnCollector does
     * when it syncs with the cross-segment queue) does not make a quiet cluster look useful: an insert counts only if
     * the vector also beat the threshold it was scored against.
     */
    @Test
    void testSearch_whenTheCollectorReportsTrueForANonCompetitiveVector_thenItIsNotAnInsert() throws IOException {
        // given — the top 2 fills, then the collector claims every later vector, none of which can compete
        Clusters clusters = clusters(new int[] { 100, 101 }, new int[] { 900 }, new int[] { 901 }, new int[] { 199 });
        KnnCollector alwaysTrue = new KnnCollector.Decorator(new TopKnnCollector(2, Integer.MAX_VALUE)) {
            @Override
            public boolean collect(int docId, float similarity) {
                super.collect(docId, similarity);
                return true;
            }
        };

        // when
        ClusterSearcher.Result result = ClusterSearcher.search(clusters, probes(4), ScanParams.of(QUERY), P2, alwaysTrue, null);

        // then
        assertEquals(3, result.scanned(), "two quiet clusters after the fill, then stop, as with an honest collector");
    }

    /**
     * A collector whose threshold starts above -inf (seeded with a minimum score of its own) does not look full before
     * this scan has collected k: the stop waits for k collects as well as the threshold.
     */
    @Test
    void testSearch_whenTheCollectorThresholdStartsAboveMinusInfinity_thenItIsNotFullBeforeK() throws IOException {
        // given — k = 5 never reached; every vector beats the seeded threshold but adds only 1 < epsilon 2
        Clusters clusters = clusters(new int[] { 900 }, new int[] { 901 }, new int[] { 902 }, new int[] { 903 });
        KnnCollector seeded = new KnnCollector.Decorator(new TopKnnCollector(5, Integer.MAX_VALUE)) {
            @Override
            public float minCompetitiveSimilarity() {
                return Math.max(super.minCompetitiveSimilarity(), 0.001f);
            }
        };

        // when
        ClusterSearcher.Result result = ClusterSearcher.search(
            clusters,
            probes(4),
            ScanParams.of(QUERY),
            new EarlyStopParams(1, 2, 0),
            seeded,
            null
        );

        // then
        assertEquals(4, result.scanned());
    }

    /**
     * Under a filter, a cluster whose accepted candidates are all pruned by the threshold (none scored, none collected)
     * still had candidates to score, so it counts: the quietest clusters are exactly these. The stub, like
     * BlockPostingScorer, applies the membership test before any score pruning.
     */
    @Test
    void testSearch_whenAFilteredClusterIsPrunedWhole_thenItStillCounts() throws IOException {
        // given
        Clusters clusters = clusters(new int[] { 100, 101 }, new int[] { 900 }, new int[] { 901 }, new int[] { 199 });
        when(clusters.scan()).thenReturn(
            (cluster, scanParams, acceptedOrds) -> new PruningStubScorer(
                new int[][] { { 100, 101 }, { 900 }, { 901 }, { 199 } }[cluster.ordinal()],
                acceptedOrds
            )
        );
        FixedBitSet accepted = new FixedBitSet(1024);
        accepted.set(0, 1024);
        TopKnnCollector collector = new TopKnnCollector(2, Integer.MAX_VALUE);

        // when
        ClusterSearcher.Result result = ClusterSearcher.search(clusters, probes(4), ScanParams.of(QUERY), P2, collector, accepted);

        // then
        assertEquals(3, result.scanned());
        assertEquals(2, collector.visitedCount(), "the two quiet clusters scored nothing at all");
    }

    /**
     * The collector must report a finite threshold (it holds k) before a quiet cluster counts: a collector that takes
     * k vectors but never raises its threshold (a diversifying collector replacing a parent's best child does this)
     * never lets the scan stop, even once this scan's own collect count has reached k.
     */
    @Test
    void testSearch_whenTheCollectorTakesKButItsThresholdStaysAtMinusInfinity_thenNeverStops() throws IOException {
        // given — the top 2 "fills" by count in the first cluster, then four quiet clusters
        Clusters clusters = clusters(new int[] { 100, 101 }, new int[] { 900 }, new int[] { 901 }, new int[] { 902 }, new int[] { 903 });
        KnnCollector neverFull = new KnnCollector.Decorator(new TopKnnCollector(2, Integer.MAX_VALUE)) {
            @Override
            public float minCompetitiveSimilarity() {
                return Float.NEGATIVE_INFINITY;
            }
        };

        // when
        ClusterSearcher.Result result = ClusterSearcher.search(clusters, probes(5), ScanParams.of(QUERY), P2, neverFull, null);

        // then
        assertEquals(5, result.scanned(), "without the threshold test the scan would stop after the third cluster");
        assertEquals(0, result.unvisited());
    }

    /** A stop on the last probe saved nothing, so it is not counted as one. */
    @Test
    void testSearch_whenTheStopFiresOnTheLastProbe_thenItIsNotCounted() throws IOException {
        // given — the second quiet cluster is the last probe
        Clusters clusters = clusters(new int[] { 100, 101 }, new int[] { 900 }, new int[] { 901 });
        long stops0 = ClusterANNStat.SCAN_EARLY_STOP.getCount();

        // when
        ClusterSearcher.Result result = ClusterSearcher.search(
            clusters,
            probes(3),
            ScanParams.of(QUERY),
            P2,
            new TopKnnCollector(2, Integer.MAX_VALUE),
            null
        );

        // then
        assertEquals(3, result.scanned());
        assertEquals(0, result.unvisited());
        assertEquals(0, ClusterANNStat.SCAN_EARLY_STOP.getCount() - stops0);
    }

    // ---------------------------------------------------------------- what counts under a filter

    /** A cluster the filter emptied is no evidence the query ran out of neighbours, so it does not count. */
    @Test
    void testSearch_whenTheFilterEmptiesClusters_thenTheyDoNotCount() throws IOException {
        // given — clusters 2-4 hold only ordinals the filter rejects; 5 and 6 are accepted but quiet
        Clusters clusters = clusters(
            new int[] { 100, 101 },
            new int[] { 300 },
            new int[] { 301 },
            new int[] { 302 },
            new int[] { 900 },
            new int[] { 901 },
            new int[] { 199 }
        );
        FixedBitSet accepted = new FixedBitSet(1024);
        accepted.set(0, 1024);
        accepted.clear(300);
        accepted.clear(301);
        accepted.clear(302);
        TopKnnCollector collector = new TopKnnCollector(2, Integer.MAX_VALUE);

        // when
        ClusterSearcher.Result result = ClusterSearcher.search(clusters, probes(7), ScanParams.of(QUERY), P2, collector, accepted);

        // then
        assertEquals(6, result.scanned(), "without the rule the scan would stop after the third cluster");
        assertEquals(1, result.unvisited());
    }

    /**
     * Under a filter, a cluster whose accepted members were all scored already (SOAR copies, deduped) had nothing to
     * score either, so it does not count; the rule reads the composed membership test, filter and dedup together.
     */
    @Test
    void testSearch_whenAFilteredClusterHoldsOnlySoarCopies_thenItDoesNotCount() throws IOException {
        // given — clusters 2 and 3 hold only copies of ordinals cluster 1 already scored
        Clusters clusters = clusters(
            new int[] { 100, 101, 900, 901 },
            new int[] { 900 },
            new int[] { 901 },
            new int[] { 902 },
            new int[] { 903 },
            new int[] { 199 }
        );
        FixedBitSet accepted = new FixedBitSet(1024);
        accepted.set(0, 1024);
        TopKnnCollector collector = new TopKnnCollector(2, Integer.MAX_VALUE);

        // when
        ClusterSearcher.Result result = ClusterSearcher.search(clusters, probes(6), ScanParams.of(QUERY), P2, collector, accepted);

        // then
        assertEquals(5, result.scanned(), "the two copy-only clusters are skipped by the count");
    }

    /** Unfiltered, every non-empty cluster counts, a copy-only one included. */
    @Test
    void testSearch_whenUnfilteredClusterHoldsOnlySoarCopies_thenItStillCounts() throws IOException {
        // given — the same layout, no filter
        Clusters clusters = clusters(
            new int[] { 100, 101, 900, 901 },
            new int[] { 900 },
            new int[] { 901 },
            new int[] { 902 },
            new int[] { 903 },
            new int[] { 199 }
        );
        TopKnnCollector collector = new TopKnnCollector(2, Integer.MAX_VALUE);

        // when
        ClusterSearcher.Result result = ClusterSearcher.search(clusters, probes(6), ScanParams.of(QUERY), P2, collector, null);

        // then
        assertEquals(3, result.scanned());
    }

    /** An empty cluster is skipped before anything is counted, as it always was. */
    @Test
    void testSearch_whenAClusterIsEmpty_thenItNeitherCountsNorIsScanned() throws IOException {
        // given
        Clusters clusters = clusters(
            new int[] { 100, 101 },
            new int[0],
            new int[] { 900 },
            new int[0],
            new int[] { 901 },
            new int[] { 199 }
        );
        TopKnnCollector collector = new TopKnnCollector(2, Integer.MAX_VALUE);
        long fractionTotal0 = ClusterANNStat.SCAN_EARLY_STOP_SCANNED_BPS.getTotal();

        // when
        ClusterSearcher.Result result = ClusterSearcher.search(clusters, probes(6), ScanParams.of(QUERY), P2, collector, null);

        // then
        assertEquals(3, result.scanned(), "the two empties are not scanned and not counted");
        assertEquals(1, result.unvisited());
        assertEquals(
            8_333,
            ClusterANNStat.SCAN_EARLY_STOP_SCANNED_BPS.getTotal() - fractionTotal0,
            "5 of 6 probes walked, the empties among them"
        );
    }

    /** Watching the membership test for the count does not change what it spans: the filter's length, as before. */
    @Test
    void testSearch_whenOnAndFiltered_thenTheMembershipTestStillSpansTheFilter() throws IOException {
        // given
        Clusters clusters = clusters(new int[] { 100 });
        List<Integer> lengths = new ArrayList<>();
        when(clusters.scan()).thenReturn((cluster, scanParams, acceptedOrds) -> {
            lengths.add(acceptedOrds.length());
            return new StubScorer(new int[] { 100 }, acceptedOrds);
        });
        FixedBitSet accepted = new FixedBitSet(2048);
        accepted.set(100);

        // when
        ClusterSearcher.search(clusters, probes(1), ScanParams.of(QUERY), P2, new TopKnnCollector(1, Integer.MAX_VALUE), accepted);

        // then
        assertEquals(List.of(2048), lengths);
    }

    // ---------------------------------------------------------------- the collector's own stop

    /** Once the collector says it is done (a visit limit, a timeout), the scan stops at the end of that cluster. */
    @Test
    void testSearch_whenTheCollectorTerminatesEarly_thenStops() throws IOException {
        // given — a visit limit of 3 is reached by the end of the second cluster
        Clusters clusters = clusters(new int[] { 100, 101 }, new int[] { 150 }, new int[] { 160 }, new int[] { 170 });
        TopKnnCollector collector = new TopKnnCollector(2, 3);
        EarlyStopParams neverQuiet = new EarlyStopParams(100, 1, 0);
        long stops0 = ClusterANNStat.SCAN_EARLY_STOP.getCount();

        // when
        ClusterSearcher.Result result = ClusterSearcher.search(clusters, probes(4), ScanParams.of(QUERY), neverQuiet, collector, null);

        // then
        assertEquals(2, result.scanned());
        assertEquals(2, result.unvisited());
        assertTrue(collector.earlyTerminated());
        assertEquals(0, ClusterANNStat.SCAN_EARLY_STOP.getCount() - stops0, "the patience stop did not fire");
    }

    /** With the stop off, the collector's early termination is not consulted, exactly as before. */
    @Test
    void testSearch_whenOffAndTheCollectorTerminatesEarly_thenScansOnAsBefore() throws IOException {
        // given
        Clusters clusters = clusters(new int[] { 100, 101 }, new int[] { 150 }, new int[] { 160 }, new int[] { 170 });
        TopKnnCollector collector = new TopKnnCollector(2, 3);

        // when
        ClusterSearcher.Result result = ClusterSearcher.search(
            clusters,
            probes(4),
            ScanParams.of(QUERY),
            EarlyStopParams.OFF,
            collector,
            null
        );

        // then
        assertEquals(4, result.scanned());
        assertEquals(0, result.unvisited());
    }

    // ---------------------------------------------------------------- off is today

    /**
     * With the stop off, the walk makes the same collector calls, in the same order, as a verbatim copy of the walk
     * before the stop existed, over seeded random layouts with SOAR copies, filters, empties and visit limits. Both
     * entry points are checked: the old signature and the new one given {@link EarlyStopParams#OFF}.
     */
    @Test
    void testSearch_whenOff_thenCallForCallIdenticalToBefore() throws IOException {
        Random random = new Random(20261119L);
        for (int round = 0; round < 200; round++) {
            // given
            int numClusters = 1 + random.nextInt(12);
            int[][] postings = new int[numClusters][];
            for (int c = 0; c < numClusters; c++) {
                postings[c] = new int[random.nextInt(6)];
                for (int i = 0; i < postings[c].length; i++) {
                    postings[c][i] = random.nextInt(40) + (random.nextBoolean() ? 100 : 900);
                }
            }
            int[] probes = new int[random.nextInt(numClusters + 3)];
            for (int i = 0; i < probes.length; i++) {
                probes[i] = random.nextInt(numClusters);
            }
            FixedBitSet filter = null;
            if (random.nextBoolean()) {
                filter = new FixedBitSet(1024);
                for (int ord = 0; ord < 1024; ord++) {
                    if (random.nextInt(3) != 0) {
                        filter.set(ord);
                    }
                }
            }
            int k = 1 + random.nextInt(5);
            int visitLimit = random.nextBoolean() ? Integer.MAX_VALUE : 1 + random.nextInt(8);
            Clusters clusters = clusters(postings);

            // when
            CallLog before = new CallLog(new TopKnnCollector(k, visitLimit));
            int beforeScanned = searchAsBefore(clusters, probes, ScanParams.of(QUERY), before, filter);
            CallLog oldSignature = new CallLog(new TopKnnCollector(k, visitLimit));
            int oldScanned = ClusterSearcher.search(clusters, probes, ScanParams.of(QUERY), oldSignature, filter);
            CallLog off = new CallLog(new TopKnnCollector(k, visitLimit));
            ClusterSearcher.Result offResult = ClusterSearcher.search(
                clusters,
                probes,
                ScanParams.of(QUERY),
                EarlyStopParams.OFF,
                off,
                filter
            );

            // then
            String where = "round " + round;
            assertEquals(beforeScanned, oldScanned, where);
            assertEquals(beforeScanned, offResult.scanned(), where);
            assertEquals(0, offResult.unvisited(), where);
            assertEquals(before.calls, oldSignature.calls, where);
            assertEquals(before.calls, off.calls, where);
            assertEquals(List.of(before.topDocs().scoreDocs).toString(), List.of(off.topDocs().scoreDocs).toString(), where);
        }
    }

    /** Off records nothing the stop owns; on, every scan records its scanned-against-planned fraction. */
    @Test
    void testSearch_thenTheStopsStatsAreRecordedOnlyWhenOn() throws IOException {
        // given
        Clusters clusters = clusters(new int[] { 100, 101 }, new int[] { 900 }, new int[] { 901 }, new int[] { 199 });
        long stops0 = ClusterANNStat.SCAN_EARLY_STOP.getCount();
        long fractions0 = ClusterANNStat.SCAN_EARLY_STOP_SCANNED_BPS.getCount();
        long fractionTotal0 = ClusterANNStat.SCAN_EARLY_STOP_SCANNED_BPS.getTotal();

        // when — off
        ClusterSearcher.search(
            clusters,
            probes(4),
            ScanParams.of(QUERY),
            EarlyStopParams.OFF,
            new TopKnnCollector(2, Integer.MAX_VALUE),
            null
        );

        // then
        assertEquals(0, ClusterANNStat.SCAN_EARLY_STOP.getCount() - stops0);
        assertEquals(0, ClusterANNStat.SCAN_EARLY_STOP_SCANNED_BPS.getCount() - fractions0);

        // when — on, stopping after 3 of 4
        ClusterSearcher.search(clusters, probes(4), ScanParams.of(QUERY), P2, new TopKnnCollector(2, Integer.MAX_VALUE), null);

        // then
        assertEquals(1, ClusterANNStat.SCAN_EARLY_STOP.getCount() - stops0);
        assertEquals(1, ClusterANNStat.SCAN_EARLY_STOP_SCANNED_BPS.getCount() - fractions0);
        assertEquals(7_500, ClusterANNStat.SCAN_EARLY_STOP_SCANNED_BPS.getTotal() - fractionTotal0, "3 of 4 probes");
    }

    /** epsilon and floor left AUTO are derived per query: from k, and from the field's cluster count. */
    @Test
    void testSearch_whenAuto_thenTheFloorIsThePlansMinProbes() throws IOException {
        // given — 12 clusters: PlanParams.of(12) has minProbes min(max(10, 3), 9) = 9, so no stop before cluster 9
        int[][] postings = new int[12][];
        postings[0] = new int[] { 100, 101 };
        for (int c = 1; c < postings.length; c++) {
            postings[c] = new int[] { 900 + c };
        }
        Clusters clusters = clusters(postings);
        assertEquals(9, PlanParams.of(12).minProbes());

        // when
        ClusterSearcher.Result result = ClusterSearcher.search(
            clusters,
            probes(12),
            ScanParams.of(QUERY),
            EarlyStopParams.PROVISIONAL,
            new TopKnnCollector(2, Integer.MAX_VALUE),
            null
        );

        // then — clusters 9, 10, 11 are the three quiet ones in a row that count
        assertEquals(11, result.scanned());
        assertEquals(1, result.unvisited());
    }

    /** The stop's own setting is required like the other three; null is not a stand-in for OFF. */
    @Test
    void testSearch_whenEarlyStopIsNull_thenThrows() throws IOException {
        Clusters clusters = clusters(new int[] { 100 });
        assertThrows(
            IllegalArgumentException.class,
            () -> ClusterSearcher.search(clusters, probes(1), ScanParams.of(QUERY), null, new TopKnnCollector(1, Integer.MAX_VALUE), null)
        );
    }

    /** A field with no clusters, or an empty plan, scans nothing whether or not the stop is on. */
    @Test
    void testSearch_whenThereIsNothingToScan_thenReturnsEmpty() throws IOException {
        Clusters none = mock(Clusters.class);
        when(none.numClusters()).thenReturn(0);
        assertEquals(
            new ClusterSearcher.Result(0, 0),
            ClusterSearcher.search(none, probes(2), ScanParams.of(QUERY), P2, new TopKnnCollector(1, Integer.MAX_VALUE), null)
        );
        assertEquals(
            new ClusterSearcher.Result(0, 0),
            ClusterSearcher.search(
                clusters(new int[] { 100 }),
                new int[0],
                ScanParams.of(QUERY),
                P2,
                new TopKnnCollector(1, Integer.MAX_VALUE),
                null
            )
        );
    }

    // ---------------------------------------------------------------- helpers

    /** The walk as it was before the early stop existed, copied verbatim, as the oracle for the OFF test. */
    private static int searchAsBefore(Clusters clusters, int[] probes, ScanParams params, KnnCollector collector, Bits acceptedOrds)
        throws IOException {
        if (clusters.numClusters() == 0 || probes.length == 0) {
            return 0;
        }
        BitSet visited = new BitSet(clusters.numVectors());
        Bits wanted = new Bits() {
            @Override
            public boolean get(int ord) {
                return (acceptedOrds == null || acceptedOrds.get(ord)) && !visited.get(ord);
            }

            @Override
            public int length() {
                return acceptedOrds != null ? acceptedOrds.length() : clusters.numVectors();
            }
        };
        ClusterScan scan = clusters.scan();
        int scanned = 0;
        for (int probe : probes) {
            Cluster cluster = clusters.get(probe);
            if (cluster.size() != 0) {
                PostingScorer scorer = scan.scorer(cluster, params, wanted);
                while (scorer.advance(collector.minCompetitiveSimilarity())) {
                    int ord = scorer.ord();
                    visited.set(ord);
                    collector.incVisitedCount(1);
                    collector.collect(ord, scorer.score());
                }
                scanned++;
            }
        }
        return scanned;
    }

    private static int[] probes(int n) {
        int[] probes = new int[n];
        for (int i = 0; i < n; i++) {
            probes[i] = i;
        }
        return probes;
    }

    /** Clusters 0..n-1 holding the given ordinals, in probe order. */
    private static Clusters clusters(int[]... postings) throws IOException {
        Clusters clusters = mock(Clusters.class);
        when(clusters.numClusters()).thenReturn(postings.length);
        when(clusters.numVectors()).thenReturn(1024);
        for (int c = 0; c < postings.length; c++) {
            Cluster cluster = mock(Cluster.class);
            when(cluster.ordinal()).thenReturn(c);
            when(cluster.size()).thenReturn(postings[c].length);
            when(clusters.get(c)).thenReturn(cluster);
        }
        when(clusters.scan()).thenReturn((cluster, scanParams, acceptedOrds) -> new StubScorer(postings[cluster.ordinal()], acceptedOrds));
        return clusters;
    }

    /** Hands out every ordinal the membership test accepts, consulting it once per candidate as the real scorer does. */
    private static final class StubScorer implements PostingScorer {

        private final int[] ordinals;
        private final Bits acceptedOrds;
        private int position = -1;

        private StubScorer(int[] ordinals, Bits acceptedOrds) {
            this.ordinals = ordinals;
            this.acceptedOrds = acceptedOrds;
        }

        @Override
        public boolean advance(float minCompetitiveSimilarity) {
            while (++position < ordinals.length) {
                if (acceptedOrds == null || acceptedOrds.get(ordinals[position])) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public int ord() {
            return ordinals[position];
        }

        @Override
        public float score() {
            return SCORES.get(ordinals[position]);
        }
    }

    /** Like {@link StubScorer}, but drops candidates at or under the threshold after the membership test, as a block scorer does. */
    private static final class PruningStubScorer implements PostingScorer {

        private final int[] ordinals;
        private final Bits acceptedOrds;
        private int position = -1;

        private PruningStubScorer(int[] ordinals, Bits acceptedOrds) {
            this.ordinals = ordinals;
            this.acceptedOrds = acceptedOrds;
        }

        @Override
        public boolean advance(float minCompetitiveSimilarity) {
            while (++position < ordinals.length) {
                if (acceptedOrds.get(ordinals[position]) && score() > minCompetitiveSimilarity) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public int ord() {
            return ordinals[position];
        }

        @Override
        public float score() {
            return SCORES.get(ordinals[position]);
        }
    }

    /** Logs every collector call the walk makes, with its arguments and answer, around a real collector. */
    private static final class CallLog extends KnnCollector.Decorator {

        private final List<String> calls = new ArrayList<>();

        private CallLog(KnnCollector collector) {
            super(collector);
        }

        @Override
        public boolean earlyTerminated() {
            boolean answer = super.earlyTerminated();
            calls.add("earlyTerminated=" + answer);
            return answer;
        }

        @Override
        public void incVisitedCount(int count) {
            calls.add("incVisitedCount(" + count + ")");
            super.incVisitedCount(count);
        }

        @Override
        public int k() {
            calls.add("k");
            return super.k();
        }

        @Override
        public boolean collect(int docId, float similarity) {
            boolean answer = super.collect(docId, similarity);
            calls.add("collect(" + docId + "," + similarity + ")=" + answer);
            return answer;
        }

        @Override
        public float minCompetitiveSimilarity() {
            float answer = super.minCompetitiveSimilarity();
            calls.add("minCompetitiveSimilarity=" + answer);
            return answer;
        }
    }
}
