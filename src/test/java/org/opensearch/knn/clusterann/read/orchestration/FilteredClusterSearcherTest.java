/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.read.Cluster;
import org.opensearch.knn.clusterann.read.PostingScorer;
import org.opensearch.knn.clusterann.read.ScanParams;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FilteredClusterSearcherTest {

    private static final float[] QUERY = { 0.1f, 0.2f, 0.3f, 0.4f };

    @Test
    void testSearchFiltered_whenClusterMatchesAreBelowK_thenUsesExactScoring() throws IOException {
        FakeCluster cluster = new FakeCluster(0, new int[] { 10, 11 }, new float[] { 0.2f, 0.1f });
        FakeClusterSource source = new FakeClusterSource(Map.of(0, cluster), 4);
        FixedBitSet accepted = accepted(10, 11);
        RecordingCollector collector = new RecordingCollector(2, Float.NEGATIVE_INFINITY);

        ClusterSearcher.searchFiltered(
            source,
            new int[] { 0 },
            ScanParams.of(QUERY),
            collector,
            accepted,
            new int[] { 1 },
            new MapRandomVectorScorer(Map.of(10, 0.9f, 11, 0.8f))
        );

        assertEquals(0, cluster.adcCalls);
        assertEquals(1, cluster.exactCalls);
        assertEquals(List.of(10, 11), collector.collectedDocs);
        assertEquals(List.of(0.9f, 0.8f), collector.collectedScores);
    }

    @Test
    void testSearchFiltered_whenDocsScoredReachThreeKWithThreshold_thenStopsBeforeLaterClusters() throws IOException {
        FakeCluster first = new FakeCluster(0, new int[] { 10, 11, 12 }, new float[] { 0.9f, 0.8f, 0.7f });
        FakeCluster second = new FakeCluster(1, new int[] { 20 }, new float[] { 0.6f });
        FakeClusterSource source = new FakeClusterSource(Map.of(0, first, 1, second), 4);

        int scanned = ClusterSearcher.searchFiltered(
            source,
            new int[] { 0, 1 },
            ScanParams.of(QUERY),
            new RecordingCollector(1, 0.5f),
            accepted(10, 11, 12, 20),
            new int[] { 3, 3 },
            new MapRandomVectorScorer(Map.of())
        );

        assertEquals(1, scanned);
        assertEquals(1, first.adcCalls);
        assertEquals(0, second.adcCalls);
    }

    @Test
    void testSearchFiltered_whenSelectivityMakesAClusterTooSparse_thenSkipsIt() throws IOException {
        FakeCluster sparse = new FakeCluster(0, new int[] { 10 }, new float[] { 0.9f });
        FakeCluster dense = new FakeCluster(1, new int[] { 20, 21, 22, 23, 24 }, new float[] { 0.8f, 0.7f, 0.6f, 0.5f, 0.4f });
        FakeClusterSource source = new FakeClusterSource(Map.of(0, sparse, 1, dense), 10);

        int scanned = ClusterSearcher.searchFiltered(
            source,
            new int[] { 0, 1 },
            ScanParams.of(QUERY),
            new RecordingCollector(2, Float.NEGATIVE_INFINITY),
            accepted(10, 20),
            new int[] { 2, 5 },
            new MapRandomVectorScorer(Map.of())
        );

        assertEquals(1, scanned);
        assertEquals(0, sparse.adcCalls);
        assertEquals(0, sparse.exactCalls);
        assertEquals(1, dense.adcCalls);
    }

    @Test
    void testContributionTerminationHelpers_thenModelTwoNonImprovingClusters() {
        int consecutive = ClusterSearcher.nextConsecutiveNonImproving(0, Float.NEGATIVE_INFINITY, 0.5f);
        assertEquals(1, consecutive);

        consecutive = ClusterSearcher.nextConsecutiveNonImproving(consecutive, 0.5f, 0.5f);
        assertTrue(ClusterSearcher.shouldStopAfterContribution(3, 4, 1, consecutive, 0.5f));
    }

    private static FixedBitSet accepted(int... ords) {
        FixedBitSet accepted = new FixedBitSet(64);
        for (int ord : ords) {
            accepted.set(ord);
        }
        return accepted;
    }

    private static final class FakeClusterSource implements ClusterSearcher.ClusterSource {

        private final Map<Integer, FakeCluster> clusters;
        private final ClusterScan scan = (cluster, scanParams, acceptedOrds) -> ((FakeCluster) cluster).adcScorer(acceptedOrds);
        private final int numVectors;

        private FakeClusterSource(Map<Integer, FakeCluster> clusters, int numVectors) {
            this.clusters = new LinkedHashMap<>(clusters);
            this.numVectors = numVectors;
        }

        @Override
        public int numClusters() {
            return clusters.size();
        }

        @Override
        public int numVectors() {
            return numVectors;
        }

        @Override
        public Cluster get(int ordinal) {
            return clusters.get(ordinal);
        }

        @Override
        public ClusterScan scan() {
            return scan;
        }
    }

    private static final class FakeCluster implements Cluster {

        private final int ordinal;
        private final int[] ordinals;
        private final float[] adcScores;
        private int adcCalls;
        private int exactCalls;

        private FakeCluster(int ordinal, int[] ordinals, float[] adcScores) {
            this.ordinal = ordinal;
            this.ordinals = ordinals;
            this.adcScores = adcScores;
        }

        @Override
        public int ordinal() {
            return ordinal;
        }

        @Override
        public int size() {
            return ordinals.length;
        }

        @Override
        public void prefetch(boolean partial) {}

        @Override
        public PostingScorer scorer(ScanContext scanContext, Bits acceptedOrds) {
            throw new UnsupportedOperationException("the filtered path uses the ClusterScan stub directly");
        }

        @Override
        public PostingScorer exactScorer(RandomVectorScorer scorer, Bits acceptedOrds) {
            exactCalls++;
            return new ExactArrayScorer(ordinals, acceptedOrds, scorer);
        }

        @Override
        public ScanContext prepareScan(ScanParams scanParams) {
            return new ScanContext() {
            };
        }

        @Override
        public long ramBytesUsed() {
            return 0L;
        }

        private PostingScorer adcScorer(Bits acceptedOrds) {
            adcCalls++;
            return new ArrayScorer(ordinals, adcScores, acceptedOrds);
        }
    }

    private static class ArrayScorer implements PostingScorer {

        private final int[] ordinals;
        private final float[] scores;
        private final Bits acceptedOrds;
        private int index = -1;

        private ArrayScorer(int[] ordinals, float[] scores, Bits acceptedOrds) {
            this.ordinals = ordinals;
            this.scores = scores;
            this.acceptedOrds = acceptedOrds;
        }

        @Override
        public boolean advance(float minCompetitiveSimilarity) {
            while (++index < ordinals.length) {
                if ((acceptedOrds == null || acceptedOrds.get(ordinals[index])) && scores[index] >= minCompetitiveSimilarity) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public int ord() {
            return ordinals[index];
        }

        @Override
        public float score() {
            return scores[index];
        }
    }

    private static final class ExactArrayScorer implements PostingScorer {

        private final int[] ordinals;
        private final Bits acceptedOrds;
        private final RandomVectorScorer scorer;
        private int index = -1;
        private float score;

        private ExactArrayScorer(int[] ordinals, Bits acceptedOrds, RandomVectorScorer scorer) {
            this.ordinals = ordinals;
            this.acceptedOrds = acceptedOrds;
            this.scorer = scorer;
        }

        @Override
        public boolean advance(float minCompetitiveSimilarity) throws IOException {
            while (++index < ordinals.length) {
                if (acceptedOrds != null && acceptedOrds.get(ordinals[index]) == false) {
                    continue;
                }
                score = scorer.score(ordinals[index]);
                if (score >= minCompetitiveSimilarity) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public int ord() {
            return ordinals[index];
        }

        @Override
        public float score() {
            return score;
        }
    }

    private static final class MapRandomVectorScorer implements RandomVectorScorer {

        private final Map<Integer, Float> scores;

        private MapRandomVectorScorer(Map<Integer, Float> scores) {
            this.scores = scores;
        }

        @Override
        public float score(int ord) {
            return scores.getOrDefault(ord, 0f);
        }

        @Override
        public int maxOrd() {
            return Integer.MAX_VALUE;
        }
    }

    private static final class RecordingCollector implements KnnCollector {

        private final int k;
        private final float threshold;
        private final List<Integer> collectedDocs = new ArrayList<>();
        private final List<Float> collectedScores = new ArrayList<>();
        private int visited;

        private RecordingCollector(int k, float threshold) {
            this.k = k;
            this.threshold = threshold;
        }

        @Override
        public boolean earlyTerminated() {
            return false;
        }

        @Override
        public void incVisitedCount(int count) {
            visited += count;
        }

        @Override
        public long visitedCount() {
            return visited;
        }

        @Override
        public long visitLimit() {
            return Long.MAX_VALUE;
        }

        @Override
        public int k() {
            return k;
        }

        @Override
        public boolean collect(int docId, float similarity) {
            collectedDocs.add(docId);
            collectedScores.add(similarity);
            return true;
        }

        @Override
        public float minCompetitiveSimilarity() {
            return collectedDocs.size() >= k ? threshold : Float.NEGATIVE_INFINITY;
        }

        @Override
        public org.apache.lucene.search.TopDocs topDocs() {
            throw new UnsupportedOperationException("not needed by these tests");
        }

        @Override
        public org.apache.lucene.search.knn.KnnSearchStrategy getSearchStrategy() {
            return null;
        }
    }
}
