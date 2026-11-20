/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TimeLimitingKnnCollectorManager;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.opensearch.knn.clusterann.read.Clusters;
import org.opensearch.knn.clusterann.read.ScanParams;
import org.opensearch.knn.clusterann.read.orchestration.CentroidPlanner;
import org.opensearch.knn.clusterann.read.orchestration.ClusterSearcher;
import org.opensearch.knn.clusterann.read.orchestration.EarlyStopExperiment;
import org.opensearch.knn.clusterann.read.orchestration.EarlyStopParams;
import org.opensearch.knn.clusterann.read.orchestration.FilterDigest;
import org.opensearch.knn.clusterann.read.orchestration.PlanParams;
import org.opensearch.knn.clusterann.read.orchestration.ProbeCurveExperiment;
import org.opensearch.knn.clusterann.read.orchestration.ShardPlanExperiment;
import org.opensearch.knn.clusterann.read.orchestration.ShardPlanStrategy;
import org.opensearch.knn.clusterann.read.orchestration.ShardPlanner;
import org.opensearch.knn.clusterann.stats.ClusterANNStat;
import org.opensearch.knn.index.codec.clusterann.ClusterANN1030TestCodec;
import org.opensearch.knn.index.codec.clusterann.vectorformat1030.KNN1030ClusterANNVectorsReader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * The shard plan through the real reader and weight: the plan's counts reach the reader on the collector, the reader
 * probes exactly its first {@code maxProbes_s}, a segment given nothing returns nothing without falling back to exact
 * search, and a one-segment shard returns what per-segment planning returns.
 *
 * <p>One index for the whole class: a near segment of 40,000 vectors (about 78 clusters at the writer's 512 per
 * cluster), so C_total exceeds R and the cut really cuts, plus two smaller segments, one of them far from every query
 * so the plan gives it no cluster.
 */
public class ClusterANNShardPlanTests extends LuceneTestCase {

    private static final String FIELD = "v";
    private static final String PARITY = "parity";
    private static final int DIM = 4;
    private static final int[] SEGMENT_SIZES = { 40_000, 3_000, 2_000 };
    private static final float FAR = 1_000f;
    private static final int K = 10;

    private Directory directory;
    private DirectoryReader reader;
    private IndexSearcher searcher;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        directory = newFSDirectory(createTempDir("shard-plan"));
        final Random random = new Random(11);
        final IndexWriterConfig config = new IndexWriterConfig().setCodec(new ClusterANN1030TestCodec(4))
            .setMergePolicy(NoMergePolicy.INSTANCE)
            .setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setRAMBufferSizeMB(256)
            .setUseCompoundFile(false);
        int doc = 0;
        try (IndexWriter writer = new IndexWriter(directory, config)) {
            for (int s = 0; s < SEGMENT_SIZES.length; s++) {
                final float offset = s == SEGMENT_SIZES.length - 1 ? FAR : 0f;
                for (int i = 0; i < SEGMENT_SIZES[s]; i++) {
                    final float[] vector = new float[DIM];
                    for (int d = 0; d < DIM; d++) {
                        vector[d] = offset + random.nextFloat();
                    }
                    final Document document = new Document();
                    document.add(new KnnFloatVectorField(FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                    document.add(new StringField(PARITY, doc++ % 2 == 0 ? "even" : "odd", Field.Store.NO));
                    writer.addDocument(document);
                }
                writer.flush();
            }
        }
        reader = DirectoryReader.open(directory);
        searcher = new IndexSearcher(reader);
        assertEquals(SEGMENT_SIZES.length, reader.leaves().size());
    }

    @Override
    public void tearDown() throws Exception {
        reader.close();
        directory.close();
        super.tearDown();
    }

    /** Lucene's collector decorators (time limit, shared threshold) forward the strategy the weight built with. */
    public void testStrategy_reachesTheReaderThroughTheCollectorChain() throws IOException {
        final LeafReaderContext leaf = reader.leaves().get(0);
        final ShardPlanStrategy strategy = new ShardPlanStrategy(clusters(leaf), new PlanParams(1, 3), new PlanParams(10, 72), null);
        final TimeLimitingKnnCollectorManager manager = new TimeLimitingKnnCollectorManager(
            new ClusterANNCollectorManager(K, null, searcher),
            null
        );

        final KnnCollector collector = manager.newCollector(Integer.MAX_VALUE, strategy, leaf);

        assertSame(strategy, collector.getSearchStrategy());
    }

    /** The weight plans once for the shard and hands each leaf the planner's counts for it. */
    public void testWeight_handsEachLeafTheShardPlansCounts() throws IOException {
        final float[] query = query(1);
        for (final Query filter : new Query[] { null, new TermQuery(new Term(PARITY, "even")) }) {
            final ClusterANNWeight weight = weight(query, filter, ShardPlanner.Budget.CAPPED);
            final List<ShardPlanner.Segment> segments = new ArrayList<>();
            for (final LeafReaderContext leaf : reader.leaves()) {
                segments.add(clusterAnnReader(leaf).shardPlanSegment(FIELD, acceptDocs(leaf, filter)));
            }
            final ShardPlanner.Plan expected = ShardPlanner.plan(segments, query, K, ShardPlanner.Budget.CAPPED);
            assertTrue("the cut must cut for this test to mean anything", expected.params().maxProbes() < expected.totalClusters());

            for (final LeafReaderContext leaf : reader.leaves()) {
                final ClusterANNWeight.LeafPlan plan = weight.shardPlan(leaf).forLeaf(leaf.ord);
                final PlanParams segmentParams = expected.segmentParams()[leaf.ord];
                if (segmentParams.maxProbes() == 0) {
                    assertNull("leaf " + leaf.ord, plan.strategy());
                } else {
                    assertEquals("leaf " + leaf.ord, segmentParams, plan.strategy().segmentParams());
                    assertEquals(expected.params(), plan.strategy().shardParams());
                }
            }
            // One plan, shared by every leaf.
            assertSame(weight.shardPlan(reader.leaves().get(0)), weight.shardPlan(reader.leaves().get(1)));
        }
    }

    /**
     * Given counts, the reader probes exactly its own first maxProbes_s clusters and nothing else: its hits are the
     * scan of that list, for every count, filtered or not.
     */
    public void testReader_scansExactlyItsFirstMaxProbes() throws IOException {
        final LeafReaderContext leaf = reader.leaves().get(0);
        final KNN1030ClusterANNVectorsReader vectors = clusterAnnReader(leaf);
        final Clusters clusters = clusters(leaf);
        final float[] query = query(2);
        for (final Query filter : new Query[] { null, new TermQuery(new Term(PARITY, "odd")) }) {
            for (final int maxProbes : new int[] { 1, 2, 5, 13, clusters.numClusters() }) {
                final AcceptDocs acceptDocs = acceptDocs(leaf, filter);
                final PlanParams segmentParams = new PlanParams(0, maxProbes);
                final TopKnnCollector planned = new TopKnnCollector(
                    K,
                    Integer.MAX_VALUE,
                    new ShardPlanStrategy(clusters, segmentParams, new PlanParams(10, 72), null)
                );
                vectors.search(FIELD, query, planned, acceptDocs);

                final FilterDigest digest = FilterDigest.of(clusters, acceptDocs, () -> { throw new AssertionError("dense"); });
                final int[] probes = CentroidPlanner.planWithin(clusters, query, segmentParams, digest);
                assertEquals(Math.min(maxProbes, clusters.numClusters()), probes.length);
                final float[] rotated = new float[DIM];
                clusters.rotation().rotate(query, rotated);
                final TopKnnCollector expected = new TopKnnCollector(K, Integer.MAX_VALUE);
                ClusterSearcher.search(clusters, probes, ScanParams.of(rotated), expected, acceptDocs.bits());

                assertSameHits("maxProbes " + maxProbes + " filter " + filter, expected.topDocs(), planned.topDocs());
            }
        }
    }

    /** Counts are planned for one segment's clusters; a reader refuses counts planned for another. */
    public void testReader_refusesAPlanMadeForAnotherSegment() throws IOException {
        final KnnCollector collector = new TopKnnCollector(
            K,
            Integer.MAX_VALUE,
            new ShardPlanStrategy(clusters(reader.leaves().get(1)), new PlanParams(0, 2), new PlanParams(10, 72), null)
        );
        final LeafReaderContext leaf = reader.leaves().get(0);
        expectThrows(
            IllegalArgumentException.class,
            () -> clusterAnnReader(leaf).search(FIELD, query(3), collector, AcceptDocs.fromLiveDocs(null, leaf.reader().maxDoc()))
        );
    }

    /**
     * The far segment holds thousands of documents, filtered or not far more than k, but the plan gives it no
     * cluster: it returns no hits, and nothing falls back to exact search, under either budget.
     */
    public void testFarSegment_getsNoClusterAndReturnsNothingWithoutExactSearch() throws IOException {
        final LeafReaderContext far = reader.leaves().get(SEGMENT_SIZES.length - 1);
        for (final ShardPlanner.Budget budget : ShardPlanner.Budget.values()) {
            for (final Query filter : new Query[] { null, new TermQuery(new Term(PARITY, "even")) }) {
                final ClusterANNWeight weight = weight(query(4), filter, budget);
                final long fallbacks = ClusterANNStat.EXACT_FALLBACK.getTotal();
                final long requests = ClusterANNStat.QUERY_REQUESTS.getTotal();

                final ClusterANNWeight.LeafPlan plan = weight.shardPlan(far).forLeaf(far.ord);
                final TopDocs hits = weight.searchLeaf(far);

                assertNull(budget + " " + filter, plan.strategy());
                assertTrue(plan.acceptDocs().cost() > K);
                assertEquals(0, hits.scoreDocs.length);
                assertEquals("no exact search", fallbacks, ClusterANNStat.EXACT_FALLBACK.getTotal());
                assertEquals("the far segment is not even searched", requests, ClusterANNStat.QUERY_REQUESTS.getTotal());
            }
        }
        // And the shard still returns k hits, all from the near segments.
        final TopDocs hits = searcher.search(new ClusterANNQuery(FIELD, query(4), K, null, null, 0), K);
        assertEquals(K, hits.scoreDocs.length);
        for (final ScoreDoc hit : hits.scoreDocs) {
            assertTrue(hit.doc < far.docBase);
        }
    }

    /** Uncapped R never probes fewer clusters than H, which never probes more than R. */
    public void testBudgets_ruleProbesAtLeastAsManyAsCapped() throws IOException {
        final float[] query = query(5);
        final ShardPlanner.Plan capped = weight(query, null, ShardPlanner.Budget.CAPPED).shardPlan(reader.leaves().get(0)).plan();
        final ShardPlanner.Plan rule = weight(query, null, ShardPlanner.Budget.RULE).shardPlan(reader.leaves().get(0)).plan();
        assertEquals(rule.params().maxProbes(), rule.probeCount());
        assertTrue(capped.probeCount() <= rule.probeCount());
    }

    /** Concurrent segment search, one leaf per slice: every leaf waits on one plan, and the hits are the sequential ones. */
    public void testConcurrentSearch_sameHitsAsSequential() throws Exception {
        final ExecutorService executor = Executors.newFixedThreadPool(SEGMENT_SIZES.length);
        try {
            final IndexSearcher concurrent = perLeafSlices(executor);
            for (int q = 0; q < 10; q++) {
                for (final Query filter : new Query[] { null, new TermQuery(new Term(PARITY, "odd")) }) {
                    final ClusterANNQuery query = new ClusterANNQuery(FIELD, query(100 + q), K, filter, null, 0);
                    assertSameHits("query " + q, searcher.search(query, K), concurrent.search(query, K));
                }
            }
        } finally {
            executor.shutdownNow();
        }
    }

    /**
     * A planning failure reaches the caller as itself. Every leaf shares the plan, so every waiting leaf sees the same
     * failure; rethrowing one instance from several tasks makes Lucene's result merge fail on self-suppression
     * instead.
     */
    public void testConcurrentSearch_whenPlanningFails_thenTheCauseSurfaces() throws Exception {
        final ExecutorService executor = Executors.newFixedThreadPool(SEGMENT_SIZES.length);
        try {
            final IndexSearcher concurrent = perLeafSlices(executor);
            for (int attempt = 0; attempt < 20; attempt++) {
                final ClusterANNQuery query = new ClusterANNQuery(FIELD, query(7), K, new FailingQuery(), null, 0);
                final IOException e = expectThrows(IOException.class, () -> concurrent.search(query, K));
                assertEquals("boom", e.getMessage());
            }
        } finally {
            executor.shutdownNow();
        }
    }

    private IndexSearcher perLeafSlices(final ExecutorService executor) {
        final IndexSearcher concurrent = new IndexSearcher(reader, executor) {
            @Override
            protected LeafSlice[] slices(final List<LeafReaderContext> leaves) {
                return IndexSearcher.slices(leaves, 1, 1, false);
            }
        };
        concurrent.setQueryCache(null);
        return concurrent;
    }

    /** A filter whose every segment fails to produce a scorer. */
    private static final class FailingQuery extends Query {
        @Override
        public Weight createWeight(final IndexSearcher searcher, final ScoreMode scoreMode, final float boost) {
            return new Weight(this) {
                @Override
                public ScorerSupplier scorerSupplier(final LeafReaderContext context) throws IOException {
                    throw new IOException("boom");
                }

                @Override
                public Explanation explain(final LeafReaderContext context, final int doc) {
                    return Explanation.noMatch("fails");
                }

                @Override
                public boolean isCacheable(final LeafReaderContext ctx) {
                    return false;
                }
            };
        }

        @Override
        public String toString(final String field) {
            return "FailingQuery";
        }

        @Override
        public void visit(final QueryVisitor visitor) {}

        @Override
        public boolean equals(final Object other) {
            return sameClassAs(other);
        }

        @Override
        public int hashCode() {
            return classHash();
        }
    }

    // ---------------------------------------------------------------- EXPERIMENT switches (do not merge)

    /**
     * Shard plan off: every segment is searched without a plan, so the far segment is searched too, and the hits are
     * exactly the pre-shard-plan path's (the reader planning each segment itself).
     */
    public void testSwitch_whenShardPlanOff_thenPerSegmentPathExactly() throws IOException {
        ShardPlanExperiment.set(new ShardPlanExperiment.Config(false, ShardPlanner.Budget.RULE, 8.0));
        try {
            for (int q = 0; q < 10; q++) {
                for (final Query filter : new Query[] { null, new TermQuery(new Term(PARITY, "even")) }) {
                    final long requests = ClusterANNStat.QUERY_REQUESTS.getTotal();
                    final TopDocs off = searcher.search(new ClusterANNQuery(FIELD, query(200 + q), K, filter, null, 0), K);
                    assertEquals("every segment searched", SEGMENT_SIZES.length, ClusterANNStat.QUERY_REQUESTS.getTotal() - requests);
                    final TopDocs today = searcher.search(new ShardPlanEvalRun.TodayQuery(FIELD, query(200 + q), K, filter), K);
                    assertTrue("query " + q + " filter " + filter + " returned hits", today.scoreDocs.length > 0);
                    assertSameHits("query " + q, today, off);
                }
            }
        } finally {
            ShardPlanExperiment.reset();
        }
    }

    /** The live budget and floor reach a query built through the public constructor, read when its weight is made. */
    public void testSwitch_whenBudgetAndFloorAreSet_thenTheWeightPlansWithThem() throws IOException {
        final LeafReaderContext leaf = reader.leaves().get(0);
        try {
            ShardPlanExperiment.set(new ShardPlanExperiment.Config(true, ShardPlanner.Budget.RULE, 8.0));
            final ShardPlanner.Plan rule = liveWeight(query(9)).shardPlan(leaf).plan();
            assertEquals(rule.params().maxProbes(), rule.budget());
            assertEquals(rule.params().maxProbes(), rule.probeCount());

            ShardPlanExperiment.set(new ShardPlanExperiment.Config(true, ShardPlanner.Budget.RULE, 0.0));
            ProbeCurveExperiment.set(ProbeCurveExperiment.DEFAULT_CURVE.withIntercept(1.0).withCoefficient(0.0));
            final ShardPlanner.Plan staticOne = liveWeight(query(9)).shardPlan(leaf).plan();
            assertEquals((int) Math.sqrt(staticOne.totalClusters()), staticOne.params().maxProbes());
            assertEquals(staticOne.params().maxProbes(), staticOne.probeCount());

            ShardPlanExperiment.reset();
            ProbeCurveExperiment.reset();
            final ShardPlanner.Plan capped = liveWeight(query(9)).shardPlan(leaf).plan();
            assertTrue(capped.budget() <= capped.params().maxProbes());
            assertEquals(ShardPlanner.shardParams(capped.totalClusters(), 1.0), capped.params());
        } finally {
            ShardPlanExperiment.reset();
            ProbeCurveExperiment.reset();
        }
    }

    /**
     * The scan early stop acts on the shard-plan path: on, a segment's scan stops short of its maxProbes_s; off (the
     * default, the baseline's setting), every probe the plan gives is scanned and no stop is recorded.
     */
    public void testSwitch_whenEarlyStop_thenTheShardPlanScanHonoursIt() throws IOException {
        ShardPlanExperiment.set(new ShardPlanExperiment.Config(true, ShardPlanner.Budget.RULE, 8.0));
        try {
            final long stopsOff = ClusterANNStat.SCAN_EARLY_STOP.getTotal();
            final long scannedOff = ClusterANNStat.CLUSTERS_SCANNED.getTotal();
            final long plannedOff = ClusterANNStat.PROBES_PLANNED.getTotal();
            searcher.search(new ClusterANNQuery(FIELD, query(11), K, null, null, 0), K);
            assertEquals("off: no stop", stopsOff, ClusterANNStat.SCAN_EARLY_STOP.getTotal());
            assertEquals(
                "off: every planned probe scanned",
                ClusterANNStat.PROBES_PLANNED.getTotal() - plannedOff,
                ClusterANNStat.CLUSTERS_SCANNED.getTotal() - scannedOff
            );

            EarlyStopExperiment.set(new EarlyStopParams(1, K, 0));
            final long stopsOn = ClusterANNStat.SCAN_EARLY_STOP.getTotal();
            final long scannedOn = ClusterANNStat.CLUSTERS_SCANNED.getTotal();
            final long plannedOn = ClusterANNStat.PROBES_PLANNED.getTotal();
            searcher.search(new ClusterANNQuery(FIELD, query(11), K, null, null, 0), K);
            assertTrue("on: a scan stopped early", ClusterANNStat.SCAN_EARLY_STOP.getTotal() > stopsOn);
            assertTrue(
                "on: fewer clusters scanned than planned",
                ClusterANNStat.CLUSTERS_SCANNED.getTotal() - scannedOn < ClusterANNStat.PROBES_PLANNED.getTotal() - plannedOn
            );
        } finally {
            EarlyStopExperiment.reset();
            ShardPlanExperiment.reset();
        }
    }

    /**
     * A query runs with the switches live when it was built, even if they change before its weight is made, and two
     * queries built under different switches are not equal (so nothing keyed on the query mixes them).
     */
    public void testSwitch_whenChangedAfterTheQueryIsBuilt_thenTheQueryKeepsItsOwn() throws IOException {
        final LeafReaderContext leaf = reader.leaves().get(0);
        try {
            ShardPlanExperiment.set(new ShardPlanExperiment.Config(true, ShardPlanner.Budget.RULE, 8.0));
            final ClusterANNQuery rule = new ClusterANNQuery(FIELD, query(13), K, null, null, 0);
            ShardPlanExperiment.set(new ShardPlanExperiment.Config(false, ShardPlanner.Budget.CAPPED, 0.0));
            final ClusterANNQuery off = new ClusterANNQuery(FIELD, query(13), K, null, null, 0);
            assertNotEquals(rule, off);

            final ClusterANNWeight weight = (ClusterANNWeight) searcher.createWeight(searcher.rewrite(rule), ScoreMode.COMPLETE, 1f);
            final ShardPlanner.Plan plan = weight.shardPlan(leaf).plan();
            assertEquals(ShardPlanner.shardParams(plan.totalClusters(), 1.0, 8.0), plan.params());
            assertEquals(plan.params().maxProbes(), plan.budget());
        } finally {
            ShardPlanExperiment.reset();
        }
    }

    private ClusterANNWeight liveWeight(final float[] query) throws IOException {
        final ClusterANNQuery q = new ClusterANNQuery(FIELD, query, K, null, null, 0);
        return (ClusterANNWeight) searcher.createWeight(searcher.rewrite(q), ScoreMode.COMPLETE, 1f);
    }

    private ClusterANNWeight weight(final float[] query, final Query filter, final ShardPlanner.Budget budget) throws IOException {
        final ClusterANNQuery q = new ClusterANNQuery(FIELD, query, K, filter, null, 0, budget);
        return (ClusterANNWeight) searcher.createWeight(searcher.rewrite(q), ScoreMode.COMPLETE, 1f);
    }

    /** The same AcceptDocs the weight builds: the filter conjoined with "has the field", or the live docs. */
    private AcceptDocs acceptDocs(final LeafReaderContext leaf, final Query filter) throws IOException {
        final ClusterANNWeight weight = weight(new float[DIM], filter, ShardPlanner.Budget.CAPPED);
        return weight.shardPlan(leaf).forLeaf(leaf.ord).acceptDocs();
    }

    private static KNN1030ClusterANNVectorsReader clusterAnnReader(final LeafReaderContext leaf) {
        final KNN1030ClusterANNVectorsReader vectors = ClusterANNWeight.clusterAnnReader(leaf.reader(), FIELD);
        assertNotNull(vectors);
        return vectors;
    }

    private static Clusters clusters(final LeafReaderContext leaf) throws IOException {
        return clusterAnnReader(leaf).shardPlanSegment(FIELD, null).clusters();
    }

    private static float[] query(final int seed) {
        final Random random = new Random(seed);
        final float[] query = new float[DIM];
        for (int d = 0; d < DIM; d++) {
            query[d] = random.nextFloat();
        }
        return query;
    }

    static void assertSameHits(final String message, final TopDocs expected, final TopDocs actual) {
        assertEquals(message, expected.scoreDocs.length, actual.scoreDocs.length);
        for (int i = 0; i < expected.scoreDocs.length; i++) {
            assertEquals(message + " hit " + i, expected.scoreDocs[i].doc, actual.scoreDocs[i].doc);
            assertEquals(message + " hit " + i, expected.scoreDocs[i].score, actual.scoreDocs[i].score, 0f);
        }
    }
}
