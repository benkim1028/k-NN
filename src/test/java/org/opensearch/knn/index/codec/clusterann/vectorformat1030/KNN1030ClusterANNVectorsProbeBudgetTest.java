/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.vectorformat1030;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.store.BaseDirectoryWrapper;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.clusterann.read.orchestration.KneeExperiment;
import org.opensearch.knn.clusterann.read.orchestration.ProbeCurveExperiment;
import org.opensearch.knn.clusterann.stats.ClusterANNStat;
import org.opensearch.knn.index.codec.clusterann.ClusterANN1030TestCodec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

/** Filtered search with recall check; LuceneTestCase directory close runs CheckIndex. */
public class KNN1030ClusterANNVectorsProbeBudgetTest extends LuceneTestCase {

    private static final int DIMENSION = 8;
    private static final int NUM_DOCS = 16384;
    private static final int K = 10;
    private static final int NUM_QUERIES = 20;
    private static final double MIN_RECALL = 0.95;

    public void testFilteredSearch_thenRecallAboveFloor() throws Exception {
        final float[][] vectors = vectors(NUM_DOCS, 42);
        final float[][] queries = vectors(NUM_QUERIES, 7);
        try (Directory dir = newDirectory()) {
            assertTrue("CheckIndex on close", ((BaseDirectoryWrapper) dir).getCheckIndexOnClose());
            final IndexWriterConfig iwc = new IndexWriterConfig().setCodec(new ClusterANN1030TestCodec(4))
                .setMergePolicy(NoMergePolicy.INSTANCE)
                .setUseCompoundFile(false)
                .setMaxBufferedDocs(NUM_DOCS + 1);
            try (IndexWriter writer = new IndexWriter(dir, iwc)) {
                for (float[] v : vectors) {
                    final Document doc = new Document();
                    doc.add(new KnnFloatVectorField("v", v, VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.commit();
            }
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final LeafReader leaf = reader.leaves().get(0).reader();
                final FixedBitSet filter = new FixedBitSet(NUM_DOCS);
                for (int doc = 0; doc < NUM_DOCS; doc += 500) {
                    filter.set(doc);
                }
                int found = 0;
                for (float[] query : queries) {
                    final AcceptDocs accept = AcceptDocs.fromIteratorSupplier(
                        () -> new BitSetIterator(filter, filter.cardinality()),
                        null,
                        NUM_DOCS
                    );
                    final TopDocs hits = leaf.searchNearestVectors("v", query, K, accept, Integer.MAX_VALUE);
                    assertEquals(K, hits.scoreDocs.length);
                    final Set<Integer> exact = exactTopK(vectors, query, filter);
                    for (ScoreDoc hit : hits.scoreDocs) {
                        if (exact.contains(hit.doc)) {
                            found++;
                        }
                    }
                }
                double recall = (double) found / (K * NUM_QUERIES);
                assertTrue("recall@" + K + " " + recall + " below " + MIN_RECALL, recall >= MIN_RECALL);
            }
        }
    }

    /**
     * EXPERIMENT (do not merge): the live probe curve reaches a real index. One segment is built once and searched
     * under four curves: the default, m = 0 (one probe), m = 1 (coefficient 0: floor(sqrt C) probes), and cap 1 on a
     * selective filter.
     * The planned probes follow each curve, and putting the default back reproduces the default run exactly.
     */
    public void testSearch_whenTheProbeCurveMoves_thenPlannedProbesFollowIt() throws Exception {
        final float[][] vectors = vectors(NUM_DOCS, 42);
        final float[][] queries = vectors(NUM_QUERIES, 7);
        try (Directory dir = newDirectory()) {
            assertTrue("CheckIndex on close", ((BaseDirectoryWrapper) dir).getCheckIndexOnClose());
            final IndexWriterConfig iwc = new IndexWriterConfig().setCodec(new ClusterANN1030TestCodec(4))
                .setMergePolicy(NoMergePolicy.INSTANCE)
                .setUseCompoundFile(false)
                .setMaxBufferedDocs(NUM_DOCS + 1)
                .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
            final long centroidsBefore = ClusterANNStat.CENTROID_COUNT.getTotal();
            try (IndexWriter writer = new IndexWriter(dir, iwc)) {
                for (float[] v : vectors) {
                    final Document doc = new Document();
                    doc.add(new KnnFloatVectorField("v", v, VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.commit();
            }
            final int root = (int) Math.sqrt(ClusterANNStat.CENTROID_COUNT.getTotal() - centroidsBefore);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals("one segment", 1, reader.leaves().size());
                final LeafReader leaf = reader.leaves().get(0).reader();
                final FixedBitSet selective = new FixedBitSet(NUM_DOCS);
                for (int doc = 0; doc < NUM_DOCS; doc += 100) {
                    selective.set(doc);
                }
                final FixedBitSet all = new FixedBitSet(NUM_DOCS);
                all.set(0, NUM_DOCS);
                final List<Set<Integer>> exactAll = exactTopKPerQuery(vectors, queries, all);
                final List<Set<Integer>> exactSelective = exactTopKPerQuery(vectors, queries, selective);
                try {
                    final double[] defaults = plannedPerQuery(leaf, queries, null, exactAll);
                    final double[] defaultsFiltered = plannedPerQuery(leaf, queries, selective, exactSelective);

                    ProbeCurveExperiment.set(ProbeCurveExperiment.DEFAULT_CURVE.withIntercept(0.0).withCoefficient(0.0));
                    final double[] zero = plannedPerQuery(leaf, queries, null, exactAll);
                    assertEquals("m = 0 plans one probe", 1.0, zero[0], 0.0);

                    ProbeCurveExperiment.set(ProbeCurveExperiment.DEFAULT_CURVE.withCoefficient(0.0));
                    final double[] one = plannedPerQuery(leaf, queries, null, exactAll);
                    assertEquals("m = 1 plans floor(sqrt C) probes", root, one[0], 0.0);

                    ProbeCurveExperiment.set(ProbeCurveExperiment.DEFAULT_CURVE.withCap(1.0));
                    final double[] capped = plannedPerQuery(leaf, queries, selective, exactSelective);
                    assertTrue(
                        "cap 1 plans fewer probes on a selective filter: " + capped[0] + " vs " + defaultsFiltered[0],
                        capped[0] < defaultsFiltered[0]
                    );

                    ProbeCurveExperiment.reset();
                    final double[] again = plannedPerQuery(leaf, queries, null, exactAll);
                    final double[] againFiltered = plannedPerQuery(leaf, queries, selective, exactSelective);
                    assertArrayEquals("the default curve reproduces the default run", defaults, again, 0.0);
                    assertArrayEquals("the default curve reproduces the default filtered run", defaultsFiltered, againFiltered, 0.0);
                    assertTrue("default planned/query " + defaults[0] + " under 3 x root " + root, defaults[0] <= 3 * root);
                    System.out.println(
                        "probe curve: clusters root "
                            + root
                            + ", planned/query default "
                            + defaults[0]
                            + " (recall "
                            + defaults[1]
                            + "), m=0 "
                            + zero[0]
                            + " (recall "
                            + zero[1]
                            + "), m=1 "
                            + one[0]
                            + " (recall "
                            + one[1]
                            + "), filtered default "
                            + defaultsFiltered[0]
                            + " (recall "
                            + defaultsFiltered[1]
                            + "), filtered cap 1 "
                            + capped[0]
                            + " (recall "
                            + capped[1]
                            + ")"
                    );
                } finally {
                    ProbeCurveExperiment.reset();
                }
            }
        }
    }

    /**
     * EXPERIMENT (do not merge): knee off and a static multiplier reach a real index. With the knee on the default curve
     * plans at most ceil(3 x root); with it off every query plans exactly that. A static m (coefficient 0, intercept m)
     * plans exactly ceil(m x root) unfiltered and on a selective filter alike.
     */
    public void testSearch_whenTheKneeIsOffAndTheMultiplierIsStatic_thenEveryQueryPlansTheStaticCeiling() throws Exception {
        final float[][] vectors = vectors(NUM_DOCS, 42);
        final float[][] queries = vectors(NUM_QUERIES, 7);
        try (Directory dir = newDirectory()) {
            assertTrue("CheckIndex on close", ((BaseDirectoryWrapper) dir).getCheckIndexOnClose());
            final IndexWriterConfig iwc = new IndexWriterConfig().setCodec(new ClusterANN1030TestCodec(4))
                .setMergePolicy(NoMergePolicy.INSTANCE)
                .setUseCompoundFile(false)
                .setMaxBufferedDocs(NUM_DOCS + 1)
                .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
            final long centroidsBefore = ClusterANNStat.CENTROID_COUNT.getTotal();
            try (IndexWriter writer = new IndexWriter(dir, iwc)) {
                for (float[] v : vectors) {
                    final Document doc = new Document();
                    doc.add(new KnnFloatVectorField("v", v, VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.commit();
            }
            final long numClusters = ClusterANNStat.CENTROID_COUNT.getTotal() - centroidsBefore;
            final int root = (int) Math.sqrt(numClusters);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals("one segment", 1, reader.leaves().size());
                final LeafReader leaf = reader.leaves().get(0).reader();
                final FixedBitSet selective = new FixedBitSet(NUM_DOCS);
                for (int doc = 0; doc < NUM_DOCS; doc += 100) {
                    selective.set(doc);
                }
                final FixedBitSet all = new FixedBitSet(NUM_DOCS);
                all.set(0, NUM_DOCS);
                final List<Set<Integer>> exactAll = exactTopKPerQuery(vectors, queries, all);
                final List<Set<Integer>> exactSelective = exactTopKPerQuery(vectors, queries, selective);
                try {
                    final double[] kneeOn = plannedPerQuery(leaf, queries, null, exactAll);
                    KneeExperiment.setEnabled(false);
                    final double[] kneeOff = plannedPerQuery(leaf, queries, null, exactAll);
                    assertEquals("knee off plans the whole ceil(3 x root)", Math.min(3 * root, numClusters), kneeOff[0], 0.0);
                    assertTrue("the knee cuts some queries on this field: " + kneeOn[0], kneeOn[0] < kneeOff[0]);

                    final StringBuilder line = new StringBuilder("static m, knee off: clusters " + numClusters + ", root " + root);
                    for (int m : new int[] { 1, 2, 4 }) {
                        ProbeCurveExperiment.set(ProbeCurveExperiment.DEFAULT_CURVE.withCoefficient(0.0).withIntercept(m));
                        final double[] unfiltered = plannedPerQuery(leaf, queries, null, exactAll);
                        final double[] filtered = plannedPerQuery(leaf, queries, selective, exactSelective);
                        final long ceiling = Math.min((long) m * root, numClusters);
                        assertEquals("static m = " + m + " unfiltered", ceiling, unfiltered[0], 0.0);
                        assertEquals("static m = " + m + " filtered", ceiling, filtered[0], 0.0);
                        line.append(", m=")
                            .append(m)
                            .append(" planned ")
                            .append(unfiltered[0])
                            .append("/")
                            .append(filtered[0])
                            .append(" recall ")
                            .append(unfiltered[1])
                            .append("/")
                            .append(filtered[1]);
                    }
                    System.out.println(line + ", knee on default " + kneeOn[0] + " vs off " + kneeOff[0]);
                } finally {
                    KneeExperiment.reset();
                    ProbeCurveExperiment.reset();
                }
            }
        }
    }

    /**
     * Planned probes per query and recall@K over {@code queries}, unfiltered when {@code filter} is null. Reads the
     * JVM-global PROBES_PLANNED delta, which is this test's alone: the suite's tests run one at a time in their JVM.
     */
    private static double[] plannedPerQuery(
        final LeafReader leaf,
        final float[][] queries,
        final FixedBitSet filter,
        final List<Set<Integer>> exact
    ) throws Exception {
        final long plannedBefore = ClusterANNStat.PROBES_PLANNED.getTotal();
        int found = 0;
        for (int q = 0; q < queries.length; q++) {
            final AcceptDocs accept = filter == null
                ? AcceptDocs.fromLiveDocs(leaf.getLiveDocs(), NUM_DOCS)
                : AcceptDocs.fromIteratorSupplier(() -> new BitSetIterator(filter, filter.cardinality()), null, NUM_DOCS);
            final TopDocs hits = leaf.searchNearestVectors("v", queries[q], K, accept, Integer.MAX_VALUE);
            for (ScoreDoc hit : hits.scoreDocs) {
                if (exact.get(q).contains(hit.doc)) {
                    found++;
                }
            }
        }
        final double planned = (double) (ClusterANNStat.PROBES_PLANNED.getTotal() - plannedBefore) / queries.length;
        return new double[] { planned, (double) found / (K * queries.length) };
    }

    private static List<Set<Integer>> exactTopKPerQuery(final float[][] vectors, final float[][] queries, final FixedBitSet accepted) {
        final List<Set<Integer>> out = new ArrayList<>(queries.length);
        for (float[] query : queries) {
            out.add(exactTopK(vectors, query, accepted));
        }
        return out;
    }

    private static Set<Integer> exactTopK(float[][] vectors, float[] query, FixedBitSet accepted) {
        final VectorSimilarityFunction sim = VectorSimilarityFunction.EUCLIDEAN;
        int[] ranked = IntStream.range(0, vectors.length)
            .filter(accepted::get)
            .boxed()
            .sorted((a, b) -> Float.compare(sim.compare(query, vectors[b]), sim.compare(query, vectors[a])))
            .mapToInt(Integer::intValue)
            .toArray();
        Set<Integer> topK = new HashSet<>();
        Arrays.stream(ranked, 0, Math.min(K, ranked.length)).forEach(topK::add);
        return topK;
    }

    private static float[][] vectors(int count, long seed) {
        final Random random = new Random(seed);
        final float[][] out = new float[count][DIMENSION];
        for (float[] v : out) {
            for (int d = 0; d < DIMENSION; d++) {
                v[d] = random.nextFloat();
            }
        }
        return out;
    }
}
