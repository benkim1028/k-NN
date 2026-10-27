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
import org.opensearch.knn.clusterann.read.orchestration.PlanParams;
import org.opensearch.knn.index.codec.clusterann.ClusterANN1030TestCodec;
import org.opensearch.knn.plugin.stats.ClusterANNQueryValue;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

/**
 * Filtered search over clustered data, end to end: the eligible ranking is knee-cut at half selectivity and at one in
 * twenty, and recall holds. LuceneTestCase directory close runs CheckIndex.
 *
 * <p>Eight well-separated blobs of 8192 vectors give about 18 clusters each, 143 in all: a floor of 11 and a budget
 * of 39 at s = 0.5, 71 at s = 0.05. A query drawn from one blob sees its own blob's clusters close and the next blob
 * far away, so the curve breaks between the floor and the budget and every query should be cut there.
 *
 * <p>There is no knee-stop counter here, so a query counts as cut when it plans fewer clusters than the unfiltered
 * ceiling, which no filtered budget is below. The cluster count is read off {@link ClusterANNQueryValue#ELIGIBLE_CLUSTERS}:
 * at both strides every cluster, about 458 vectors, holds an accepted one.
 */
public class KNN1030ClusterANNVectorsFilteredKneeTest extends LuceneTestCase {

    private static final int DIMENSION = 8;
    private static final int BLOBS = 8;
    private static final int BLOB_SIZE = 8192;
    private static final int NUM_DOCS = BLOBS * BLOB_SIZE;
    private static final float BLOB_SPACING = 100f;
    private static final int K = 10;
    private static final int NUM_QUERIES = 24;
    /**
     * Guards against the cut dropping the query's own blob, which would read near zero, not against quantization: on
     * these dense blobs the 4-bit codes hold recall@10 near 0.9 whether or not the plan is cut.
     */
    private static final double MIN_RECALL = 0.85;

    public void testFilteredSearch_whenHalfTheDocumentsPass_thenCutsAtTheKneeAndKeepsRecall() throws Exception {
        assertFilteredSearchCutsAtTheKnee(2);
    }

    /**
     * No selectivity gate: one document in twenty passes, the budget rises to 71, and the plan is still cut at the
     * query's own blob, whose clusters hold about 410 accepted vectors, far more than k.
     */
    public void testFilteredSearch_whenOneDocumentInTwentyPasses_thenStillCutsAtTheKneeAndKeepsRecall() throws Exception {
        assertFilteredSearchCutsAtTheKnee(20);
    }

    /** Indexes the blobs, filters to every {@code stride}-th document, and checks each query is cut with recall kept. */
    private void assertFilteredSearchCutsAtTheKnee(final int stride) throws Exception {
        final float[][] vectors = blobs(BLOB_SIZE, 42);
        final float[][] queries = blobs(NUM_QUERIES / BLOBS, 7);
        try (Directory dir = newDirectory()) {
            assertTrue("CheckIndex on close", ((BaseDirectoryWrapper) dir).getCheckIndexOnClose());
            final IndexWriterConfig iwc = new IndexWriterConfig().setCodec(new ClusterANN1030TestCodec(4))
                .setMergePolicy(NoMergePolicy.INSTANCE)
                .setUseCompoundFile(false)
                .setMaxBufferedDocs(NUM_DOCS + 1)
                .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);
            try (IndexWriter writer = new IndexWriter(dir, iwc)) {
                for (float[] v : vectors) {
                    final Document doc = new Document();
                    doc.add(new KnnFloatVectorField("v", v, VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.commit();
            }
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals("one segment", 1, reader.leaves().size());
                final LeafReader leaf = reader.leaves().get(0).reader();
                final FixedBitSet filter = new FixedBitSet(NUM_DOCS);
                for (int doc = 0; doc < NUM_DOCS; doc += stride) {
                    filter.set(doc);
                }
                final long[] planned = new long[queries.length];
                final long eligibleBefore = ClusterANNQueryValue.ELIGIBLE_CLUSTERS.getValue();
                final long filteredScansBefore = ClusterANNQueryValue.FILTERED_SEGMENT_SCANS.getValue();
                int found = 0;
                for (int q = 0; q < queries.length; q++) {
                    final AcceptDocs accept = AcceptDocs.fromIteratorSupplier(
                        () -> new BitSetIterator(filter, filter.cardinality()),
                        null,
                        NUM_DOCS
                    );
                    final long queryPlannedBefore = ClusterANNQueryValue.CLUSTERS_PROBED.getValue();
                    final TopDocs hits = leaf.searchNearestVectors("v", queries[q], K, accept, Integer.MAX_VALUE);
                    planned[q] = ClusterANNQueryValue.CLUSTERS_PROBED.getValue() - queryPlannedBefore;
                    assertEquals(K, hits.scoreDocs.length);
                    final Set<Integer> exact = exactTopK(vectors, queries[q], filter);
                    for (ScoreDoc hit : hits.scoreDocs) {
                        if (exact.contains(hit.doc)) {
                            found++;
                        }
                    }
                }
                final long filteredScans = ClusterANNQueryValue.FILTERED_SEGMENT_SCANS.getValue() - filteredScansBefore;
                assertEquals("one filtered segment scan per query", queries.length, filteredScans);
                final int clusters = (int) ((ClusterANNQueryValue.ELIGIBLE_CLUSTERS.getValue() - eligibleBefore) / filteredScans);
                final int ceiling = PlanParams.of(clusters).maxProbes();
                final long kneeStops = Arrays.stream(planned).filter(p -> p < ceiling).count();
                final long mostPlanned = Arrays.stream(planned).max().orElse(0);
                final double plannedPerQuery = Arrays.stream(planned).average().orElse(0);
                final double recall = (double) found / (K * queries.length);
                System.out.println(
                    "filtered knee: 1 in "
                        + stride
                        + ", clusters "
                        + clusters
                        + ", planned/query "
                        + plannedPerQuery
                        + ", most planned "
                        + mostPlanned
                        + ", knee stops "
                        + kneeStops
                        + " of "
                        + queries.length
                        + ", recall@"
                        + K
                        + " "
                        + recall
                );
                // The forked test JVM runs suites one at a time (no parallel execution is configured), so these
                // deltas are this test's alone.
                assertEquals("fixture premise: root 11, so floor 11", 11, (int) Math.sqrt(clusters));
                assertEquals("every filtered query is knee-cut", queries.length, kneeStops);
                // Per query, not on average: guards a cut extended past the query's own blob.
                final double ownBlobBound = 1.5 * clusters / BLOBS;
                assertTrue(
                    "every query cut at its own blob of about " + (double) clusters / BLOBS + " clusters, most planned " + mostPlanned,
                    mostPlanned <= ownBlobBound
                );
                assertTrue("recall@" + K + " " + recall + " below " + MIN_RECALL, recall >= MIN_RECALL);
            }
        }
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

    /** {@code perBlob} unit-gaussian vectors around each of {@link #BLOBS} centres spaced along the first axis. */
    private static float[][] blobs(int perBlob, long seed) {
        final Random random = new Random(seed);
        final float[][] out = new float[BLOBS * perBlob][DIMENSION];
        for (int i = 0; i < out.length; i++) {
            out[i][0] = (i / perBlob) * BLOB_SPACING;
            for (int d = 0; d < DIMENSION; d++) {
                out[i][d] += (float) random.nextGaussian();
            }
        }
        return out;
    }
}
