/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.vectorformat1030;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.CheckIndex;
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
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.clusterann.stats.ClusterANNStat;
import org.opensearch.knn.index.codec.clusterann.ClusterANN1030TestCodec;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

/** Filtered search end to end: 8 blobs of about 18 clusters, floor 11; every query is knee-cut and recall holds. */
@LuceneTestCase.SuppressSysoutChecks(bugUrl = "The CheckIndex report and cut stats are the point of this test")
public class KNN1030ClusterANNVectorsFilteredKneeTest extends LuceneTestCase {

    private static final int DIMENSION = 8;
    private static final int BLOBS = 8;
    private static final int BLOB_SIZE = 8192;
    private static final int NUM_DOCS = BLOBS * BLOB_SIZE;
    private static final float BLOB_SPACING = 100f;
    private static final int K = 10;
    private static final int NUM_QUERIES = 24;
    /** Catches a cut that drops the query's own blob; 4-bit codes hold recall@10 near 0.9 either way. */
    private static final double MIN_RECALL = 0.85;
    /** Most clusters a query may plan, in blobs: its own, with slack. */
    private static final double OWN_BLOB = 1.5;
    /** Its own blob and both neighbours, with slack. */
    private static final double WITH_NEIGHBOURS = 3.5;

    public void testFilteredSearch_whenHalfTheDocumentsPass_thenCutsAtTheKneeAndKeepsRecall() throws Exception {
        assertFilteredSearchCutsAtTheKnee(2, OWN_BLOB);
    }

    /** s = 0.05, knee factor about 4.97: still cut at the query's own blob. */
    public void testFilteredSearch_whenOneDocumentInTwentyPasses_thenStillCutsAtTheKneeAndKeepsRecall() throws Exception {
        assertFilteredSearchCutsAtTheKnee(20, OWN_BLOB);
    }

    /** s = 0.01, knee factor at its cap of 8: still cut, but may keep the neighbouring blobs too. */
    public void testFilteredSearch_whenOneDocumentInAHundredPasses_thenCutsByTheNeighbouringBlobsAndKeepsRecall() throws Exception {
        assertFilteredSearchCutsAtTheKnee(100, WITH_NEIGHBOURS);
    }

    /** Filters to every {@code stride}-th document and checks each query is cut within {@code blobsPlanned} blobs. */
    private void assertFilteredSearchCutsAtTheKnee(final int stride, final double blobsPlanned) throws Exception {
        final float[][] vectors = blobs(BLOB_SIZE, 42);
        final float[][] queries = blobs(NUM_QUERIES / BLOBS, 7);
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
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals("one segment", 1, reader.leaves().size());
                final LeafReader leaf = reader.leaves().get(0).reader();
                final FixedBitSet filter = new FixedBitSet(NUM_DOCS);
                for (int doc = 0; doc < NUM_DOCS; doc += stride) {
                    filter.set(doc);
                }
                final long kneeStopsBefore = ClusterANNStat.CENTROID_KNEE_EARLY_STOP.getCount();
                final long plannedBefore = ClusterANNStat.PROBES_PLANNED.getTotal();
                int found = 0;
                long mostPlanned = 0;
                for (float[] query : queries) {
                    final AcceptDocs accept = AcceptDocs.fromIteratorSupplier(
                        () -> new BitSetIterator(filter, filter.cardinality()),
                        null,
                        NUM_DOCS
                    );
                    final long queryPlannedBefore = ClusterANNStat.PROBES_PLANNED.getTotal();
                    final TopDocs hits = leaf.searchNearestVectors("v", query, K, accept, Integer.MAX_VALUE);
                    mostPlanned = Math.max(mostPlanned, ClusterANNStat.PROBES_PLANNED.getTotal() - queryPlannedBefore);
                    assertEquals(K, hits.scoreDocs.length);
                    final Set<Integer> exact = exactTopK(vectors, query, filter);
                    for (ScoreDoc hit : hits.scoreDocs) {
                        if (exact.contains(hit.doc)) {
                            found++;
                        }
                    }
                }
                final long centroids = ClusterANNStat.CENTROID_COUNT.getTotal() - centroidsBefore;
                final long kneeStops = ClusterANNStat.CENTROID_KNEE_EARLY_STOP.getCount() - kneeStopsBefore;
                final double plannedPerQuery = (double) (ClusterANNStat.PROBES_PLANNED.getTotal() - plannedBefore) / queries.length;
                final double recall = (double) found / (K * queries.length);
                System.out.println(
                    "filtered knee: 1 in "
                        + stride
                        + ", clusters "
                        + centroids
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
                // Suites run one at a time in the forked JVM, so these deltas are this test's alone.
                assertEquals("fixture premise: root 11, so floor 11", 11, (int) Math.sqrt(centroids));
                assertEquals("every filtered query is knee-cut", queries.length, kneeStops);
                // Per query, not on average.
                final double plannedBound = blobsPlanned * centroids / BLOBS;
                assertTrue(
                    "every query cut within "
                        + blobsPlanned
                        + " blobs of about "
                        + (double) centroids / BLOBS
                        + " clusters, most planned "
                        + mostPlanned,
                    mostPlanned <= plannedBound
                );
                assertTrue("recall@" + K + " " + recall + " below " + MIN_RECALL, recall >= MIN_RECALL);
            }
            // Explicit CheckIndex too, so its report for the vector field lands in the test log.
            final ByteArrayOutputStream checkIndexOutput = new ByteArrayOutputStream();
            final CheckIndex.Status status = TestUtil.checkIndex(
                dir,
                CheckIndex.Level.MIN_LEVEL_FOR_INTEGRITY_CHECKS,
                true,
                false,
                checkIndexOutput
            );
            final String report = checkIndexOutput.toString(StandardCharsets.UTF_8);
            System.out.println("CheckIndex report for vector field v:\n" + report);
            assertTrue(report, status.clean);
            final CheckIndex.Status.VectorValuesStatus vectorStatus = status.segmentInfos.get(0).vectorValuesStatus;
            assertNull(report, vectorStatus.error);
            assertEquals(report, 1, vectorStatus.totalKnnVectorFields);
            assertEquals(report, NUM_DOCS, vectorStatus.totalVectorValues);
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
