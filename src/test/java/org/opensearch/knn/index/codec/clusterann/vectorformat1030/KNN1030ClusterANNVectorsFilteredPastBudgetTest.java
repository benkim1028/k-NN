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

/** Filtered search end to end with a k the probe budget cannot hold: the plan passes maxProbes and returns k. */
@LuceneTestCase.SuppressSysoutChecks(bugUrl = "The CheckIndex report and plan stats are the point of this test")
public class KNN1030ClusterANNVectorsFilteredPastBudgetTest extends LuceneTestCase {

    private static final int DIMENSION = 8;
    private static final int BLOBS = 8;
    private static final int BLOB_SIZE = 8192;
    private static final int NUM_DOCS = BLOBS * BLOB_SIZE;
    private static final float BLOB_SPACING = 100f;
    private static final int NUM_QUERIES = 16;
    /** 4-bit codes; well above what a plan stopped at the budget could reach. */
    private static final double MIN_RECALL = 0.8;

    /**
     * Clumped: 16 matches in each of the 7 blobs nearest the query, all of the farthest blob. The budget's clusters hold
     * fewer than 200; the dense blob ranks past it.
     */
    public void testFilteredSearch_whenTheMatchesAreClumpedPastTheBudget_thenReturnsK() throws Exception {
        final FixedBitSet filter = new FixedBitSet(NUM_DOCS);
        for (int doc = 0; doc < (BLOBS - 1) * BLOB_SIZE; doc += 512) {
            filter.set(doc);
        }
        filter.set((BLOBS - 1) * BLOB_SIZE, NUM_DOCS);
        assertFilteredSearchReturnsK("clumped", filter, 200);
    }

    /** Large k at s = 0.01: 656 matches spread evenly; the budget of 114 clusters holds fewer than k = 640 for any root 11. */
    public void testFilteredSearch_whenKIsLargeAtLowSelectivity_thenReturnsK() throws Exception {
        final FixedBitSet filter = new FixedBitSet(NUM_DOCS);
        for (int doc = 0; doc < NUM_DOCS; doc += 100) {
            filter.set(doc);
        }
        assertFilteredSearchReturnsK("large k", filter, 640);
    }

    private void assertFilteredSearchReturnsK(final String shape, final FixedBitSet filter, final int k) throws Exception {
        final float[][] vectors = blobs(BLOB_SIZE, 42);
        final float[][] queries = nearTheFirstBlob(NUM_QUERIES, 7);
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
                final long pastBudgetBefore = ClusterANNStat.FILTER_EXTENDED_PAST_MAX_PROBES.getCount();
                final long kneeStopsBefore = ClusterANNStat.CENTROID_KNEE_EARLY_STOP.getCount();
                final long plannedBefore = ClusterANNStat.PROBES_PLANNED.getTotal();
                final long eligibleBefore = ClusterANNStat.FILTER_ELIGIBLE_CLUSTERS.getTotal();
                int found = 0;
                int fewestReturned = Integer.MAX_VALUE;
                for (float[] query : queries) {
                    final AcceptDocs accept = AcceptDocs.fromIteratorSupplier(
                        () -> new BitSetIterator(filter, filter.cardinality()),
                        null,
                        NUM_DOCS
                    );
                    final TopDocs hits = leaf.searchNearestVectors("v", query, k, accept, Integer.MAX_VALUE);
                    fewestReturned = Math.min(fewestReturned, hits.scoreDocs.length);
                    final Set<Integer> exact = exactTopK(vectors, query, filter, k);
                    for (ScoreDoc hit : hits.scoreDocs) {
                        if (exact.contains(hit.doc)) {
                            found++;
                        }
                    }
                }
                final long centroids = ClusterANNStat.CENTROID_COUNT.getTotal() - centroidsBefore;
                final long pastBudget = ClusterANNStat.FILTER_EXTENDED_PAST_MAX_PROBES.getCount() - pastBudgetBefore;
                final long kneeStops = ClusterANNStat.CENTROID_KNEE_EARLY_STOP.getCount() - kneeStopsBefore;
                final double plannedPerQuery = (double) (ClusterANNStat.PROBES_PLANNED.getTotal() - plannedBefore) / queries.length;
                final double eligiblePerQuery = (double) (ClusterANNStat.FILTER_ELIGIBLE_CLUSTERS.getTotal() - eligibleBefore)
                    / queries.length;
                final double recall = (double) found / (k * queries.length);
                System.out.println(
                    "filtered past budget, "
                        + shape
                        + ": accepted "
                        + filter.cardinality()
                        + ", k "
                        + k
                        + ", clusters "
                        + centroids
                        + ", eligible/query "
                        + eligiblePerQuery
                        + ", planned/query "
                        + plannedPerQuery
                        + ", fewest returned "
                        + fewestReturned
                        + ", past max probes "
                        + pastBudget
                        + ", knee stops "
                        + kneeStops
                        + " of "
                        + queries.length
                        + ", recall@"
                        + k
                        + " "
                        + recall
                );
                // Suites run one at a time in the forked JVM, so these deltas are this test's alone.
                assertEquals("fixture premise: root 11, so floor 11", 11, (int) Math.sqrt(centroids));
                assertEquals("every query returns k", k, fewestReturned);
                assertEquals("every query extends past maxProbes", queries.length, pastBudget);
                assertTrue("recall@" + k + " " + recall + " below " + MIN_RECALL, recall >= MIN_RECALL);
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

    private static Set<Integer> exactTopK(float[][] vectors, float[] query, FixedBitSet accepted, int k) {
        final VectorSimilarityFunction sim = VectorSimilarityFunction.EUCLIDEAN;
        int[] ranked = IntStream.range(0, vectors.length)
            .filter(accepted::get)
            .boxed()
            .sorted((a, b) -> Float.compare(sim.compare(query, vectors[b]), sim.compare(query, vectors[a])))
            .mapToInt(Integer::intValue)
            .toArray();
        Set<Integer> topK = new HashSet<>();
        Arrays.stream(ranked, 0, Math.min(k, ranked.length)).forEach(topK::add);
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

    /** Unit-gaussian queries around the first blob's centre, the end the clumped filter thins out. */
    private static float[][] nearTheFirstBlob(int count, long seed) {
        final Random random = new Random(seed);
        final float[][] out = new float[count][DIMENSION];
        for (float[] query : out) {
            for (int d = 0; d < DIMENSION; d++) {
                query[d] = (float) random.nextGaussian();
            }
        }
        return out;
    }
}
