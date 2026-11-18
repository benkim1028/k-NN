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
import org.opensearch.knn.index.codec.clusterann.ClusterANN1030TestCodec;

import java.util.Arrays;
import java.util.HashSet;
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
