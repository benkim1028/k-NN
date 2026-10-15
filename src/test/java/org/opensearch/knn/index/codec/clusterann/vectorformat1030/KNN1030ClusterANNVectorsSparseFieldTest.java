/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.vectorformat1030;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.opensearch.knn.index.codec.clusterann.ClusterANN1030TestCodec;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.IntPredicate;

/**
 * Search over a field that only some documents carry, so a vector's ordinal and its document ID differ and the
 * reader resolves every hit through the field's sparse ord&rarr;doc mapping. Covers documents that simply go without
 * a vector, and nested blocks the way OpenSearch indexes a nested field: child documents that carry the vectors,
 * each block closed by a parent that carries none. Each layout is searched on a flushed and on a merged segment,
 * since each writes that mapping.
 */
public class KNN1030ClusterANNVectorsSparseFieldTest extends LuceneTestCase {

    private static final String FIELD = "vector";
    private static final String ROW = "row";
    private static final int DIMENSION = 8;
    private static final int NUM_ROWS = 60;
    private static final int CHILDREN_PER_PARENT = 3;
    private static final int K = 10;
    private static final int DOC_BITS = 4;
    private static final float[][] VECTORS = vectors();

    public void testSearch_whenSomeDocsHaveNoVector_flushedSegment() throws Exception {
        try (Directory dir = new ByteBuffersDirectory()) {
            try (IndexWriter writer = new IndexWriter(dir, flushedConfig())) {
                addDocsSkippingEveryThird(writer, 0, NUM_ROWS);
                writer.commit();
            }
            assertEverySearchFindsItsOwnDoc(dir, KNN1030ClusterANNVectorsSparseFieldTest::hasVector);
        }
    }

    public void testSearch_whenSomeDocsHaveNoVector_mergedSegment() throws Exception {
        try (Directory dir = new ByteBuffersDirectory()) {
            try (IndexWriter writer = new IndexWriter(dir, mergedConfig())) {
                addDocsSkippingEveryThird(writer, 0, NUM_ROWS / 2);
                writer.commit();
                addDocsSkippingEveryThird(writer, NUM_ROWS / 2, NUM_ROWS);
                writer.commit();
                writer.forceMerge(1);
            }
            assertEverySearchFindsItsOwnDoc(dir, KNN1030ClusterANNVectorsSparseFieldTest::hasVector);
        }
    }

    public void testSearch_whenNestedBlocks_flushedSegment() throws Exception {
        try (Directory dir = new ByteBuffersDirectory()) {
            try (IndexWriter writer = new IndexWriter(dir, flushedConfig())) {
                addNestedBlocks(writer, 0, NUM_ROWS);
                writer.commit();
            }
            assertEverySearchFindsItsOwnDoc(dir, row -> true);
        }
    }

    public void testSearch_whenNestedBlocks_mergedSegment() throws Exception {
        try (Directory dir = new ByteBuffersDirectory()) {
            try (IndexWriter writer = new IndexWriter(dir, mergedConfig())) {
                addNestedBlocks(writer, 0, NUM_ROWS / 2);
                writer.commit();
                addNestedBlocks(writer, NUM_ROWS / 2, NUM_ROWS);
                writer.commit();
                writer.forceMerge(1);
            }
            assertEverySearchFindsItsOwnDoc(dir, row -> true);
        }
    }

    private static IndexWriterConfig flushedConfig() {
        return config().setMergePolicy(NoMergePolicy.INSTANCE).setMaxBufferedDocs(2 * NUM_ROWS);
    }

    private static IndexWriterConfig mergedConfig() {
        return config().setMergeScheduler(new SerialMergeScheduler());
    }

    private static IndexWriterConfig config() {
        return new IndexWriterConfig().setCodec(new ClusterANN1030TestCodec(DOC_BITS)).setUseCompoundFile(false);
    }

    private static void addDocsSkippingEveryThird(final IndexWriter writer, final int fromRow, final int toRow) throws Exception {
        for (int row = fromRow; row < toRow; row++) {
            writer.addDocument(hasVector(row) ? vectorDoc(row) : new Document());
        }
    }

    private static void addNestedBlocks(final IndexWriter writer, final int fromRow, final int toRow) throws Exception {
        for (int first = fromRow; first < toRow; first += CHILDREN_PER_PARENT) {
            final List<Document> block = new ArrayList<>(CHILDREN_PER_PARENT + 1);
            for (int row = first; row < Math.min(first + CHILDREN_PER_PARENT, toRow); row++) {
                block.add(vectorDoc(row));
            }
            block.add(new Document());
            writer.addDocuments(block);
        }
    }

    private static Document vectorDoc(final int row) {
        final Document doc = new Document();
        doc.add(new KnnFloatVectorField(FIELD, VECTORS[row], VectorSimilarityFunction.EUCLIDEAN));
        doc.add(new StoredField(ROW, row));
        return doc;
    }

    private static void assertEverySearchFindsItsOwnDoc(final Directory dir, final IntPredicate indexedRows) throws Exception {
        try (DirectoryReader reader = DirectoryReader.open(dir)) {
            assertEquals(1, reader.leaves().size());
            final IndexSearcher searcher = new IndexSearcher(reader);
            for (int row = 0; row < NUM_ROWS; row++) {
                if (!indexedRows.test(row)) {
                    continue;
                }
                final TopDocs hits = searcher.search(new KnnFloatVectorQuery(FIELD, VECTORS[row], K), K);
                boolean foundSelf = false;
                for (final ScoreDoc hit : hits.scoreDocs) {
                    final IndexableField hitRow = searcher.storedFields().document(hit.doc).getField(ROW);
                    assertNotNull("query " + row + " hit doc " + hit.doc + ", which has no vector", hitRow);
                    foundSelf |= hitRow.numericValue().intValue() == row;
                }
                assertTrue("query " + row + " did not find its own doc in the top " + K, foundSelf);
            }
        }
    }

    private static boolean hasVector(final int row) {
        // Every third doc goes without, so ordinals fall behind doc IDs from the third doc on.
        return row % 3 != 2;
    }

    private static float[][] vectors() {
        final Random random = new Random(42);
        final float[][] vectors = new float[NUM_ROWS][DIMENSION];
        for (final float[] vector : vectors) {
            for (int d = 0; d < DIMENSION; d++) {
                vector[d] = random.nextFloat();
            }
        }
        return vectors;
    }
}
