/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.vectorformat1030;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.VectorUtil;
import org.opensearch.knn.index.codec.clusterann.ClusterANN1030TestCodec;

/**
 * Cosine fields cluster and quantize unit vectors, while the flat file keeps the vectors as they were indexed. Each
 * path — flush and merge — is run with inputs that are already unit length and with inputs that are not. Tests run
 * with assertions on, so a non-unit vector reaching the scalar quantizer fails the write
 * ({@code OptimizedScalarQuantizer} asserts unit vectors under cosine); a successful write is the check that the
 * quantizer saw unit vectors, and reading the flat vectors back is the check that they were left alone.
 *
 * <p>Every vector points along one direction, at different magnitudes, so every cluster's mean is already unit
 * length once its members are. That isolates the member-vector normalization under test from how centroids are
 * formed.
 */
public class KNN1030ClusterANNVectorsWriterCosineTest extends LuceneTestCase {

    private static final String FIELD = "cosine_field";
    private static final String ID = "id";
    private static final int DIMENSION = 4;
    private static final int NUM_DOCS = 40;
    private static final float[] DIRECTION = { 1f, 2f, 3f, 4f };

    public void testFlush_unnormalizedInput() throws Exception {
        assertFlushKeepsRawVectors(false);
    }

    public void testFlush_normalizedInput() throws Exception {
        assertFlushKeepsRawVectors(true);
    }

    public void testMerge_unnormalizedInput() throws Exception {
        assertMergeKeepsRawVectors(false);
    }

    public void testMerge_normalizedInput() throws Exception {
        assertMergeKeepsRawVectors(true);
    }

    private void assertFlushKeepsRawVectors(final boolean normalized) throws Exception {
        try (Directory dir = new ByteBuffersDirectory()) {
            // One flushed segment and no merges, so only the flush path clusters the vectors.
            final IndexWriterConfig iwc = newIndexWriterConfig().setCodec(new ClusterANN1030TestCodec())
                .setMergePolicy(NoMergePolicy.INSTANCE)
                .setMaxBufferedDocs(NUM_DOCS + 1);
            try (IndexWriter writer = new IndexWriter(dir, iwc)) {
                addDocs(writer, 0, NUM_DOCS, normalized);
                writer.commit();
            }
            assertRawVectors(dir, normalized);
        }
    }

    private void assertMergeKeepsRawVectors(final boolean normalized) throws Exception {
        try (Directory dir = new ByteBuffersDirectory()) {
            // A serial merge scheduler surfaces a failed merge here rather than on a background thread.
            final IndexWriterConfig iwc = new IndexWriterConfig().setCodec(new ClusterANN1030TestCodec())
                .setMergeScheduler(new SerialMergeScheduler());
            try (IndexWriter writer = new IndexWriter(dir, iwc)) {
                addDocs(writer, 0, NUM_DOCS / 2, normalized);
                writer.commit(); // a second segment, so the forceMerge below has something to merge
                addDocs(writer, NUM_DOCS / 2, NUM_DOCS, normalized);
                writer.commit();
                writer.forceMerge(1);
            }
            assertRawVectors(dir, normalized);
        }
    }

    private static void addDocs(final IndexWriter writer, final int from, final int to, final boolean normalized) throws Exception {
        for (int id = from; id < to; id++) {
            final Document doc = new Document();
            doc.add(new KnnFloatVectorField(FIELD, vector(id, normalized), VectorSimilarityFunction.COSINE));
            doc.add(new NumericDocValuesField(ID, id));
            writer.addDocument(doc);
        }
    }

    /** The index is one segment, and every doc's flat vector reads back exactly as it was indexed. */
    private static void assertRawVectors(final Directory dir, final boolean normalized) throws Exception {
        try (DirectoryReader reader = DirectoryReader.open(dir)) {
            assertEquals("one segment", 1, reader.leaves().size());
            int seen = 0;
            for (final LeafReaderContext context : reader.leaves()) {
                final LeafReader leaf = context.reader();
                final FloatVectorValues vectors = leaf.getFloatVectorValues(FIELD);
                final NumericDocValues ids = leaf.getNumericDocValues(ID);
                final KnnVectorValues.DocIndexIterator iterator = vectors.iterator();
                for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
                    assertTrue(ids.advanceExact(doc));
                    final int id = (int) ids.longValue();
                    assertArrayEquals(
                        "doc " + id + " keeps its indexed vector",
                        vector(id, normalized),
                        vectors.vectorValue(iterator.index()),
                        0f
                    );
                    seen++;
                }
            }
            assertEquals("every doc has a vector", NUM_DOCS, seen);
        }
    }

    /** {@link #DIRECTION} scaled by a per-doc magnitude above 1, or that direction at unit length. */
    private static float[] vector(final int id, final boolean normalized) {
        final float[] vector = new float[DIMENSION];
        for (int d = 0; d < DIMENSION; d++) {
            vector[d] = DIRECTION[d] * (id + 2);
        }
        return normalized ? VectorUtil.l2normalize(vector) : vector;
    }
}
