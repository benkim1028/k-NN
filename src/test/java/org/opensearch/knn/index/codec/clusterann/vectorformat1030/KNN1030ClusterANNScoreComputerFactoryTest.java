/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.vectorformat1030;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.lucene103.Lucene103Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.opensearch.knn.clusterann.format.ClusterANNEncoding;
import org.opensearch.knn.clusterann.format.ClusterANNFieldMeta;
import org.opensearch.knn.clusterann.format.QuantizationParams;
import org.opensearch.knn.clusterann.read.ComputeScoreContext;
import org.opensearch.knn.clusterann.read.ScoreComputer;
import org.opensearch.knn.clusterann.read.ScoreComputerFactory;
import org.opensearch.knn.clusterann.read.block.scalar.JavaAdcScoreComputer;
import org.opensearch.knn.index.codec.clusterann.ClusterANN1030TestCodec;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The {@link ScoreComputerFactory} handed to {@link KNN1030ClusterANNVectorsReader}'s constructor reaches the search:
 * it is asked once per field when the segment opens, with that field's quantizer and metadata, and the computer it
 * returns is what scores the search. A {@code null} answer, and the no-factory constructors, score in Java.
 *
 * <p>The index is read back through the writer ({@link DirectoryReader#open(IndexWriter)}), which reuses the codec
 * instance the segment was written with; a reader opened from the directory resolves the codec by SPI name and so
 * takes the Java path, which is what the equality check relies on.
 */
public class KNN1030ClusterANNScoreComputerFactoryTest extends LuceneTestCase {

    private static final String FIELD = "vector";
    private static final int DIMENSION = 8;
    private static final int NUM_DOCS = 200;
    private static final int K = 10;

    public void testFactory_isAskedOncePerField_andItsComputerScoresTheSearch() throws Exception {
        final RecordingFactory factory = new RecordingFactory(new CountingComputer(JavaAdcScoreComputer.INSTANCE));

        try (Directory dir = newDirectory()) {
            try (IndexWriter writer = new IndexWriter(dir, config(codec(factory)))) {
                index(writer);
                assertEquals("the factory is for reading, not writing", 0, factory.calls.size());

                try (DirectoryReader nrt = DirectoryReader.open(writer)) {
                    assertEquals("one segment, so one field to resolve", 1, factory.calls.size());
                    final RecordingFactory.Call call = factory.calls.get(0);
                    assertEquals(ClusterANNEncoding.OPTIMIZED_SCALAR_QUANTIZATION, call.quantizer);
                    assertEquals(DIMENSION, call.meta.dimension());
                    assertEquals(QuantizationParams.DEFAULT.docBits(), call.meta.docBits());
                    assertEquals(VectorSimilarityFunction.EUCLIDEAN, call.meta.similarityFunction());
                    assertEquals(NUM_DOCS, call.meta.vectorCount());

                    final TopDocs viaFactory = search(nrt);
                    assertTrue("the supplied computer must have scored blocks", factory.computer.scoreCalls.get() > 0);
                    assertEquals("still one resolution after searching", 1, factory.calls.size());

                    try (DirectoryReader bySpi = DirectoryReader.open(dir)) {
                        assertSameHits(search(bySpi), viaFactory);
                    }
                }
            }
        }
    }

    public void testFactory_whenItReturnsNull_thenTheJavaComputerScores() throws Exception {
        final RecordingFactory factory = new RecordingFactory(null);

        try (Directory dir = newDirectory()) {
            try (IndexWriter writer = new IndexWriter(dir, config(codec(factory)))) {
                index(writer);
                try (DirectoryReader nrt = DirectoryReader.open(writer); DirectoryReader bySpi = DirectoryReader.open(dir)) {
                    assertEquals(1, factory.calls.size());
                    assertSameHits(search(bySpi), search(nrt));
                }
            }
        }
    }

    public void testNone_returnsNull() {
        assertNull(ScoreComputerFactory.NONE.create(ClusterANNEncoding.OPTIMIZED_SCALAR_QUANTIZATION, null));
    }

    // ---------------------------------------------------------------- fixtures

    /** Records every resolution and hands out one computer (or {@code null}) for all of them. */
    private static final class RecordingFactory implements ScoreComputerFactory {
        record Call(ClusterANNEncoding quantizer, ClusterANNFieldMeta meta) {
        }

        final List<Call> calls = new ArrayList<>();
        final CountingComputer computer;

        RecordingFactory(CountingComputer computer) {
            this.computer = computer;
        }

        @Override
        public ScoreComputer create(ClusterANNEncoding quantizer, ClusterANNFieldMeta meta) {
            calls.add(new Call(quantizer, meta));
            return computer;
        }
    }

    /** Delegates the arithmetic and counts the calls, so the test can tell it ran without changing any score. */
    private static final class CountingComputer implements ScoreComputer {
        final AtomicInteger scoreCalls = new AtomicInteger();
        private final ScoreComputer delegate;

        CountingComputer(ScoreComputer delegate) {
            this.delegate = delegate;
        }

        @Override
        public float score(ComputeScoreContext context, int[] positions, int count, float[] scores) {
            scoreCalls.incrementAndGet();
            return delegate.score(context, positions, count, scores);
        }
    }

    /**
     * The test codec's name, so a reader opened from the directory resolves to {@link ClusterANN1030TestCodec} (the
     * Java path), but with a reader built with {@code factory} for the reader opened through the writer.
     */
    private static Codec codec(final ScoreComputerFactory factory) {
        return new FilterCodec(ClusterANN1030TestCodec.CODEC_NAME, new Lucene103Codec()) {
            @Override
            public KnnVectorsFormat knnVectorsFormat() {
                final KNN1030ClusterANNVectorsFormat writeSide = new KNN1030ClusterANNVectorsFormat(
                    KNN1030ClusterANNVectorsFormat.FORMAT_NAME,
                    QuantizationParams.DEFAULT
                );
                final FlatVectorsFormat raw = new Lucene99FlatVectorsFormat(FlatVectorScorerUtil.getLucene99FlatVectorsScorer());
                return new KnnVectorsFormat(KNN1030ClusterANNVectorsFormat.FORMAT_NAME) {
                    @Override
                    public KnnVectorsWriter fieldsWriter(final SegmentWriteState state) throws IOException {
                        return writeSide.fieldsWriter(state);
                    }

                    @Override
                    public KnnVectorsReader fieldsReader(final SegmentReadState state) throws IOException {
                        return new KNN1030ClusterANNVectorsReader(state, raw.fieldsReader(state), factory);
                    }

                    @Override
                    public int getMaxDimensions(final String fieldName) {
                        return writeSide.getMaxDimensions(fieldName);
                    }
                };
            }
        };
    }

    private static IndexWriterConfig config(final Codec codec) {
        // One flushed segment, sidecars as separate files: see KNN1030ClusterANNVectorsWriterFlushTest.
        return newIndexWriterConfig().setCodec(codec)
            .setMergePolicy(NoMergePolicy.INSTANCE)
            .setUseCompoundFile(false)
            .setMaxBufferedDocs(NUM_DOCS + 1);
    }

    private static void index(final IndexWriter writer) throws Exception {
        final Random random = new Random(42);
        for (int i = 0; i < NUM_DOCS; i++) {
            final Document doc = new Document();
            doc.add(new KnnFloatVectorField(FIELD, vector(random), VectorSimilarityFunction.EUCLIDEAN));
            writer.addDocument(doc);
        }
        writer.commit();
    }

    private static TopDocs search(final DirectoryReader reader) throws Exception {
        return new IndexSearcher(reader).search(new KnnFloatVectorQuery(FIELD, vector(new Random(7)), K), K);
    }

    private static void assertSameHits(final TopDocs expected, final TopDocs actual) {
        assertEquals(expected.scoreDocs.length, actual.scoreDocs.length);
        for (int i = 0; i < expected.scoreDocs.length; i++) {
            final ScoreDoc e = expected.scoreDocs[i];
            final ScoreDoc a = actual.scoreDocs[i];
            assertEquals("hit " + i + " doc", e.doc, a.doc);
            assertEquals("hit " + i + " score", Float.floatToIntBits(e.score), Float.floatToIntBits(a.score));
        }
    }

    private static float[] vector(final Random random) {
        final float[] v = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            v[i] = (float) random.nextGaussian();
        }
        return v;
    }
}
