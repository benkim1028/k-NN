/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.vectorformat1030;

import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.ConcurrentMergeScheduler;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.ThreadInterruptedException;
import org.opensearch.knn.clusterann.stats.ClusterANNStat;
import org.opensearch.knn.index.codec.clusterann.ClusterANN1030TestCodec;

import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Rolls back an {@link IndexWriter} while a {@code ConcurrentMergeScheduler} merge is inside the ClusterANN vector
 * writer, as the engine does when a shard is closed or relocated mid-merge, and asserts the build stops and is counted
 * as an abort rather than a failure.
 *
 * <p>Each case:
 * <ol>
 *   <li>writes {@link #SEGMENTS} segments with {@link NoMergePolicy}, optionally deleting every other document so the
 *       merge drops deleted documents before it clusters;</li>
 *   <li>reopens the index with a {@code ConcurrentMergeScheduler} and runs {@code forceMerge(1)} on a background
 *       thread;</li>
 *   <li>holds the merge thread as it reaches the vector writer's {@code mergeOneField};</li>
 *   <li>calls {@link IndexWriter#rollback()} from the test thread, which marks the running merge aborted and waits for
 *       it;</li>
 *   <li>releases the merge thread once {@code IndexWriter#abortMerges} logs {@value #MERGES_ABORTED_MESSAGE} to the
 *       {@code IW} info stream, which it does after marking every running merge aborted, so the abort always lands
 *       inside the ClusterANN build regardless of segment size;</li>
 *   <li>asserts the abort is counted in {@code clusterann_merge_aborted}, no merge or error is counted, the in-flight
 *       gauge returns to zero and the aborted merge leaves no cluster files behind.</li>
 * </ol>
 *
 * <p>The structure follows Lucene's {@code TestHnswMergeAbort}, added with apache/lucene#16368 (merge abort checks
 * during HNSW graph construction) and made deterministic in apache/lucene#16401: rollback during a background
 * {@code forceMerge}, holding the merge thread until the {@code IW} "now wait for" message confirms the abort, and a
 * variant with half the documents deleted. Lucene holds on its HNSW "build graph" info-stream message and can only
 * assert that the graph build never finished, because Lucene swallows the abort; the ClusterANN writer logs no
 * info-stream messages, so this test holds in a {@code mergeOneField} wrapper instead and asserts the abort through
 * the ClusterANN merge metrics.
 */
public class KNN1030ClusterANNVectorsWriterMergeAbortTest extends LuceneTestCase {

    private static final String FIELD = "vector";
    private static final String ID = "id";
    private static final int DIMENSION = 8;
    private static final int SEGMENTS = 4;
    private static final int DOCS_PER_SEGMENT = 200;
    /** Logged by {@code IndexWriter#abortMerges} once every running merge has been marked aborted. */
    private static final String MERGES_ABORTED_MESSAGE = "now wait for";

    public void testRollbackDuringMerge_thenBuildStopsAndCountsAsAbort() throws Exception {
        doTestRollbackDuringMerge(false);
    }

    /** Half of every source segment is deleted, so the merge drops deleted documents before it clusters. */
    public void testRollbackDuringMergeWithDeletes_thenBuildStopsAndCountsAsAbort() throws Exception {
        doTestRollbackDuringMerge(true);
    }

    private void doTestRollbackDuringMerge(final boolean withDeletes) throws Exception {
        final long abortedBefore = ClusterANNStat.MERGE_ABORTED.getTotal();
        final long operationsBefore = ClusterANNStat.MERGE_OPERATIONS.getTotal();
        final long errorsBefore = ClusterANNStat.MERGE_ERRORS.getTotal();
        final long inFlightBefore = ClusterANNStat.MERGE_CURRENT_OPERATIONS.getTotal();

        try (Directory dir = newDirectory()) {
            writeSegments(dir, withDeletes);

            final CountDownLatch buildStarted = new CountDownLatch(1);
            final CountDownLatch mergeAborted = new CountDownLatch(1);
            final AtomicBoolean releasedAfterAbort = new AtomicBoolean();
            final Runnable holdUntilAborted = () -> {
                if (buildStarted.getCount() > 0) {
                    buildStarted.countDown();
                    try {
                        releasedAfterAbort.set(mergeAborted.await(2, TimeUnit.MINUTES));
                    } catch (final InterruptedException e) {
                        throw new ThreadInterruptedException(e);
                    }
                }
            };

            final IndexWriterConfig iwc = new IndexWriterConfig().setCodec(new HoldingCodec(holdUntilAborted))
                .setUseCompoundFile(false)
                .setMergeScheduler(new ConcurrentMergeScheduler())
                .setInfoStream(new AbortListeningInfoStream(mergeAborted));
            final IndexWriter writer = new IndexWriter(dir, iwc);
            final Thread merger = new Thread(() -> {
                try {
                    writer.forceMerge(1);
                } catch (final Throwable expected) {
                    // forceMerge may surface the rollback; the merge itself is checked through the metrics below
                }
            });
            merger.start();
            try {
                assertTrue("the merge never reached the ClusterANN writer", buildStarted.await(1, TimeUnit.MINUTES));
                writer.rollback();
            } finally {
                merger.join(TimeUnit.MINUTES.toMillis(2));
            }
            assertFalse("merge thread did not terminate", merger.isAlive());
            assertTrue("the merge was released before it was marked aborted", releasedAfterAbort.get());
            assertEquals(
                "the source segments remain and the aborted merge left no cluster files",
                SEGMENTS,
                Arrays.stream(dir.listAll()).filter(f -> f.endsWith(".clam")).count()
            );
        }

        assertEquals("one merge aborted", 1, ClusterANNStat.MERGE_ABORTED.getTotal() - abortedBefore);
        assertEquals("no merge completed", 0, ClusterANNStat.MERGE_OPERATIONS.getTotal() - operationsBefore);
        assertEquals("an abort is not an error", 0, ClusterANNStat.MERGE_ERRORS.getTotal() - errorsBefore);
        assertEquals("merge gauge returns to zero", 0, ClusterANNStat.MERGE_CURRENT_OPERATIONS.getTotal() - inFlightBefore);
    }

    private static void writeSegments(final Directory dir, final boolean withDeletes) throws IOException {
        final IndexWriterConfig iwc = new IndexWriterConfig().setCodec(new ClusterANN1030TestCodec())
            .setUseCompoundFile(false)
            .setMergePolicy(NoMergePolicy.INSTANCE);
        try (IndexWriter writer = new IndexWriter(dir, iwc)) {
            int id = 0;
            for (int segment = 0; segment < SEGMENTS; segment++) {
                for (int i = 0; i < DOCS_PER_SEGMENT; i++, id++) {
                    final Document doc = new Document();
                    doc.add(new StringField(ID, Integer.toString(id), Field.Store.NO));
                    doc.add(new KnnFloatVectorField(FIELD, vector(id), VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.flush();
            }
            if (withDeletes) {
                for (int d = 0; d < id; d += 2) {
                    writer.deleteDocuments(new Term(ID, Integer.toString(d)));
                }
            }
            writer.commit();
        }
    }

    private static float[] vector(final int seed) {
        final float[] vector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            vector[i] = ((seed * 31 + i * 7) % 97) * 0.1f;
        }
        return vector;
    }

    /** Counts down once {@code IndexWriter#abortMerges} reports that every running merge is marked aborted. */
    private static final class AbortListeningInfoStream extends InfoStream {
        private final CountDownLatch mergeAborted;

        AbortListeningInfoStream(final CountDownLatch mergeAborted) {
            this.mergeAborted = mergeAborted;
        }

        @Override
        public void message(final String component, final String message) {
            if ("IW".equals(component) && message.startsWith(MERGES_ABORTED_MESSAGE)) {
                mergeAborted.countDown();
            }
        }

        @Override
        public boolean isEnabled(final String component) {
            return "IW".equals(component);
        }

        @Override
        public void close() {}
    }

    /** The ClusterANN test codec with a hook that runs when a merge reaches the vector writer. */
    private static final class HoldingCodec extends FilterCodec {
        private final Runnable onMerge;

        HoldingCodec(final Runnable onMerge) {
            super(ClusterANN1030TestCodec.CODEC_NAME, new ClusterANN1030TestCodec());
            this.onMerge = onMerge;
        }

        @Override
        public KnnVectorsFormat knnVectorsFormat() {
            return new HoldingFormat(delegate.knnVectorsFormat(), onMerge);
        }
    }

    private static final class HoldingFormat extends KnnVectorsFormat {
        private final KnnVectorsFormat delegate;
        private final Runnable onMerge;

        HoldingFormat(final KnnVectorsFormat delegate, final Runnable onMerge) {
            super(delegate.getName());
            this.delegate = delegate;
            this.onMerge = onMerge;
        }

        @Override
        public KnnVectorsWriter fieldsWriter(final SegmentWriteState state) throws IOException {
            return new HoldingWriter(delegate.fieldsWriter(state), onMerge);
        }

        @Override
        public KnnVectorsReader fieldsReader(final SegmentReadState state) throws IOException {
            return delegate.fieldsReader(state);
        }

        @Override
        public int getMaxDimensions(final String fieldName) {
            return delegate.getMaxDimensions(fieldName);
        }
    }

    private static final class HoldingWriter extends KnnVectorsWriter {
        private final KnnVectorsWriter delegate;
        private final Runnable onMerge;

        HoldingWriter(final KnnVectorsWriter delegate, final Runnable onMerge) {
            this.delegate = delegate;
            this.onMerge = onMerge;
        }

        @Override
        public void mergeOneField(final FieldInfo fieldInfo, final MergeState mergeState) throws IOException {
            onMerge.run();
            delegate.mergeOneField(fieldInfo, mergeState);
        }

        @Override
        public KnnFieldVectorsWriter<?> addField(final FieldInfo fieldInfo) throws IOException {
            return delegate.addField(fieldInfo);
        }

        @Override
        public void flush(final int maxDoc, final Sorter.DocMap sortMap) throws IOException {
            delegate.flush(maxDoc, sortMap);
        }

        @Override
        public void finish() throws IOException {
            delegate.finish();
        }

        @Override
        public void close() throws IOException {
            delegate.close();
        }

        @Override
        public long ramBytesUsed() {
            return delegate.ramBytesUsed();
        }
    }
}
