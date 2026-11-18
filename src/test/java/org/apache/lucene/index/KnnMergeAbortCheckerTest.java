/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.apache.lucene.index;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KnnMergeAbortCheckerTest {

    @Test
    void isMergeAborted_offMergeThread_isFalse() {
        assertFalse(KnnMergeAbortChecker.isMergeAborted());
    }

    @Test
    void isMergeAborted_onMergeThreadWithLiveMerge_isFalse() throws Exception {
        assertFalse(probeOnMergeThread(oneMerge()));
    }

    @Test
    void isMergeAborted_onMergeThreadWithAbortedMerge_isTrue() throws Exception {
        final MergePolicy.OneMerge merge = oneMerge();
        merge.setAborted();

        assertTrue(probeOnMergeThread(merge));
    }

    /** Runs the check on a real merge thread whose body is the probe alone. */
    private static boolean probeOnMergeThread(final MergePolicy.OneMerge merge) throws Exception {
        final AtomicReference<Boolean> result = new AtomicReference<>();
        try (ConcurrentMergeScheduler scheduler = new ConcurrentMergeScheduler()) {
            final Thread thread = scheduler.new MergeThread(new NoMergeSource(), merge) {
                @Override
                public void run() {
                    result.set(KnnMergeAbortChecker.isMergeAborted());
                }
            };
            thread.start();
            thread.join();
        }
        return result.get();
    }

    private static MergePolicy.OneMerge oneMerge() throws IOException {
        try (Directory dir = new ByteBuffersDirectory()) {
            try (IndexWriter writer = new IndexWriter(dir, new IndexWriterConfig())) {
                final Document doc = new Document();
                doc.add(new StringField("id", "1", Field.Store.NO));
                writer.addDocument(doc);
                writer.commit();
            }
            return new MergePolicy.OneMerge(SegmentInfos.readLatestCommit(dir).asList());
        }
    }

    private static final class NoMergeSource implements MergeScheduler.MergeSource {
        @Override
        public MergePolicy.OneMerge getNextMerge() {
            return null;
        }

        @Override
        public void onMergeFinished(final MergePolicy.OneMerge merge) {}

        @Override
        public boolean hasPendingMerges() {
            return false;
        }

        @Override
        public void merge(final MergePolicy.OneMerge merge) {}
    }

    @Test
    void ensureMergeNotAborted_offMergeThread_passes() {
        assertDoesNotThrow(KnnMergeAbortChecker::ensureMergeNotAborted);
    }

    @Test
    void ensureMergeNotAborted_onAbortedMergeThread_throws() {
        assertThrows(MergePolicy.MergeAbortedException.class, () -> AbortedMergeThread.run(KnnMergeAbortChecker::ensureMergeNotAborted));
    }
}
