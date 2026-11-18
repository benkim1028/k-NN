/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.component;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.clusterann.stats.ClusterANNStat;
import org.opensearch.knn.index.codec.clusterann.ClusterANN1030TestCodec;

import java.util.EnumMap;
import java.util.Map;

/**
 * End-to-end component test that drives the real ClusterANN codec through flush, merge, and search and asserts the
 * {@link ClusterANNStat} counters advance accordingly. Unlike the holder unit tests (which poke the enum directly),
 * this exercises the production {@code KNN1030ClusterANNVectorsWriter}/{@code Reader} and the probe orchestration, so
 * it is the regression guard that the {@code record(...)} calls are actually on the hot paths.
 *
 * <p>The {@link ClusterANNStat} accumulators are process-global and monotonic, so every assertion is a <em>delta</em>
 * captured around one operation, never an absolute value — the test is therefore independent of anything else that
 * ran in the JVM first.
 */
@ThreadLeakFilters(defaultFilters = true, filters = {})
public class ClusterANNMetricsComponentTest extends LuceneTestCase {

    private static final String FIELD = "float_field";
    private static final int DIMENSION = 8;
    /** Comfortably above TARGET_CLUSTER_SIZE (512) so clustering is guaranteed to produce multiple centroids. */
    private static final int NUM_DOCS = 1200;

    /** Snapshot of (total, count) for every metric, for delta assertions. */
    private static Map<ClusterANNStat, long[]> snapshot() {
        final Map<ClusterANNStat, long[]> snap = new EnumMap<>(ClusterANNStat.class);
        for (final ClusterANNStat stat : ClusterANNStat.values()) {
            snap.put(stat, new long[] { stat.getTotal(), stat.getCount() });
        }
        return snap;
    }

    private static long totalDelta(final Map<ClusterANNStat, long[]> before, final ClusterANNStat stat) {
        return stat.getTotal() - before.get(stat)[0];
    }

    private static long countDelta(final Map<ClusterANNStat, long[]> before, final ClusterANNStat stat) {
        return stat.getCount() - before.get(stat)[1];
    }

    /** A flush of a single ClusterANN field advances the flush, cluster-build, encode, and structure metrics. */
    public void testFlush_thenWriteAndStructureMetricsAdvance() throws Exception {
        final Map<ClusterANNStat, long[]> before = snapshot();

        try (Directory dir = newDirectory()) {
            final IndexWriterConfig iwc = new IndexWriterConfig().setCodec(new ClusterANN1030TestCodec())
                .setUseCompoundFile(false)
                .setMergeScheduler(new SerialMergeScheduler());
            try (IndexWriter writer = new IndexWriter(dir, iwc)) {
                for (int i = 0; i < NUM_DOCS; i++) {
                    final Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD, vector(i), VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.commit(); // flush path
            }
        }

        assertTrue("at least one flush operation recorded", countDelta(before, ClusterANNStat.FLUSH_OPERATIONS) >= 1);
        assertEquals("all docs counted on flush", NUM_DOCS, totalDelta(before, ClusterANNStat.FLUSH_DOCS));
        assertTrue("flush cluster-build time recorded", countDelta(before, ClusterANNStat.FLUSH_CLUSTER_BUILD_TIME_MS) >= 1);
        assertTrue("flush encode time recorded", countDelta(before, ClusterANNStat.FLUSH_ENCODE_TIME_MS) >= 1);
        assertTrue("centroid count recorded", countDelta(before, ClusterANNStat.CENTROID_COUNT) >= 1);
        assertTrue(
            "more than one centroid for a " + NUM_DOCS + "-doc field (guards against under-clustering)",
            totalDelta(before, ClusterANNStat.CENTROID_COUNT) > 1
        );
        assertTrue("avg cluster size recorded", countDelta(before, ClusterANNStat.AVG_CLUSTER_SIZE) >= 1);
        assertTrue("no flush errors", totalDelta(before, ClusterANNStat.FLUSH_ERRORS) == 0);
    }

    /** A force-merge of two segments advances the merge metrics and the merge cluster/encode split. */
    public void testMerge_thenMergeMetricsAdvance() throws Exception {
        final Map<ClusterANNStat, long[]> before = snapshot();

        try (Directory dir = newDirectory()) {
            final Codec codec = new ClusterANN1030TestCodec();
            final IndexWriterConfig iwc = new IndexWriterConfig().setCodec(codec)
                .setUseCompoundFile(false)
                .setMergeScheduler(new SerialMergeScheduler());
            try (IndexWriter writer = new IndexWriter(dir, iwc)) {
                for (int i = 0; i < NUM_DOCS; i++) {
                    final Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD, vector(i), VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                    if (i == NUM_DOCS / 2) {
                        writer.commit(); // a second segment so forceMerge actually merges
                    }
                }
                writer.commit();
                writer.forceMerge(1);
            }
        }

        assertTrue("at least one merge operation recorded", countDelta(before, ClusterANNStat.MERGE_OPERATIONS) >= 1);
        assertEquals("all docs counted on merge", NUM_DOCS, totalDelta(before, ClusterANNStat.MERGE_DOCS));
        assertTrue("merge cluster-build time recorded", countDelta(before, ClusterANNStat.MERGE_CLUSTER_BUILD_TIME_MS) >= 1);
        assertTrue("merge encode time recorded", countDelta(before, ClusterANNStat.MERGE_ENCODE_TIME_MS) >= 1);
        assertEquals("merge gauge returns to zero", 0, totalDelta(before, ClusterANNStat.MERGE_CURRENT_OPERATIONS));
        assertTrue("no merge errors", totalDelta(before, ClusterANNStat.MERGE_ERRORS) == 0);
        assertEquals("no merge aborted", 0, totalDelta(before, ClusterANNStat.MERGE_ABORTED));
    }

    /** An unfiltered search advances the query and probe/scan metrics. */
    public void testSearch_thenQueryAndProbeMetricsAdvance() throws Exception {
        try (Directory dir = buildIndex()) {
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final Map<ClusterANNStat, long[]> before = snapshot();

                reader.leaves()
                    .get(0)
                    .reader()
                    .searchNearestVectors(FIELD, vector(7), 10, AcceptDocs.fromLiveDocs(null, NUM_DOCS), Integer.MAX_VALUE);

                assertEquals("one query request", 1, countDelta(before, ClusterANNStat.QUERY_REQUESTS));
                assertEquals("no query errors", 0, totalDelta(before, ClusterANNStat.QUERY_ERRORS));
                assertEquals("query latency recorded once", 1, countDelta(before, ClusterANNStat.QUERY_LATENCY_US));
                assertEquals("probes planned recorded once", 1, countDelta(before, ClusterANNStat.PROBES_PLANNED));
                assertEquals("clusters scanned recorded once", 1, countDelta(before, ClusterANNStat.CLUSTERS_SCANNED));
                assertEquals("vectors collected recorded once", 1, countDelta(before, ClusterANNStat.VECTORS_COLLECTED_PER_QUERY));
                assertTrue("at least some vectors collected", totalDelta(before, ClusterANNStat.VECTORS_COLLECTED_PER_QUERY) >= 1);
                assertEquals(
                    "unfiltered query records no filter request",
                    0,
                    countDelta(before, ClusterANNStat.QUERY_WITH_FILTER_REQUESTS)
                );
            }
        }
    }

    /** A filtered search additionally advances the filter selectivity / eligible-cluster metrics. */
    public void testFilteredSearch_thenFilterMetricsAdvance() throws Exception {
        try (Directory dir = buildIndex()) {
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final Map<ClusterANNStat, long[]> before = snapshot();

                final FixedBitSet accepted = new FixedBitSet(NUM_DOCS);
                accepted.set(0, NUM_DOCS / 2); // half the docs pass the filter
                reader.leaves()
                    .get(0)
                    .reader()
                    .searchNearestVectors(FIELD, vector(3), 10, AcceptDocs.fromLiveDocs(accepted, NUM_DOCS), Integer.MAX_VALUE);

                assertEquals("one query request", 1, countDelta(before, ClusterANNStat.QUERY_REQUESTS));
                assertEquals("one filtered query request", 1, countDelta(before, ClusterANNStat.QUERY_WITH_FILTER_REQUESTS));
                assertEquals("filter selectivity recorded once", 1, countDelta(before, ClusterANNStat.FILTER_SELECTIVITY_BPS));
                assertEquals("filter probe multiplier recorded once", 1, countDelta(before, ClusterANNStat.FILTER_PROBE_MULTIPLIER_BPS));
                assertEquals("eligible clusters recorded once", 1, countDelta(before, ClusterANNStat.FILTER_ELIGIBLE_CLUSTERS));
            }
        }
    }

    private Directory buildIndex() throws Exception {
        final Directory dir = newDirectory();
        final IndexWriterConfig iwc = new IndexWriterConfig().setCodec(new ClusterANN1030TestCodec())
            .setUseCompoundFile(false)
            .setMergeScheduler(new SerialMergeScheduler());
        try (IndexWriter writer = new IndexWriter(dir, iwc)) {
            for (int i = 0; i < NUM_DOCS; i++) {
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD, vector(i), VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc);
            }
            writer.commit();
            writer.forceMerge(1);
        }
        return dir;
    }

    private static float[] vector(final int seed) {
        final float[] vector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            // Spread across a wide range so k-means forms several distinct centroids.
            vector[i] = ((seed * 31 + i * 7) % 97) * 0.1f;
        }
        return vector;
    }
}
