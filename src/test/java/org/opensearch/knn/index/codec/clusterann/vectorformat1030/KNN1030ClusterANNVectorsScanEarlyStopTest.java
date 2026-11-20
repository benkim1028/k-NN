/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.vectorformat1030;

import com.carrotsearch.randomizedtesting.RandomizedContext;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.SeededKnnVectorQuery;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.store.BaseDirectoryWrapper;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.opensearch.knn.clusterann.read.orchestration.EarlyStopExperiment;
import org.opensearch.knn.clusterann.read.orchestration.EarlyStopParams;
import org.opensearch.knn.clusterann.stats.ClusterANNStat;
import org.opensearch.knn.index.codec.clusterann.ClusterANN1030TestCodec;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

/**
 * The scan early stop end to end: index a clustered corpus through the format on a real on-disk {@link Directory},
 * search it through {@link IndexSearcher} with the stop turned on through the EXPERIMENT switch
 * ({@link EarlyStopExperiment}, which the reader reads once per segment search), and check that it scans
 * fewer clusters than the plan named while recall stays at or above {@link #MIN_RECALL} and within
 * {@link #MAX_RECALL_LOSS} of the full plan. The same queries are run with
 * the stop off as the reference. The directory's close runs {@code CheckIndex} with slow checks, which searches every
 * vector field; the test also runs it explicitly and prints its report.
 *
 * <p>The corpus is generated from fixed seeds ({@link #CORPUS_SEED}, {@link #QUERY_SEED}), so it is the same whatever
 * the test seed; the test seed only varies the directory implementation and is printed for reproduction.
 */
@LuceneTestCase.SuppressSysoutChecks(bugUrl = "The CheckIndex report and scan stats are the point of this test")
public class KNN1030ClusterANNVectorsScanEarlyStopTest extends LuceneTestCase {

    private static final String FIELD = "v";
    private static final String PARITY = "parity";
    private static final int DIMENSION = 16;
    /** About 100 clusters at the writer's 512-vector target, so the plan is 30 probes and the floor is 10. */
    private static final int NUM_DOCS = 51_200;
    /** Many more blobs than a plan visits, so the plan reaches well past the clusters that hold the neighbours. */
    private static final int NUM_BLOBS = 400;
    private static final float BLOB_SPREAD = 0.05f;
    private static final int NUM_QUERIES = 60;
    private static final int K = 10;
    private static final long CORPUS_SEED = 42L;
    private static final long QUERY_SEED = 7L;
    /**
     * Recall the stop must keep, unfiltered and under a half-selective filter. Absolute, and set by the corpus rather
     * than the stop: 4-bit codes on tight 16-dimensional blobs hold the full plan in the mid 0.7s here.
     */
    private static final double MIN_RECALL = 0.70;
    /** Recall the stop may cost against the full plan on the same queries. */
    private static final double MAX_RECALL_LOSS = 0.01;

    /** One segment of about 100 clusters: plan 30 probes, floor 10. */
    public void testSearch_whenTheStopIsOn_thenScansFewerClustersThanPlannedAndKeepsRecall() throws Exception {
        check(1);
    }

    /**
     * Four segments of about 25 clusters (plan 15, floor 10), searched as Lucene searches several segments (optimistic
     * per-segment k); every segment search must see the stop, {@code SeededKnnVectorQuery} included.
     */
    public void testSearch_whenTheStopIsOnOverSegments_thenEverySegmentSearchSeesIt() throws Exception {
        check(4);
    }

    private void check(int segments) throws Exception {
        final String seed = RandomizedContext.current().getRunnerSeedAsString();
        final float[][] vectors = corpus();
        final float[][] queries = queries(vectors);
        final List<Set<Integer>> exact = new ArrayList<>();
        final List<Set<Integer>> exactFiltered = new ArrayList<>();
        for (float[] query : queries) {
            exact.add(exactTopK(vectors, query, false));
            exactFiltered.add(exactTopK(vectors, query, true));
        }
        final EarlyStopParams stopOn = EarlyStopParams.PROVISIONAL;
        try (Directory dir = newFSDirectory(createTempDir("scan-early-stop"))) {
            assertTrue("CheckIndex on close", ((BaseDirectoryWrapper) dir).getCheckIndexOnClose());
            index(dir, vectors, NUM_DOCS / segments);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals("flushed in order, so doc ids are corpus rows", segments, reader.leaves().size());
                final IndexSearcher searcher = new IndexSearcher(reader);

                final Run off = run("off", searcher, queries, exact, null, false);
                final Run on = run("on", searcher, queries, exact, stopOn, false);
                final Run offFiltered = run("off-filtered", searcher, queries, exactFiltered, null, true);
                final Run onFiltered = run("on-filtered", searcher, queries, exactFiltered, stopOn, true);
                // SeededKnnVectorQuery searches every segment through its own seeded query; the switch is read there too.
                final Run onSeeded = run("on-seeded", searcher, queries, exact, stopOn, false, true);

                System.out.println(
                    String.format(
                        Locale.ROOT,
                        "scan-early-stop segments=%d tests.seed=%s corpusSeed=%d querySeed=%d",
                        segments,
                        seed,
                        CORPUS_SEED,
                        QUERY_SEED
                    )
                );
                for (Run r : new Run[] { off, on, offFiltered, onFiltered, onSeeded }) {
                    System.out.println("scan-early-stop " + r);
                }

                for (Run r : new Run[] { off, offFiltered }) {
                    assertEquals(r.name + " scans the whole plan", r.planned, r.scanned);
                    assertEquals(r.name + " never stops", 0, r.stops);
                    assertEquals(r.name + " records no stop fraction", 0, r.fractions);
                }
                for (Run[] pair : new Run[][] { { off, on }, { offFiltered, onFiltered }, { off, onSeeded } }) {
                    final Run o = pair[0];
                    final Run r = pair[1];
                    assertTrue(r.name + " scans fewer clusters than planned: " + r, r.scanned < r.planned);
                    assertTrue(r.name + " scans fewer clusters than off: " + r, r.scanned < o.scanned);
                    assertTrue(r.name + ", the stop fired: " + r, r.stops > 0);
                    assertEquals(r.name + ": every segment search saw the stop", r.searches, r.fractions);
                    assertEquals(r.name + ": probes cut by the stop are not empty probes skipped", 0, r.emptySkipped);
                    assertTrue(r.name + " recall@" + K + " " + r.recall + " below " + MIN_RECALL, r.recall >= MIN_RECALL);
                    assertTrue(
                        r.name + " recall " + r.recall + " more than " + MAX_RECALL_LOSS + " under off " + o.recall,
                        r.recall >= o.recall - MAX_RECALL_LOSS
                    );
                }
            }

            final ByteArrayOutputStream report = new ByteArrayOutputStream();
            TestUtil.checkIndex(dir, CheckIndex.Level.MIN_LEVEL_FOR_SLOW_CHECKS, true, false, report);
            final String checkIndex = report.toString(StandardCharsets.UTF_8);
            // Only the vector lines: the whole report would overrun the test framework's stdout limit.
            checkIndex.lines()
                .filter(line -> line.contains("test: vectors") || line.contains("No problems were detected"))
                .forEach(line -> System.out.println("scan-early-stop CheckIndex " + line.strip()));
            assertTrue(
                "CheckIndex checked the vector field:\n" + checkIndex,
                checkIndex.contains("test: vectors") && checkIndex.contains("OK [1 fields, " + (NUM_DOCS / segments) + " vectors]")
            );
        } // close runs CheckIndex again, at slow-check level
    }

    private static void index(Directory dir, float[][] vectors, int docsPerSegment) throws Exception {
        final IndexWriterConfig iwc = new IndexWriterConfig().setCodec(new ClusterANN1030TestCodec(4))
            .setMergePolicy(NoMergePolicy.INSTANCE)
            .setUseCompoundFile(false)
            .setRAMBufferSizeMB(256)
            .setMaxBufferedDocs(docsPerSegment);
        try (IndexWriter writer = new IndexWriter(dir, iwc)) {
            for (int i = 0; i < vectors.length; i++) {
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD, vectors[i], VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(PARITY, i % 2 == 0 ? "even" : "odd", Field.Store.NO));
                writer.addDocument(doc);
            }
            writer.commit();
        }
    }

    private static Run run(
        String name,
        IndexSearcher searcher,
        float[][] queries,
        List<Set<Integer>> exact,
        EarlyStopParams earlyStop,
        boolean filtered
    ) throws Exception {
        return run(name, searcher, queries, exact, earlyStop, filtered, false);
    }

    private static Run run(
        String name,
        IndexSearcher searcher,
        float[][] queries,
        List<Set<Integer>> exact,
        EarlyStopParams earlyStop,
        boolean filtered,
        boolean seeded
    ) throws Exception {
        final long planned0 = ClusterANNStat.PROBES_PLANNED.getTotal();
        final long searches0 = ClusterANNStat.PROBES_PLANNED.getCount();
        final long scanned0 = ClusterANNStat.CLUSTERS_SCANNED.getTotal();
        final long empty0 = ClusterANNStat.EMPTY_PROBES_SKIPPED.getTotal();
        final long stops0 = ClusterANNStat.SCAN_EARLY_STOP.getCount();
        final long fractions0 = ClusterANNStat.SCAN_EARLY_STOP_SCANNED_BPS.getCount();
        final Query filter = filtered ? new TermQuery(new Term(PARITY, "even")) : null;
        int found = 0;
        // EXPERIMENT (do not merge): the stop is a JVM-global switch here; null leaves it at its default, off.
        if (earlyStop != null) {
            EarlyStopExperiment.set(earlyStop);
        }
        try {
            for (int q = 0; q < queries.length; q++) {
                final KnnFloatVectorQuery knn = new KnnFloatVectorQuery(FIELD, queries[q], K, filter);
                final Query query = seeded ? SeededKnnVectorQuery.fromFloatQuery(knn, new TermQuery(new Term(PARITY, "odd"))) : knn;
                final TopDocs hits = searcher.search(query, K);
                assertEquals(K, hits.scoreDocs.length);
                for (ScoreDoc hit : hits.scoreDocs) {
                    if (exact.get(q).contains(hit.doc)) {
                        found++;
                    }
                }
            }
        } finally {
            EarlyStopExperiment.reset();
        }
        return new Run(
            name,
            (double) found / (K * queries.length),
            ClusterANNStat.PROBES_PLANNED.getTotal() - planned0,
            ClusterANNStat.PROBES_PLANNED.getCount() - searches0,
            ClusterANNStat.CLUSTERS_SCANNED.getTotal() - scanned0,
            ClusterANNStat.EMPTY_PROBES_SKIPPED.getTotal() - empty0,
            ClusterANNStat.SCAN_EARLY_STOP.getCount() - stops0,
            ClusterANNStat.SCAN_EARLY_STOP_SCANNED_BPS.getCount() - fractions0
        );
    }

    private record Run(String name, double recall, long planned, long searches, long scanned, long emptySkipped, long stops,
        long fractions) {
        @Override
        public String toString() {
            return String.format(
                Locale.ROOT,
                "%-12s recall@%d=%.3f segmentSearches=%d planned=%d scanned=%d (%.1f%%) stops=%d",
                name,
                K,
                recall,
                searches,
                planned,
                scanned,
                100.0 * scanned / Math.max(1, planned),
                stops
            );
        }
    }

    private static Set<Integer> exactTopK(float[][] vectors, float[] query, boolean filtered) {
        final VectorSimilarityFunction sim = VectorSimilarityFunction.EUCLIDEAN;
        final Set<Integer> topK = new HashSet<>();
        IntStream.range(0, vectors.length)
            .filter(i -> filtered == false || i % 2 == 0)
            .boxed()
            .sorted((a, b) -> Float.compare(sim.compare(query, vectors[b]), sim.compare(query, vectors[a])))
            .limit(K)
            .forEach(topK::add);
        return topK;
    }

    /** Gaussian blobs around uniformly placed centres, rows grouped by blob. */
    private static float[][] corpus() {
        final Random random = new Random(CORPUS_SEED);
        final float[][] centres = new float[NUM_BLOBS][DIMENSION];
        for (float[] c : centres) {
            for (int d = 0; d < DIMENSION; d++) {
                c[d] = random.nextFloat();
            }
        }
        final float[][] out = new float[NUM_DOCS][DIMENSION];
        for (int i = 0; i < NUM_DOCS; i++) {
            final float[] v = out[i];
            // Rows grouped by blob, so a multi-segment index keeps each blob in one segment and a query's neighbours sit
            // in one segment, as they would in a time- or tenant-ordered index.
            final float[] c = centres[(int) ((long) i * NUM_BLOBS / NUM_DOCS)];
            for (int d = 0; d < DIMENSION; d++) {
                v[d] = c[d] + (float) random.nextGaussian() * BLOB_SPREAD;
            }
        }
        return out;
    }

    /** Queries drawn from the corpus's own distribution: a corpus vector, nudged. */
    private static float[][] queries(float[][] vectors) {
        final Random random = new Random(QUERY_SEED);
        final float[][] out = new float[NUM_QUERIES][DIMENSION];
        for (float[] q : out) {
            final float[] v = vectors[random.nextInt(vectors.length)];
            for (int d = 0; d < DIMENSION; d++) {
                q[d] = v[d] + (float) random.nextGaussian() * BLOB_SPREAD;
            }
        }
        return out;
    }
}
