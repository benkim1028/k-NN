/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.Configurator;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TimeLimitingKnnCollectorManager;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.opensearch.knn.clusterann.format.ClusterANNEncoding;
import org.opensearch.knn.clusterann.read.Clusters;
import org.opensearch.knn.clusterann.read.orchestration.FilterDigest;
import org.opensearch.knn.clusterann.read.orchestration.ShardPlanExperiment;
import org.opensearch.knn.clusterann.read.orchestration.ShardPlanner;
import org.opensearch.knn.clusterann.stats.ClusterANNStat;
import org.opensearch.knn.index.codec.clusterann.ClusterANN1030TestCodec;
import org.opensearch.knn.index.codec.clusterann.component.corpus.Corpora;
import org.opensearch.knn.index.codec.clusterann.component.corpus.Corpus;
import org.opensearch.knn.index.codec.clusterann.component.suite.Scenario;
import org.opensearch.knn.index.codec.clusterann.component.suite.Suite;
import org.opensearch.knn.index.codec.clusterann.vectorformat1030.KNN1030ClusterANNVectorsReader;
import org.opensearch.knn.index.query.KNNScorer;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Recall and cost of shard-level planning against today's per-segment planning, through the real query path. Not a
 * test: skipped unless {@code -Dknn.shardPlanEval=true}.
 *
 * <p>For each suite scenario and each forced segment count it indexes the corpus with {@code NoMergePolicy} and one
 * explicit flush per segment, then runs the same queries three ways through {@link IndexSearcher#search}:
 *
 * <ul>
 *   <li>{@code today}: the reader searched with no shard plan on the collector, so each segment plans itself (the
 *       weight as it was before shard planning, reproduced in {@link TodayQuery});
 *   <li>{@code H}: {@link ClusterANNQuery}, default budget;
 *   <li>{@code R}: {@link ClusterANNQuery}, uncapped rule R.
 * </ul>
 *
 * <p>Filters are the shard-budget evaluation's: a seeded uniform bucket per row, {@code bucket < s * 10000}, so the
 * accepted sets match it. Ground truth is the corpus's neighbours unfiltered, exact brute force over accepted rows
 * filtered. Vectors scored is {@link ClusterANNStat#VECTORS_SCORED}; probes is {@link ClusterANNStat#PROBES_PLANNED};
 * centroids scored is counted from the plan (each ranking scores every candidate centroid: non-empty unfiltered,
 * eligible filtered), so shard planning pays the shard ranking plus each searched segment's re-rank.
 *
 * <pre>
 * ./gradlew clusterAnnTest --no-daemon --tests '*ShardPlanEvalRun' -Dknn.shardPlanEval=true \
 *   -Dknn.suite=src/test/resources/shard-plan-1k-suite.yml -Dknn.shardPlanEval.out=build/shard-plan-eval
 * </pre>
 * {@code -Dknn.shardPlanEval.segments} (default 1,4,16), {@code .filters} (default 1,0.1,0.01), {@code .layout=uneven}.
 */
class ShardPlanEvalRun {
    private static final Logger log = LogManager.getLogger(ShardPlanEvalRun.class);

    private static final String FIELD = "float_field";
    private static final String ROW = "row";
    private static final String BUCKET = "bucket";
    private static final int BUCKETS = 10_000;
    private static final long FILTER_SEED = 0x5eedL;
    private static final boolean UNEVEN = "uneven".equals(System.getProperty("knn.shardPlanEval.layout", "equal"));
    private static final String[] OPTIONS = { "today", "H", "R" };

    static {
        Configurator.setLevel(ShardPlanEvalRun.class.getName(), Level.INFO);
    }

    private record Result(double recall, double probes, double vectors, double centroids, double p50us, double p90us, double zeroSegFrac) {
    }

    @Test
    @EnabledIfSystemProperty(named = "knn.shardPlanEval", matches = "true")
    void evaluate() throws Exception {
        // EXPERIMENT (do not merge): H and R rows are labelled by budget alone, so the other switches must be the defaults.
        ShardPlanExperiment.reset();
        final int[] segmentCounts = Arrays.stream(System.getProperty("knn.shardPlanEval.segments", "1,4,16").split(","))
            .mapToInt(s -> Integer.parseInt(s.trim()))
            .toArray();
        final double[] filters = Arrays.stream(System.getProperty("knn.shardPlanEval.filters", "1,0.1,0.01").split(","))
            .mapToDouble(s -> Double.parseDouble(s.trim()))
            .toArray();
        final Path outDir = Paths.get(System.getProperty("knn.shardPlanEval.out", "build/shard-plan-eval"));
        Files.createDirectories(outDir);

        for (final Scenario scenario : Suite.load()) {
            final Corpus corpus = Corpora.of(scenario.corpus());
            final int k = scenario.search().k();
            final int queryCount = Math.min(corpus.queries(), scenario.search().queries());
            final int warmup = Math.max(0, Math.min(scenario.search().warmup(), queryCount));
            final String tag = corpus.name() + (UNEVEN ? "-uneven" : "");
            log.info(
                "== {} {}: {} x {}d {}, k={}, queries={}",
                scenario.name(),
                tag,
                corpus.size(),
                corpus.dimension(),
                corpus.similarity(),
                k,
                queryCount
            );
            final float[][] queries = new float[queryCount][];
            for (int q = 0; q < queryCount; q++) {
                queries[q] = corpus.query(q).clone();
            }
            final List<Set<Integer>>[] truth = groundTruth(corpus, filters, queries, k);

            final Path csv = outDir.resolve("shard-plan-eval-" + scenario.name() + "-" + tag + ".csv");
            try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(csv, StandardCharsets.UTF_8))) {
                out.println(
                    "corpus,segments,c_total,filter,option,queries,k,recall,probes,vectors_scored,centroids_scored,p50_us,p90_us,zero_seg_frac"
                );
                for (final int segments : segmentCounts) {
                    runCell(scenario, corpus, tag, segments, filters, queries, truth, k, warmup, out);
                    out.flush();
                }
            }
            log.info("wrote {}", csv.toAbsolutePath());
        }
    }

    private void runCell(
        final Scenario scenario,
        final Corpus corpus,
        final String tag,
        final int segmentCount,
        final double[] filters,
        final float[][] queries,
        final List<Set<Integer>>[] truth,
        final int k,
        final int warmup,
        final PrintWriter out
    ) throws Exception {
        final Path dirPath = Files.createTempDirectory("shard-plan-eval-");
        try (Directory dir = new MMapDirectory(dirPath)) {
            final int[] ends = segmentEnds(corpus.size(), segmentCount, UNEVEN);
            try (
                IndexWriter writer = new IndexWriter(
                    dir,
                    new IndexWriterConfig().setCodec(codec(scenario))
                        .setMaxBufferedDocs(corpus.size() + 1)
                        .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
                        .setMergePolicy(NoMergePolicy.INSTANCE)
                        .setUseCompoundFile(false)
                )
            ) {
                int next = 0;
                for (int row = 0; row < corpus.size(); row++) {
                    final Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD, corpus.vector(row), corpus.similarity()));
                    doc.add(new StoredField(ROW, row));
                    doc.add(new IntPoint(BUCKET, bucket(row)));
                    writer.addDocument(doc);
                    if (row + 1 == ends[next]) {
                        writer.flush();
                        next++;
                    }
                }
                writer.commit();
            }
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals(segmentCount, reader.leaves().size(), "forced segment count");
                int totalClusters = 0;
                for (final LeafReaderContext leaf : reader.leaves()) {
                    totalClusters += clusters(leaf).numClusters();
                    final int last = leaf.reader().maxDoc() - 1;
                    assertEquals(leaf.docBase + last, leaf.reader().storedFields().document(last).getField(ROW).numericValue().intValue());
                }
                log.info("[{} S={}] clusters total {}", tag, segmentCount, totalClusters);
                final IndexSearcher searcher = new IndexSearcher(reader);
                for (int f = 0; f < filters.length; f++) {
                    final Query filter = filters[f] >= 1.0 ? null : IntPoint.newRangeQuery(BUCKET, 0, threshold(filters[f]) - 1);
                    final int[] candidates = candidateCentroids(reader, filter, searcher);
                    for (final String option : OPTIONS) {
                        final Result r = run(searcher, option, filter, queries, truth[f], k, warmup, candidates);
                        out.printf(
                            Locale.ROOT,
                            "%s,%d,%d,%s,%s,%d,%d,%.5f,%.2f,%.1f,%.1f,%.1f,%.1f,%.4f%n",
                            tag,
                            segmentCount,
                            totalClusters,
                            fmt(filters[f]),
                            option,
                            queries.length,
                            k,
                            r.recall(),
                            r.probes(),
                            r.vectors(),
                            r.centroids(),
                            r.p50us(),
                            r.p90us(),
                            r.zeroSegFrac()
                        );
                        log.info(
                            "[{} S={} f={}] {} recall {} probes {} vectors {} centroids {} p50 {}us p90 {}us zeroSeg {}",
                            tag,
                            segmentCount,
                            fmt(filters[f]),
                            option,
                            String.format(Locale.ROOT, "%.4f", r.recall()),
                            String.format(Locale.ROOT, "%.1f", r.probes()),
                            String.format(Locale.ROOT, "%.0f", r.vectors()),
                            String.format(Locale.ROOT, "%.0f", r.centroids()),
                            String.format(Locale.ROOT, "%.0f", r.p50us()),
                            String.format(Locale.ROOT, "%.0f", r.p90us()),
                            String.format(Locale.ROOT, "%.3f", r.zeroSegFrac())
                        );
                    }
                }
            }
        } finally {
            try (var walk = Files.walk(dirPath)) {
                walk.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
    }

    private Result run(
        final IndexSearcher searcher,
        final String option,
        final Query filter,
        final float[][] queries,
        final List<Set<Integer>> truth,
        final int k,
        final int warmup,
        final int[] candidates
    ) throws IOException {
        for (int q = 0; q < warmup; q++) {
            searcher.search(query(option, queries[q], k, filter), k);
        }
        final long probes0 = ClusterANNStat.PROBES_PLANNED.getTotal();
        final long vectors0 = ClusterANNStat.VECTORS_SCORED.getTotal();
        final long[] latencies = new long[queries.length];
        double recall = 0;
        for (int q = 0; q < queries.length; q++) {
            final long t0 = System.nanoTime();
            final TopDocs hits = searcher.search(query(option, queries[q], k, filter), k);
            latencies[q] = System.nanoTime() - t0;
            int found = 0;
            for (final ScoreDoc hit : hits.scoreDocs) {
                if (truth.get(q).contains(hit.doc)) {
                    found++;
                }
            }
            recall += truth.get(q).isEmpty() ? 1.0 : (double) found / Math.min(k, truth.get(q).size());
        }
        final long probes = ClusterANNStat.PROBES_PLANNED.getTotal() - probes0;
        final long vectors = ClusterANNStat.VECTORS_SCORED.getTotal() - vectors0;

        // Centroids scored and zero-cluster segments, from the plan, outside the timed loop.
        long centroids = 0;
        long zeroPairs = 0;
        long pairs = 0;
        final List<LeafReaderContext> leaves = searcher.getIndexReader().leaves();
        for (int q = 0; q < queries.length; q++) {
            if (option.equals("today")) {
                for (final int c : candidates) {
                    centroids += c;
                }
                continue;
            }
            final ClusterANNWeight weight = (ClusterANNWeight) searcher.createWeight(
                searcher.rewrite(query(option, queries[q], k, filter)),
                ScoreMode.COMPLETE,
                1f
            );
            final ClusterANNWeight.LeafPlans plans = weight.shardPlan(leaves.get(0));
            for (final LeafReaderContext leaf : leaves) {
                final ClusterANNWeight.LeafPlan plan = plans.forLeaf(leaf.ord);
                if (candidates[leaf.ord] == 0) {
                    continue;
                }
                pairs++;
                centroids += candidates[leaf.ord];
                if (plan.strategy() == null) {
                    zeroPairs++;
                } else {
                    centroids += candidates[leaf.ord];
                }
            }
        }
        Arrays.sort(latencies);
        final double n = queries.length;
        return new Result(
            recall / n,
            probes / n,
            vectors / n,
            centroids / n,
            pct(latencies, 50) / 1e3,
            pct(latencies, 90) / 1e3,
            pairs == 0 ? 0 : (double) zeroPairs / pairs
        );
    }

    private static Query query(final String option, final float[] vector, final int k, final Query filter) {
        return switch (option) {
            case "today" -> new TodayQuery(FIELD, vector, k, filter);
            case "H" -> new ClusterANNQuery(FIELD, vector, k, filter, null, 0, ShardPlanner.Budget.CAPPED);
            case "R" -> new ClusterANNQuery(FIELD, vector, k, filter, null, 0, ShardPlanner.Budget.RULE);
            default -> throw new IllegalArgumentException(option);
        };
    }

    /** Centroids one ranking of each leaf scores under this filter: its non-empty clusters, or its eligible ones. */
    private static int[] candidateCentroids(final DirectoryReader reader, final Query filter, final IndexSearcher searcher)
        throws IOException {
        final int[] candidates = new int[reader.leaves().size()];
        final Weight filterWeight = filterWeight(searcher, filter);
        for (final LeafReaderContext leaf : reader.leaves()) {
            final Clusters clusters = clusters(leaf);
            final AcceptDocs acceptDocs = acceptDocs(filterWeight, leaf);
            if (acceptDocs.cost() == 0) {
                continue;
            }
            final FilterDigest digest = FilterDigest.of(clusters, acceptDocs, () -> { throw new IllegalStateException("dense"); });
            if (digest != null) {
                candidates[leaf.ord] = digest.eligibleClusters();
                continue;
            }
            for (int c = 0; c < clusters.numClusters(); c++) {
                if (clusters.clusterSize(c) != 0) {
                    candidates[leaf.ord]++;
                }
            }
        }
        return candidates;
    }

    private static Clusters clusters(final LeafReaderContext leaf) throws IOException {
        final KNN1030ClusterANNVectorsReader vectors = ClusterANNWeight.clusterAnnReader(leaf.reader(), FIELD);
        return vectors.shardPlanSegment(FIELD, null).clusters();
    }

    private static Weight filterWeight(final IndexSearcher searcher, final Query filter) throws IOException {
        return filterWeight(searcher, filter, FIELD);
    }

    /** The filter conjoined with "has a vector in {@code field}", as ClusterANNQuery builds it; null when unfiltered. */
    private static Weight filterWeight(final IndexSearcher searcher, final Query filter, final String field) throws IOException {
        if (filter == null) {
            return null;
        }
        final Query conjoined = new BooleanQuery.Builder().add(filter, BooleanClause.Occur.FILTER)
            .add(new FieldExistsQuery(field), BooleanClause.Occur.FILTER)
            .build();
        return searcher.createWeight(searcher.rewrite(conjoined), ScoreMode.COMPLETE_NO_SCORES, 1f);
    }

    private static AcceptDocs acceptDocs(final Weight filterWeight, final LeafReaderContext context) {
        if (filterWeight == null) {
            return AcceptDocs.fromLiveDocs(context.reader().getLiveDocs(), context.reader().maxDoc());
        }
        return AcceptDocs.fromIteratorSupplier(() -> {
            final Scorer scorer = filterWeight.scorer(context);
            return scorer == null ? DocIdSetIterator.empty() : scorer.iterator();
        }, context.reader().getLiveDocs(), context.reader().maxDoc());
    }

    /**
     * The ClusterANN query as it was before shard planning: every segment searched with the default strategy, so its
     * reader plans the segment itself (knee, per-segment hold of k). Same filter, collectors and shared threshold.
     */
    static final class TodayQuery extends Query {
        private final String field;
        private final float[] vector;
        private final int k;
        private final Query filter;

        TodayQuery(final String field, final float[] vector, final int k, final Query filter) {
            this.field = field;
            this.vector = vector;
            this.k = k;
            this.filter = filter;
        }

        @Override
        public Weight createWeight(final IndexSearcher searcher, final ScoreMode scoreMode, final float boost) throws IOException {
            final TimeLimitingKnnCollectorManager manager = new TimeLimitingKnnCollectorManager(
                new ClusterANNCollectorManager(k, null, searcher),
                searcher.getTimeout()
            );
            final Weight filterWeight = filterWeight(searcher, filter, field);
            return new Weight(this) {
                @Override
                public ScorerSupplier scorerSupplier(final LeafReaderContext context) {
                    return new ScorerSupplier() {
                        @Override
                        public Scorer get(final long leadCost) throws IOException {
                            final AcceptDocs acceptDocs = acceptDocs(filterWeight, context);
                            TopDocs results = TopDocsCollector.EMPTY_TOPDOCS;
                            if (acceptDocs.cost() > 0) {
                                final KnnCollector collector = manager.newCollector(
                                    Integer.MAX_VALUE,
                                    KnnSearchStrategy.Hnsw.DEFAULT,
                                    context
                                );
                                context.reader().searchNearestVectors(field, vector, collector, acceptDocs);
                                results = collector.topDocs();
                            }
                            return results.scoreDocs.length == 0 ? KNNScorer.emptyScorer() : new KNNScorer(results, boost);
                        }

                        @Override
                        public long cost() {
                            return k;
                        }
                    };
                }

                @Override
                public Explanation explain(final LeafReaderContext context, final int doc) {
                    return Explanation.noMatch("not explained");
                }

                @Override
                public boolean isCacheable(final LeafReaderContext ctx) {
                    return false;
                }
            };
        }

        @Override
        public String toString(final String f) {
            return "TodayQuery";
        }

        @Override
        public void visit(final QueryVisitor visitor) {
            visitor.visitLeaf(this);
        }

        @Override
        public boolean equals(final Object o) {
            return this == o;
        }

        @Override
        public int hashCode() {
            return System.identityHashCode(this);
        }
    }

    // ---------------------------------------------------------------- filters, ground truth, plumbing

    private static int bucket(final int row) {
        long z = row + FILTER_SEED + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z = z ^ (z >>> 31);
        return (int) Long.remainderUnsigned(z, BUCKETS);
    }

    private static int threshold(final double selectivity) {
        return (int) Math.round(selectivity * BUCKETS);
    }

    @SuppressWarnings("unchecked")
    private static List<Set<Integer>>[] groundTruth(final Corpus corpus, final double[] filters, final float[][] queries, final int k) {
        final List<Set<Integer>>[] truth = new List[filters.length];
        for (int f = 0; f < filters.length; f++) {
            final List<Set<Integer>> perQuery = new ArrayList<>(queries.length);
            if (filters[f] >= 1.0) {
                for (int q = 0; q < queries.length; q++) {
                    final Set<Integer> set = new HashSet<>();
                    for (final int row : corpus.neighbours(q, k)) {
                        set.add(row);
                    }
                    perQuery.add(set);
                }
                truth[f] = perQuery;
                continue;
            }
            final int thr = threshold(filters[f]);
            final List<PriorityQueue<float[]>> heaps = new ArrayList<>(queries.length);
            for (int q = 0; q < queries.length; q++) {
                heaps.add(new PriorityQueue<>(Comparator.comparingDouble(e -> e[0])));
            }
            final VectorSimilarityFunction sim = corpus.similarity();
            for (int row = 0; row < corpus.size(); row++) {
                if (bucket(row) >= thr) {
                    continue;
                }
                final float[] v = corpus.vector(row);
                for (int q = 0; q < queries.length; q++) {
                    final float score = sim.compare(queries[q], v);
                    final PriorityQueue<float[]> heap = heaps.get(q);
                    if (heap.size() < k) {
                        heap.add(new float[] { score, row });
                    } else if (score > heap.peek()[0]) {
                        heap.poll();
                        heap.add(new float[] { score, row });
                    }
                }
            }
            for (int q = 0; q < queries.length; q++) {
                final Set<Integer> set = new HashSet<>();
                for (final float[] e : heaps.get(q)) {
                    set.add((int) e[1]);
                }
                perQuery.add(set);
            }
            truth[f] = perQuery;
        }
        return truth;
    }

    private static ClusterANN1030TestCodec codec(final Scenario scenario) {
        final Object quantization = scenario.codec().params().get("quantization");
        if (quantization instanceof Map<?, ?> q && q.get("docBits") instanceof Number bits) {
            final Object encoding = q.get("encoding");
            return new ClusterANN1030TestCodec(
                encoding == null ? ClusterANNEncoding.OPTIMIZED_SCALAR_QUANTIZATION : ClusterANNEncoding.valueOf(encoding.toString()),
                bits.intValue()
            );
        }
        return new ClusterANN1030TestCodec();
    }

    static int[] segmentEnds(final int n, final int segments, final boolean uneven) {
        final int[] ends = new int[segments];
        if (!uneven) {
            final int per = (n + segments - 1) / segments;
            for (int i = 0; i < segments; i++) {
                ends[i] = Math.min(n, (i + 1) * per);
            }
            return ends;
        }
        double weightSum = 0;
        for (int i = 0; i < segments; i++) {
            weightSum += Math.pow(0.5, i);
        }
        int end = 0;
        for (int i = 0; i < segments; i++) {
            final int size = Math.max(1, (int) Math.round(n * Math.pow(0.5, i) / weightSum));
            end = i == segments - 1 ? n : Math.min(n - (segments - 1 - i), end + size);
            ends[i] = end;
        }
        return ends;
    }

    private static long pct(final long[] sorted, final int p) {
        final int rank = (int) Math.ceil(p / 100.0 * sorted.length) - 1;
        return sorted[Math.max(0, Math.min(sorted.length - 1, rank))];
    }

    private static String fmt(final double d) {
        return d == Math.rint(d) ? String.valueOf((long) d) : String.valueOf(d);
    }
}
