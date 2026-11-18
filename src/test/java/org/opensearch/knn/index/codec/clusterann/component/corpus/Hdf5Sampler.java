/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.component.corpus;

import io.jhdf.HdfFile;
import io.jhdf.WritableHdfFile;
import io.jhdf.api.Dataset;
import io.jhdf.api.DatasetCreationOptions;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Cuts a small corpus out of a big one. Draws {@code size} rows of {@code train} uniformly without replacement
 * (seeded, so the same rows every time), keeps the first {@code queries} rows of {@code test}, computes exact
 * ground truth over the sample, and writes a new file in the ann-benchmarks layout that {@link Hdf5Corpus} opens
 * like any other: {@code train}, {@code test}, {@code neighbors}, {@code distances}, plus {@code source_rows} (the
 * source row of every sampled vector) and root attributes saying where the sample came from.
 *
 * <p>The source is read in {@link Hdf5Corpus#BLOCK_ROWS}-row slices and only for the blocks that hold a chosen row,
 * so memory is the sample, the queries and the top-k heaps, whatever the source size. Sampling 10k rows out of the
 * 1M-row SIFT file touches nearly every block once and holds about 5 MB.
 *
 * <p>{@code distances} holds {@link VectorSimilarityFunction#compare} scores (higher is better), not raw distances,
 * because that is what the component test scores hits against; the {@code similarity} attribute says which.
 *
 * <p>Under {@code COSINE} the sampled vectors and queries are written at unit length. Cosine is scale invariant, so
 * the ground truth is what a normalising cosine over the raw vectors gives; writing them normalised means a
 * quantized write path that assumes unit vectors for cosine (Lucene's {@code OptimizedScalarQuantizer} asserts it)
 * can index the file, and cosine equals dot product on it. The {@code normalized} attribute records this.
 */
public final class Hdf5Sampler {
    private static final Logger log = LogManager.getLogger(Hdf5Sampler.class);

    /** System property / argument prefix. */
    static final String PREFIX = "knn.sample.";

    private static final String SOURCE_ROWS = "source_rows";
    private static final String DISTANCES = "distances";

    private static final int DEFAULT_QUERIES = 1000;
    private static final int DEFAULT_DEPTH = 100;
    private static final long DEFAULT_SEED = 42;

    /** The four files the suites name; {@code knn.sample.standard=true} produces the ones that are missing. */
    static final List<Options> STANDARD = List.of(
        standard("http://ann-benchmarks.com/sift-128-euclidean.hdf5", 1_000, VectorSimilarityFunction.EUCLIDEAN),
        standard("http://ann-benchmarks.com/sift-128-euclidean.hdf5", 10_000, VectorSimilarityFunction.EUCLIDEAN),
        standard("http://ann-benchmarks.com/glove-100-angular.hdf5", 1_000, VectorSimilarityFunction.COSINE),
        standard("http://ann-benchmarks.com/glove-100-angular.hdf5", 10_000, VectorSimilarityFunction.COSINE)
    );

    private Hdf5Sampler() {}

    /**
     * What to sample. {@code source} is anything {@link Hdf5Corpus} opens (a path, or a URL fetched into the corpus
     * directory); {@code out} is the file to write.
     */
    public record Options(String source, int size, int queries, int depth, VectorSimilarityFunction similarity, long seed, Path out) {
        public Options {
            if (source == null || source.isBlank()) {
                throw new IllegalArgumentException(PREFIX + "source is required: a path or URL of an HDF5 corpus");
            }
            if (size <= 0) {
                throw new IllegalArgumentException(PREFIX + "size must be positive, was " + size);
            }
            if (queries <= 0) {
                throw new IllegalArgumentException(PREFIX + "queries must be positive, was " + queries);
            }
            if (depth <= 0) {
                throw new IllegalArgumentException(PREFIX + "depth must be positive, was " + depth);
            }
            if (depth > size) {
                throw new IllegalArgumentException(
                    PREFIX
                        + "depth "
                        + depth
                        + " exceeds the sample size "
                        + size
                        + "; a query cannot have more"
                        + " neighbours than there are vectors"
                );
            }
            if (similarity == null) {
                throw new IllegalArgumentException(PREFIX + "similarity is required");
            }
            if (out == null) {
                throw new IllegalArgumentException("out is required");
            }
        }

        /**
         * Reads {@code knn.sample.source|size|queries|depth|similarity|seed|out} from {@code properties} (prefix
         * optional). Defaults: 1000 queries, depth 100, seed 42, similarity from the source file name
         * ({@code euclidean}, {@code angular}/{@code cosine}, {@code dot}), output
         * {@code <corpus dir>/<source stem>-sample<N>-q<Q>-seed<seed>.hdf5}.
         */
        public static Options parse(final Map<String, String> properties) {
            final Map<String, String> given = new HashMap<>();
            for (Map.Entry<String, String> entry : properties.entrySet()) {
                final String key = entry.getKey();
                given.put(key.startsWith(PREFIX) ? key.substring(PREFIX.length()) : key, entry.getValue());
            }
            final String source = given.get("source");
            if (source == null || source.isBlank()) {
                throw new IllegalArgumentException(PREFIX + "source is required: a path or URL of an HDF5 corpus");
            }
            final int size = integer(given, "size", -1);
            if (size <= 0) {
                throw new IllegalArgumentException(PREFIX + "size is required: how many train rows to sample");
            }
            final int queries = integer(given, "queries", DEFAULT_QUERIES);
            final int depth = integer(given, "depth", DEFAULT_DEPTH);
            final long seed = given.containsKey("seed") ? Long.parseLong(given.get("seed").trim()) : DEFAULT_SEED;

            final VectorSimilarityFunction similarity;
            if (given.containsKey("similarity")) {
                similarity = similarity(given.get("similarity"));
            } else {
                similarity = Hdf5Corpus.impliedSimilarity(fileName(source));
                if (similarity == null) {
                    throw new IllegalArgumentException(
                        "Cannot tell the metric from the name "
                            + fileName(source)
                            + "; give "
                            + PREFIX
                            + "similarity (euclidean, cosine, dot_product or maximum_inner_product)"
                    );
                }
            }

            final Path out = given.containsKey("out") ? Paths.get(given.get("out")) : defaultOut(source, size, queries, seed);
            return new Options(source, size, queries, depth, similarity, seed, out);
        }

        /** {@code <corpus dir>/<source stem>-sample<N>-q<Q>-seed<seed>.hdf5}; the stem keeps the metric hint. */
        static Path defaultOut(final String source, final int size, final int queries, final long seed) {
            return Hdf5Corpus.corpusDirectory()
                .resolve(stem(fileName(source)) + "-sample" + size + "-q" + queries + "-seed" + seed + ".hdf5");
        }

        private static int integer(final Map<String, String> given, final String key, final int fallback) {
            final String value = given.get(key);
            if (value == null || value.isBlank()) {
                return fallback;
            }
            try {
                return Integer.parseInt(value.trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(PREFIX + key + " must be an integer, was \"" + value + "\"", e);
            }
        }

        private static VectorSimilarityFunction similarity(final String name) {
            try {
                return VectorSimilarityFunction.valueOf(name.trim().toUpperCase(Locale.ROOT));
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(
                    "Unknown similarity \""
                        + name
                        + "\"; one of "
                        + Arrays.toString(VectorSimilarityFunction.values()).toLowerCase(Locale.ROOT),
                    e
                );
            }
        }
    }

    /**
     * Command line: {@code key=value} pairs in the {@code knn.sample.*} names (prefix optional), on top of any
     * {@code -Dknn.sample.*} system properties. With {@code standard=true} and no {@code source}, produces whichever
     * of the {@link #STANDARD} files are not already in the corpus directory.
     */
    public static void main(final String[] args) {
        final Map<String, String> properties = systemProperties();
        for (String arg : args) {
            final int eq = arg.indexOf('=');
            if (eq <= 0) {
                throw new IllegalArgumentException("Expected key=value, got \"" + arg + "\"");
            }
            final String key = arg.substring(0, eq).trim();
            properties.put(key.startsWith(PREFIX) ? key : PREFIX + key, arg.substring(eq + 1).trim());
        }

        final boolean standard = Boolean.parseBoolean(properties.getOrDefault(PREFIX + "standard", "false"));
        final boolean hasSource = properties.get(PREFIX + "source") != null && !properties.get(PREFIX + "source").isBlank();
        if (standard && !hasSource) {
            sampleStandard();
            return;
        }
        if (!hasSource) {
            throw new IllegalArgumentException(
                "Give "
                    + PREFIX
                    + "source=<path or URL> and "
                    + PREFIX
                    + "size=<rows>, or "
                    + PREFIX
                    + "standard=true"
                    + " for the four standard files"
            );
        }
        sample(Options.parse(properties));
    }

    /** Every {@code knn.sample.*} system property, keyed with the prefix. */
    static Map<String, String> systemProperties() {
        final Map<String, String> properties = new HashMap<>();
        for (String name : System.getProperties().stringPropertyNames()) {
            if (name.startsWith(PREFIX)) {
                properties.put(name, System.getProperty(name));
            }
        }
        return properties;
    }

    /** Produces the {@link #STANDARD} files that do not exist yet; existing ones are left alone. */
    public static List<Path> sampleStandard() {
        final List<Path> written = new ArrayList<>();
        for (Options options : STANDARD) {
            if (Files.exists(options.out())) {
                log.info("Standard sample already present, keeping it: {}", options.out());
                continue;
            }
            written.add(sample(options));
        }
        return written;
    }

    /** Samples per {@code options} and returns the file written. */
    public static Path sample(final Options options) {
        final Path file = Hdf5Corpus.resolve(options.source());
        try (HdfFile hdf = new HdfFile(file)) {
            final Dataset train = Hdf5Corpus.dataset(hdf, Hdf5Corpus.TRAIN);
            final Dataset test = Hdf5Corpus.dataset(hdf, Hdf5Corpus.TEST);
            return sample(train, test, options);
        }
    }

    /**
     * The work, on already opened datasets so a caller can hand in its own. Reads {@code train} only through
     * {@link Dataset#getData(long[], int[])}, never whole.
     */
    static Path sample(final Dataset train, final Dataset test, final Options options) {
        final int[] trainShape = shape(train, Hdf5Corpus.TRAIN);
        final int[] testShape = shape(test, Hdf5Corpus.TEST);
        final int rows = trainShape[0];
        final int dimension = trainShape[1];
        if (testShape[1] != dimension) {
            throw new IllegalArgumentException(
                Hdf5Corpus.TEST + " vectors have " + testShape[1] + " dimensions but " + Hdf5Corpus.TRAIN + " has " + dimension
            );
        }
        if (options.size() > rows) {
            throw new IllegalArgumentException(
                "Asked to sample " + options.size() + " rows but " + fileName(options.source()) + " holds only " + rows
            );
        }
        if (options.queries() > testShape[0]) {
            throw new IllegalArgumentException(
                "Asked for " + options.queries() + " queries but " + fileName(options.source()) + " holds only " + testShape[0]
            );
        }

        final long start = System.nanoTime();
        final int[] selected = select(rows, options.size(), options.seed());
        final int[] blocksRead = new int[1];
        final float[][] sample = readRows(train, selected, rows, dimension, blocksRead);
        final float[][] queries = Hdf5Corpus.rows(
            test.getData(new long[] { 0, 0 }, new int[] { options.queries(), dimension }),
            Hdf5Corpus.TEST
        );
        final boolean normalized = options.similarity() == VectorSimilarityFunction.COSINE;
        if (normalized) {
            normalize(sample);
            normalize(queries);
        }
        log.info(
            "Sampled {} of {} rows from {} ({} blocks of {} read{}) in {} s; computing exact top-{} for {} queries under {}",
            options.size(),
            rows,
            fileName(options.source()),
            blocksRead[0],
            Hdf5Corpus.BLOCK_ROWS,
            normalized ? ", vectors and queries normalised to unit length" : "",
            (System.nanoTime() - start) / 1_000_000_000L,
            options.depth(),
            options.queries(),
            options.similarity()
        );

        final int[][] neighbours = new int[queries.length][];
        final float[][] distances = new float[queries.length][];
        groundTruth(sample, queries, options.depth(), options.similarity(), neighbours, distances);

        write(options, sample, queries, neighbours, distances, selected, rows, normalized);
        log.info(
            "Wrote {}: train {} x {}, test {} x {}, neighbors {} x {} ({} MB) in {} s",
            options.out(),
            sample.length,
            dimension,
            queries.length,
            dimension,
            neighbours.length,
            options.depth(),
            megabytes(options.out()),
            (System.nanoTime() - start) / 1_000_000_000L
        );
        return options.out();
    }

    /** {@code size} distinct rows out of {@code [0, rows)}, uniformly, sorted ascending. Partial Fisher-Yates. */
    static int[] select(final int rows, final int size, final long seed) {
        final int[] index = new int[rows];
        for (int i = 0; i < rows; i++) {
            index[i] = i;
        }
        final SplittableRandom random = new SplittableRandom(seed);
        for (int i = 0; i < size; i++) {
            final int j = i + random.nextInt(rows - i);
            final int swap = index[i];
            index[i] = index[j];
            index[j] = swap;
        }
        final int[] selected = Arrays.copyOf(index, size);
        Arrays.sort(selected);
        return selected;
    }

    /** The rows at {@code selected} (ascending), read one block at a time and only for blocks that hold one. */
    private static float[][] readRows(
        final Dataset train,
        final int[] selected,
        final int rows,
        final int dimension,
        final int[] blocksRead
    ) {
        final float[][] sample = new float[selected.length][];
        int next = 0;
        while (next < selected.length) {
            final int blockIndex = selected[next] / Hdf5Corpus.BLOCK_ROWS;
            final int from = blockIndex * Hdf5Corpus.BLOCK_ROWS;
            final int height = Math.min(Hdf5Corpus.BLOCK_ROWS, rows - from);
            final float[][] block = Hdf5Corpus.rows(
                train.getData(new long[] { from, 0 }, new int[] { height, dimension }),
                Hdf5Corpus.TRAIN
            );
            blocksRead[0]++;
            while (next < selected.length && selected[next] < from + height) {
                sample[next] = block[selected[next] - from];
                next++;
            }
        }
        return sample;
    }

    /**
     * Scales every vector to unit length in place (a zero vector is left alone). Cosine is scale invariant, so the
     * ground truth is the same either way; the point is that a quantized write path may assume unit vectors under
     * {@code COSINE}, as Lucene's {@code OptimizedScalarQuantizer} asserts, and GloVe is not unit length.
     */
    private static void normalize(final float[][] vectors) {
        for (float[] vector : vectors) {
            VectorUtil.l2normalize(vector, false);
        }
    }

    private static void groundTruth(
        final float[][] sample,
        final float[][] queries,
        final int depth,
        final VectorSimilarityFunction similarity,
        final int[][] neighbours,
        final float[][] distances
    ) {
        final long start = System.nanoTime();
        for (int query = 0; query < queries.length; query++) {
            final TopK top = new TopK(depth);
            final float[] vector = queries[query];
            for (int row = 0; row < sample.length; row++) {
                top.offer(row, similarity.compare(vector, sample[row]));
            }
            neighbours[query] = top.rows();
            distances[query] = top.scores();
        }
        log.info("Exact ground truth in {} s", (System.nanoTime() - start) / 1_000_000_000L);
    }

    private static void write(
        final Options options,
        final float[][] sample,
        final float[][] queries,
        final int[][] neighbours,
        final float[][] distances,
        final int[] selected,
        final int sourceRows,
        final boolean normalized
    ) {
        final Path out = options.out().toAbsolutePath();
        final Path part = out.resolveSibling(out.getFileName() + ".part");
        try {
            Files.createDirectories(out.getParent());
            Files.deleteIfExists(part);
            try (WritableHdfFile hdf = HdfFile.write(part)) {
                hdf.putDataset(Hdf5Corpus.TRAIN, sample, chunked(sample.length, sample[0].length));
                hdf.putDataset(Hdf5Corpus.TEST, queries, chunked(queries.length, queries[0].length));
                hdf.putDataset(Hdf5Corpus.NEIGHBOURS, neighbours, chunked(neighbours.length, options.depth()));
                hdf.putDataset(DISTANCES, distances, chunked(distances.length, options.depth()));
                hdf.putDataset(SOURCE_ROWS, selected, DatasetCreationOptions.builder().deflate().build());

                final boolean remote = options.source().startsWith("http://") || options.source().startsWith("https://");
                hdf.putAttribute("source_file", fileName(options.source()));
                hdf.putAttribute("source_url", remote ? options.source() : "");
                hdf.putAttribute("source_rows_total", sourceRows);
                hdf.putAttribute("seed", options.seed());
                hdf.putAttribute("similarity", options.similarity().name().toLowerCase(Locale.ROOT));
                hdf.putAttribute("normalized", Boolean.toString(normalized));
                hdf.putAttribute("size", sample.length);
                hdf.putAttribute("queries", queries.length);
                hdf.putAttribute("depth", options.depth());
                hdf.putAttribute("sampler", Hdf5Sampler.class.getName());
            }
            Files.move(part, out, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            throw new UncheckedIOException("Could not write the sample to " + out, e);
        } finally {
            try {
                Files.deleteIfExists(part);
            } catch (IOException ignored) {}
        }
    }

    /** Chunks of {@link Hdf5Corpus#BLOCK_ROWS} full rows, deflated, so a reader's block is one chunk. */
    private static DatasetCreationOptions chunked(final int rows, final int columns) {
        return DatasetCreationOptions.builder().chunkDimensions(Math.min(Hdf5Corpus.BLOCK_ROWS, rows), columns).deflate().build();
    }

    private static int[] shape(final Dataset dataset, final String name) {
        final int[] shape = dataset.getDimensions();
        if (shape.length != 2) {
            throw new IllegalArgumentException(name + " must be a 2D dataset, found " + Arrays.toString(shape));
        }
        return shape;
    }

    private static Options standard(final String url, final int size, final VectorSimilarityFunction similarity) {
        return new Options(
            url,
            size,
            DEFAULT_QUERIES,
            DEFAULT_DEPTH,
            similarity,
            DEFAULT_SEED,
            Options.defaultOut(url, size, DEFAULT_QUERIES, DEFAULT_SEED)
        );
    }

    /** Last path segment of a path or URL. */
    static String fileName(final String source) {
        final String path = source.startsWith("http://") || source.startsWith("https://") ? URI.create(source).getPath() : source;
        final int slash = Math.max(path.lastIndexOf('/'), path.lastIndexOf(java.io.File.separatorChar));
        return path.substring(slash + 1);
    }

    /** {@code documents-1k.hdf5.bz2} and {@code documents-1k.hdf5} both give {@code documents-1k}. */
    static String stem(final String fileName) {
        String stem = fileName;
        if (stem.endsWith(".bz2")) {
            stem = stem.substring(0, stem.length() - ".bz2".length());
        }
        final int dot = stem.lastIndexOf('.');
        return dot > 0 ? stem.substring(0, dot) : stem;
    }

    private static long megabytes(final Path file) {
        try {
            return Files.size(file) / (1024 * 1024);
        } catch (IOException e) {
            return -1;
        }
    }
}
