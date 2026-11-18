/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.component.corpus;

import io.jhdf.HdfFile;
import io.jhdf.api.Dataset;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.VectorSimilarityFunction;

import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URL;
import java.net.URLConnection;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Reads an ann-benchmarks HDF5 corpus - a {@code train}, {@code test} and {@code neighbors} dataset - fetching and
 * decompressing it on a cache miss. Train vectors are read in blocks, so memory does not scale with the dataset.
 */
public final class Hdf5Corpus implements Corpus, AutoCloseable {
    private static final Logger log = LogManager.getLogger(Hdf5Corpus.class);

    static final String TRAIN = "train";
    static final String TEST = "test";
    static final String NEIGHBOURS = "neighbors";

    /** Rows per read of {@code train}; also the chunk height the sampler writes, so a block is one chunk. */
    static final int BLOCK_ROWS = 1024;

    private static final int CACHED_BLOCKS = 4;

    private static final String DEFAULT_CACHE = Paths.get("src", "test", "resources", "dataset").toString();

    private final String name;
    private final HdfFile hdf;
    private final Dataset trainSet;
    private final int size;
    private final int dimension;
    private final float[][] queries;
    private int[][] neighbours;
    private final VectorSimilarityFunction similarity;

    private final Map<Integer, float[][]> blocks = new LinkedHashMap<>();

    private Hdf5Corpus(
        final String name,
        final HdfFile hdf,
        final Dataset trainSet,
        final int size,
        final int dimension,
        final float[][] queries,
        final int[][] neighbours,
        final VectorSimilarityFunction similarity
    ) {
        this.name = name;
        this.hdf = hdf;
        this.trainSet = trainSet;
        this.size = size;
        this.dimension = dimension;
        this.queries = queries;
        this.neighbours = neighbours;
        this.similarity = similarity;
    }

    /** Opens {@code location}: a path already on disk, or a URL fetched into the corpus directory on a miss. */
    public static Hdf5Corpus open(
        final String location,
        final int queryLimit,
        final int sizeLimit,
        final VectorSimilarityFunction similarity
    ) {
        final Path file = resolve(location);

        final HdfFile hdf = new HdfFile(file);
        boolean opened = false;
        try {
            final Dataset trainSet = dataset(hdf, TRAIN);
            final int[] shape = trainSet.getDimensions();
            if (shape.length != 2) {
                throw new IllegalArgumentException(TRAIN + " must be a 2D dataset, found " + Arrays.toString(shape));
            }
            final float[][] allQueries = floats(dataset(hdf, TEST));
            final int[][] allNeighbours = ints(dataset(hdf, NEIGHBOURS));
            if (allQueries.length != allNeighbours.length) {
                throw new IllegalArgumentException(
                    file
                        + " has "
                        + allQueries.length
                        + " queries but "
                        + allNeighbours.length
                        + " neighbour rows; they must line up for the ground truth to apply"
                );
            }

            final int queries = queryLimit > 0 ? Math.min(queryLimit, allQueries.length) : allQueries.length;
            final int size = sizeLimit > 0 ? Math.min(sizeLimit, shape[0]) : shape[0];
            warnIfNameDisagrees(file, similarity);
            log.info(
                "Opened {}: {} x {} train (read in blocks of {}), {} of {} queries, {} neighbours each, metric {}",
                file.getFileName(),
                shape[0],
                shape[1],
                BLOCK_ROWS,
                queries,
                allQueries.length,
                allNeighbours.length == 0 ? 0 : allNeighbours[0].length,
                similarity
            );
            final Hdf5Corpus corpus = new Hdf5Corpus(
                name(file, size, queries),
                hdf,
                trainSet,
                size,
                shape[1],
                Arrays.copyOf(allQueries, queries),
                Arrays.copyOf(allNeighbours, queries),
                similarity
            );
            opened = true;
            if (size < shape[0]) {
                log.info(
                    "Using the first {} of {} vectors, so the file's neighbours do not apply; recomputing ground truth"
                        + " for {} queries at depth {} (one pass over the subset)",
                    size,
                    shape[0],
                    queries,
                    allNeighbours.length == 0 ? 0 : allNeighbours[0].length
                );
                corpus.recomputeNeighbours(allNeighbours.length == 0 ? 100 : allNeighbours[0].length);
            }
            return corpus;
        } finally {
            if (!opened) {
                hdf.close();
            }
        }
    }

    /**
     * A local path as given, or a URL fetched into {@code knn.corpus.dir} on a cache miss ({@code .bz2} decompressed
     * as it arrives). Package-private so the sampler shares the cache rather than keeping its own.
     */
    static Path resolve(final String location) {
        final boolean remote = location.startsWith("http://") || location.startsWith("https://");
        if (!remote) {
            final Path file = Paths.get(location);
            if (Files.notExists(file)) {
                throw new IllegalArgumentException(
                    "No corpus at "
                        + file.toAbsolutePath()
                        + ". Give corpus.file a URL to have it downloaded, or point"
                        + " it at a dataset already on disk; a suite naming a file it does not have must fail here"
                        + " rather than quietly fall back to generated vectors."
                );
            }
            return file;
        }

        final URI uri = URI.create(location);
        final String path = uri.getPath();
        final String remoteName = path.substring(path.lastIndexOf('/') + 1);
        if (remoteName.isEmpty()) {
            throw new IllegalArgumentException("Cannot tell a file name from " + location);
        }

        final boolean compressed = remoteName.endsWith(".bz2");
        final String cachedName = compressed ? remoteName.substring(0, remoteName.length() - ".bz2".length()) : remoteName;
        final Path directory = corpusDirectory();
        final Path target = directory.resolve(cachedName);
        if (Files.exists(target)) {
            log.info("Using cached corpus {} ({} MB)", target, megabytes(target));
            return target;
        }

        final Path part = target.resolveSibling(cachedName + ".part");
        try {
            Files.createDirectories(directory);
            final URLConnection connection = new URL(location).openConnection();
            connection.setConnectTimeout(30_000);
            connection.setReadTimeout(120_000);
            final long expected = connection.getContentLengthLong();
            log.info(
                "Downloading {} to {}{}{}",
                location,
                target,
                expected > 0 ? " (" + expected / (1024 * 1024) + " MB" : "",
                compressed ? ", decompressing as it arrives)" : ")"
            );
            final long start = System.nanoTime();
            try (
                InputStream raw = connection.getInputStream();
                InputStream in = compressed ? new BZip2CompressorInputStream(raw, true) : raw
            ) {
                Files.copy(in, part, StandardCopyOption.REPLACE_EXISTING);
            }

            if (!compressed && expected > 0 && Files.size(part) != expected) {
                throw new IOException("Downloaded " + Files.size(part) + " bytes of an expected " + expected);
            }
            Files.move(part, target, StandardCopyOption.ATOMIC_MOVE);
            log.info("Corpus ready: {} MB in {} s", megabytes(target), (System.nanoTime() - start) / 1_000_000_000L);
            return target;
        } catch (IOException e) {
            throw new IllegalStateException(
                "Could not fetch the corpus from "
                    + location
                    + ". Download it by hand into "
                    + directory
                    + " (decompressed, named "
                    + cachedName
                    + "), or point corpus.file at a local path",
                e
            );
        } finally {
            try {
                Files.deleteIfExists(part);
            } catch (IOException ignored) {}
        }
    }

    /** Where downloaded corpora are cached and where sampled ones land: {@code knn.corpus.dir}, or the test dataset dir. */
    static Path corpusDirectory() {
        return Paths.get(System.getProperty("knn.corpus.dir", DEFAULT_CACHE));
    }

    private static long megabytes(final Path file) {
        try {
            return Files.size(file) / (1024 * 1024);
        } catch (IOException e) {
            return -1;
        }
    }

    /** The size a scenario asked for is not a knob here, so say so rather than appear to have honoured it. */
    public static void warnIfSizeIgnored(final String corpus, final int requested, final int actual) {
        if (requested > 0 && requested != actual) {
            log.warn(
                "Scenario asks for corpus.size {} but {} holds {} vectors; using all of them, because the file's"
                    + " neighbours are row numbers into the whole train set and a subset would invalidate them",
                requested,
                corpus,
                actual
            );
        }
    }

    @Override
    public float[] vector(final int row) {
        if (row < 0 || row >= size) {
            throw new IndexOutOfBoundsException("row " + row + " outside [0, " + size + ")");
        }
        final int blockIndex = row / BLOCK_ROWS;
        float[][] block = blocks.remove(blockIndex);
        if (block == null) {
            block = readBlock(blockIndex);
            if (blocks.size() >= CACHED_BLOCKS) {
                final Integer coldest = blocks.keySet().iterator().next();
                blocks.remove(coldest);
            }
        }

        blocks.put(blockIndex, block);
        return block[row - blockIndex * BLOCK_ROWS];
    }

    private float[][] readBlock(final int blockIndex) {
        final int from = blockIndex * BLOCK_ROWS;
        final int rows = Math.min(BLOCK_ROWS, size - from);
        return rows(trainSet.getData(new long[] { from, 0 }, new int[] { rows, dimension }), TRAIN);
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int queries() {
        return queries.length;
    }

    @Override
    public float[] query(final int index) {
        return queries[index];
    }

    @Override
    public int[] neighbours(final int index, final int count) {
        final int[] row = neighbours[index];
        if (count > row.length) {
            throw new IllegalArgumentException(
                "This corpus holds "
                    + row.length
                    + " neighbours per query, asked for "
                    + count
                    + "; a k above the file's neighbour depth cannot be measured"
            );
        }
        return Arrays.copyOf(row, count);
    }

    @Override
    public VectorSimilarityFunction similarity() {
        return similarity;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public String name() {
        return name;
    }

    private void recomputeNeighbours(final int depth) {
        final TopK[] best = new TopK[queries.length];
        for (int query = 0; query < queries.length; query++) {
            best[query] = new TopK(depth);
        }

        final long start = System.nanoTime();
        for (int from = 0; from < size; from += BLOCK_ROWS) {
            final float[][] block = readBlock(from / BLOCK_ROWS);
            final int rows = Math.min(block.length, size - from);
            for (int row = 0; row < rows; row++) {
                final float[] vector = block[row];
                for (int query = 0; query < queries.length; query++) {
                    best[query].offer(from + row, similarity.compare(queries[query], vector));
                }
            }
        }
        for (int query = 0; query < queries.length; query++) {
            neighbours[query] = best[query].rows();
        }
        log.info("Recomputed ground truth in {} s", (System.nanoTime() - start) / 1_000_000_000L);
    }

    @Override
    public void close() {
        blocks.clear();
        hdf.close();
    }

    private static void warnIfNameDisagrees(final Path file, final VectorSimilarityFunction declared) {
        final VectorSimilarityFunction implied = impliedSimilarity(file.getFileName().toString());
        if (implied == null || implied == declared) {
            return;
        }
        if (implied == VectorSimilarityFunction.DOT_PRODUCT && declared == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
            return;
        }
        log.warn(
            "{} looks like a {} dataset but the scenario declares {}; if that is wrong every query is scored against"
                + " the answers to a different question, and recall will look like the format's fault",
            file.getFileName(),
            implied,
            declared
        );
    }

    /**
     * The metric an ann-benchmarks style file name advertises ({@code sift-128-euclidean}, {@code glove-100-angular},
     * {@code lastfm-64-dot}), or {@code null} when the name says nothing. Angular is cosine.
     */
    static VectorSimilarityFunction impliedSimilarity(final String fileName) {
        final String lower = fileName.toLowerCase(Locale.ROOT);
        if (lower.contains("euclidean") || lower.contains("-l2")) {
            return VectorSimilarityFunction.EUCLIDEAN;
        }
        if (lower.contains("angular") || lower.contains("cosine")) {
            return VectorSimilarityFunction.COSINE;
        }
        if (lower.contains("-ip") || lower.contains("innerproduct") || lower.contains("dot")) {
            return VectorSimilarityFunction.DOT_PRODUCT;
        }
        return null;
    }

    private static String name(final Path file, final int size, final int queries) {
        final String fileName = file.getFileName().toString();
        final int dot = fileName.lastIndexOf('.');
        return (dot > 0 ? fileName.substring(0, dot) : fileName) + "-n" + size + "-q" + queries;
    }

    private static float[][] floats(final Dataset dataset) {
        return rows(dataset.getData(), "Dataset \"" + dataset.getName() + "\"");
    }

    /** {@code data} as read by jhdf, as float rows; {@code what} names the dataset for the error. */
    static float[][] rows(final Object data, final String what) {
        if (data instanceof float[][] rows) {
            return rows;
        }
        if (data instanceof double[][] rows) {
            return narrow(rows);
        }
        throw new IllegalArgumentException(what + " holds " + data.getClass().getSimpleName() + ", expected a 2D float or double array");
    }

    private static float[][] narrow(final double[][] rows) {
        final float[][] narrowed = new float[rows.length][];
        for (int row = 0; row < rows.length; row++) {
            narrowed[row] = new float[rows[row].length];
            for (int i = 0; i < rows[row].length; i++) {
                narrowed[row][i] = (float) rows[row][i];
            }
        }
        return narrowed;
    }

    private static int[][] ints(final Dataset dataset) {
        final Object data = dataset.getData();
        if (data instanceof int[][] rows) {
            return rows;
        }
        if (data instanceof long[][] rows) {
            final int[][] narrowed = new int[rows.length][];
            for (int row = 0; row < rows.length; row++) {
                narrowed[row] = new int[rows[row].length];
                for (int i = 0; i < rows[row].length; i++) {
                    narrowed[row][i] = (int) rows[row][i];
                }
            }
            return narrowed;
        }
        throw new IllegalArgumentException(
            "Dataset \"" + dataset.getName() + "\" holds " + data.getClass().getSimpleName() + ", expected a 2D int or long array"
        );
    }

    static Dataset dataset(final HdfFile hdf, final String path) {
        final Dataset dataset = hdf.getDatasetByPath(path);
        if (dataset == null) {
            throw new IllegalArgumentException(
                "No \""
                    + path
                    + "\" dataset in the file; expected the ann-benchmarks layout of train/test/neighbors,"
                    + " found "
                    + hdf.getChildren().keySet()
            );
        }
        return dataset;
    }
}
