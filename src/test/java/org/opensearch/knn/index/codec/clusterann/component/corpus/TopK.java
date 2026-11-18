/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.component.corpus;

import java.util.Arrays;

/**
 * The best {@code depth} rows seen so far, ordered by descending score. Higher is better, matching
 * {@link org.apache.lucene.index.VectorSimilarityFunction#compare}, which already maps distances to similarities.
 *
 * <p>Insertion is a single shift into a sorted array: the arrays are tiny (a neighbour depth, 100 at most in
 * practice) and the hot loop is the similarity computation, not the bookkeeping. Ties keep the earlier row, so
 * a corpus scanned in row order yields the lower row number first, as the ann-benchmarks ground truth does.
 *
 * <p>Shared by {@link Hdf5Corpus} (ground truth for a prefix of a file) and the sampler (ground truth over a sample),
 * so both agree on what the exact answer is.
 */
final class TopK {
    private final int[] rows;
    private final float[] scores;
    private int held;

    TopK(final int depth) {
        if (depth < 0) {
            throw new IllegalArgumentException("depth must not be negative, was " + depth);
        }
        this.rows = new int[depth];
        this.scores = new float[depth];
    }

    /** Offers {@code row} with {@code score}; it is kept if it beats the current worst or the set is not full. */
    void offer(final int row, final float score) {
        held = offer(rows, scores, held, row, score);
    }

    /** How many rows are held: {@code depth} once that many have been offered, fewer before. */
    int held() {
        return held;
    }

    /** Held rows, best first. */
    int[] rows() {
        return Arrays.copyOf(rows, held);
    }

    /** Scores of {@link #rows()}, in the same order. */
    float[] scores() {
        return Arrays.copyOf(scores, held);
    }

    /**
     * Inserts {@code row} into {@code rows}/{@code scores}, which hold {@code size} entries sorted by descending
     * score, and returns the new size. A row scoring no better than the worst of a full set is dropped.
     */
    static int offer(final int[] rows, final float[] scores, final int size, final int row, final float score) {
        final int capacity = rows.length;
        if (capacity == 0 || (size == capacity && score <= scores[capacity - 1])) {
            return size;
        }
        final int next = size == capacity ? size : size + 1;
        int at = size == capacity ? capacity - 1 : size;
        while (at > 0 && scores[at - 1] < score) {
            scores[at] = scores[at - 1];
            rows[at] = rows[at - 1];
            at--;
        }
        scores[at] = score;
        rows[at] = row;
        return next;
    }
}
