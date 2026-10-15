/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

/**
 * Inter-cluster pruning: cuts a ranked centroid plan at the knee in its distance curve.
 *
 * <p>Past the knee the centroids are not the query's neighbours — they are the rest of the index — so each costs a
 * posting read while holding nothing competitive. Where it falls is a property of the query, not the field, so it is
 * read off the curve per query. Only a prefix is cut, never a hole, so closest-first order survives.
 */
final class CentroidKneePruner {

    /** Multiple of the mean gap a gap must exceed to be the knee. Well above 1, or ordinary scatter would fire it. */
    private static final float KNEE_FACTOR = 3.0f;

    private CentroidKneePruner() {}

    /**
     * How many of the ranked clusters to keep, counted from the closest. Returns {@code distances.length} when there
     * is no knee to find: a flat curve, a floor already covering everything ranked, or no gap standing out. Needs
     * more than {@link #KNEE_FACTOR} gaps to work with, which {@link PlanParams#MIN_NPROBE} guarantees.
     *
     * @param distances ranking keys, ascending. A true squared distance only under EUCLIDEAN; elsewhere a negated
     *     similarity, still monotone in cluster quality, which is all the cut relies on
     * @param minProbes the floor the cut may not go below
     */
    static int prune(final float[] distances, final int minProbes) {
        final int ranked = distances.length;
        if (ranked <= minProbes) {
            return ranked;
        }

        final float rise = distances[ranked - 1] - distances[0];
        if (rise <= 0) {
            // Equidistant centroids: nothing can stand out against a mean of zero, and dividing by it is not finite.
            return ranked;
        }

        // The n-1 gaps sum to the rise, so rise/(n-1) is the mean gap. Testing against it keeps this scale-free.
        final float threshold = KNEE_FACTOR * rise / (ranked - 1);
        for (int i = 0; i < ranked - 1; i++) {
            if (distances[i + 1] - distances[i] > threshold) {
                // Search the whole curve, then clamp. Searching from the floor would miss a knee below it and keep
                // every ranked cluster — the opposite of what the curve asked for.
                return Math.max(i + 1, minProbes);
            }
        }
        return ranked;
    }
}
