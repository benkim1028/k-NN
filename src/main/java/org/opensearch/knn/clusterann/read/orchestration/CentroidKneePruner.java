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
     * How many of the ranked clusters to keep, counted from the closest. Every plan keeps the floor, and the knee is
     * looked for only in the tail past it: the gaps from rank {@code minProbes - 1} on, each tested against
     * {@link #KNEE_FACTOR} times the tail's own mean gap. The first gap over that keeps everything before it, so a cut
     * lands anywhere from the floor to the budget. Returns {@code distances.length} when there is no knee to find: a
     * flat tail, a floor already covering everything ranked, or no gap standing out.
     *
     * <p>The head is left out of both the search and the mean. A break inside the floor costs nothing to step over,
     * since the floor pays for those clusters anyway, and the head is steep, so the whole curve's mean would sit above
     * every tail gap and nothing past the floor could ever cut. A tail of {@link #KNEE_FACTOR} gaps or fewer cannot cut
     * at all: no one gap exceeds the whole rise.
     *
     * @param distances ranking keys, ascending. A true squared distance only under EUCLIDEAN; elsewhere a negated
     *     similarity, still monotone in cluster quality, which is all the cut relies on
     * @param minProbes the floor the cut may not go below; at least 1, as {@link PlanParams} guarantees
     */
    static int prune(final float[] distances, final int minProbes) {
        final int ranked = distances.length;
        if (ranked <= minProbes) {
            return ranked;
        }

        final int first = minProbes - 1;
        final float rise = distances[ranked - 1] - distances[first];
        if (rise <= 0) {
            // An equidistant tail: nothing can stand out against a mean of zero, and dividing by it is not finite.
            return ranked;
        }

        // The ranked - minProbes tail gaps sum to the rise, so this is the tail's mean gap. Testing against it keeps
        // this scale-free; dividing first keeps a rise near Float.MAX_VALUE finite.
        final float threshold = rise / (ranked - minProbes) * KNEE_FACTOR;
        for (int i = first; i < ranked - 1; i++) {
            if (distances[i + 1] - distances[i] > threshold) {
                return i + 1;
            }
        }
        return ranked;
    }
}
