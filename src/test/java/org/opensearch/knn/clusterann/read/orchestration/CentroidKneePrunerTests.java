/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Unit tests for {@link CentroidKneePruner}.
 *
 * <p>The pruner is arithmetic over a curve, so the curves here are written out literally and every expected cut is
 * worked out in the comment above it. The bar is three times the mean gap, and the mean gap is the total rise over
 * {@code n-1}, so each case states its rise and the gap that does or does not clear the bar that implies.
 */
class CentroidKneePrunerTests {

    // ---------------------------------------------------------------- cutting

    /**
     * Rise 9999 over 5 gaps bars at 5999.4. The gaps are 3, 5, 7, 9 and 9975: the first four are the query's own
     * neighbourhood and the last is the rest of the index, which is where the plan ends.
     */
    @Test
    void testPrune_whenTheCurveBreaks_thenCutsAtTheBreak() {
        // given
        float[] distances = { 1f, 4f, 9f, 16f, 25f, 10000f };

        // when
        int keep = CentroidKneePruner.prune(distances, 1);

        // then
        assertEquals(5, keep, "everything up to the cliff, and nothing past it");
    }

    /**
     * The cut keeps the clusters before the gap, not after it: a break between index 1 and 2 keeps two. Rise 300 over
     * 5 gaps bars at 180, and the 294 gap is the only one to clear it.
     */
    @Test
    void testPrune_thenKeepsTheClustersBeforeTheGap() {
        // given
        float[] distances = { 1f, 2f, 296f, 297f, 298f, 301f };

        // when
        int keep = CentroidKneePruner.prune(distances, 1);

        // then
        assertEquals(2, keep);
    }

    /**
     * Two breaks is not ambiguous — the plan ends at the first, so the second is never reached. Needs eight points to
     * set up: two gaps can only both clear a bar of {@code 3 × rise / (n-1)} once {@code n-1} exceeds six. Rise 205
     * over 7 gaps bars at 87.9, and the gaps are 1, 100, 1, 1, 100, 1, 1.
     */
    @Test
    void testPrune_whenTheCurveBreaksTwice_thenCutsAtTheFirst() {
        // given
        float[] distances = { 0f, 1f, 101f, 102f, 103f, 203f, 204f, 205f };

        // when
        int keep = CentroidKneePruner.prune(distances, 1);

        // then
        assertEquals(2, keep, "the second break is past the end of the plan");
    }

    /**
     * The key is a negated similarity under MAXIMUM_INNER_PRODUCT and COSINE, so a curve can be entirely negative.
     * The cut reads gaps, which are differences, so it does not care where zero is.
     */
    @Test
    void testPrune_whenTheCurveIsNegative_thenCutsTheSameWay() {
        // given — the shape of the breaking curve above, shifted below zero
        float[] distances = { -10000f, -9997f, -9992f, -9985f, -9976f, -1f };

        // when
        int keep = CentroidKneePruner.prune(distances, 1);

        // then
        assertEquals(5, keep);
    }

    // ---------------------------------------------------------------- not cutting

    /**
     * A smooth curve has no knee and keeps everything. These gaps are 2, 4, 6, 8 and 10 — rising the whole way, and
     * the largest still under the 18 the rise of 30 bars at, because growing steadily is not the same as breaking.
     */
    @Test
    void testPrune_whenTheCurveIsSmooth_thenKeepsEverything() {
        // given
        float[] distances = { 1f, 3f, 7f, 13f, 21f, 31f };

        // when
        int keep = CentroidKneePruner.prune(distances, 1);

        // then
        assertEquals(distances.length, keep);
    }

    /** Equidistant centroids give a mean gap of zero to compare against, so nothing stands out and nothing is cut. */
    @Test
    void testPrune_whenTheCurveIsFlat_thenKeepsEverything() {
        // given
        float[] distances = { 5f, 5f, 5f, 5f, 5f, 5f };

        // when
        int keep = CentroidKneePruner.prune(distances, 1);

        // then
        assertEquals(distances.length, keep);
    }

    /**
     * A gap has to beat three times the mean, so the cut cannot fire on four points: the bar is {@code 3 × rise / 3},
     * the rise itself, and no one gap exceeds the total rise. Asserted because it bounds where this is useful — the
     * real floor is {@link PlanParams#MIN_NPROBE}, well above four, so a production plan is never in this regime.
     */
    @Test
    void testPrune_whenFewerThanFiveRanked_thenCannotCut() {
        // given — as sharp a cliff as a four-point curve can carry
        float[] distances = { 1f, 2f, 3f, 100000f };

        // when
        int keep = CentroidKneePruner.prune(distances, 1);

        // then
        assertEquals(distances.length, keep);
    }

    // ---------------------------------------------------------------- the floor

    /**
     * A knee below the floor is found and then clamped up to it, rather than abandoning the cut: this curve breaks
     * after two, and a floor of four probes four — not the whole plan of six. Searching from the floor instead would
     * never see this break and would keep all six, which is the opposite of what the curve asked for.
     */
    @Test
    void testPrune_whenTheKneeIsBelowTheFloor_thenCutsToTheFloor() {
        // given — the 294 gap clears the bar of 180, as in the keeps-before-the-gap case
        float[] distances = { 1f, 2f, 296f, 297f, 298f, 301f };

        // when
        int keep = CentroidKneePruner.prune(distances, 4);

        // then
        assertEquals(4, keep);
    }

    /**
     * A knee exactly at the floor is honoured rather than pushed one past it — the floor is a count the plan may
     * reach, not one it must exceed. Rise 301 over 5 gaps bars at 180.6, and the 295 gap sits between index 3 and 4.
     */
    @Test
    void testPrune_whenTheKneeIsAtTheFloor_thenCutsToTheFloor() {
        // given
        float[] distances = { 1f, 2f, 3f, 4f, 299f, 302f };

        // when
        int keep = CentroidKneePruner.prune(distances, 4);

        // then
        assertEquals(4, keep, "the cut lands on the floor, not above it");
    }

    /** A knee above the floor is left where it is; the floor only ever raises a cut. */
    @Test
    void testPrune_whenTheKneeIsAboveTheFloor_thenCutsAtTheKnee() {
        // given — the breaking curve again, which cuts at five
        float[] distances = { 1f, 4f, 9f, 16f, 25f, 10000f };

        // when
        int keep = CentroidKneePruner.prune(distances, 2);

        // then
        assertEquals(5, keep);
    }

    /** Nothing ranked beyond the floor leaves no room to cut in, and the curve is not even looked at. */
    @Test
    void testPrune_whenTheFloorCoversEverythingRanked_thenKeepsEverything() {
        // given
        float[] distances = { 1f, 2f, 3f, 100000f };

        // when
        int keep = CentroidKneePruner.prune(distances, 4);

        // then
        assertEquals(distances.length, keep);
    }

    /** A field with one occupied cluster is below any floor, so it is returned whole. */
    @Test
    void testPrune_whenOneRanked_thenKeepsIt() {
        // when
        int keep = CentroidKneePruner.prune(new float[] { 7f }, 1);

        // then
        assertEquals(1, keep);
    }

    /** A field with no occupied cluster has no curve and no plan. */
    @Test
    void testPrune_whenNoneRanked_thenKeepsNothing() {
        // when
        int keep = CentroidKneePruner.prune(new float[0], 1);

        // then
        assertEquals(0, keep);
    }
}
