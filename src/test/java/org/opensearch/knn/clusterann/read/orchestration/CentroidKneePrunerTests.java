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
 * worked out in the comment above it. Every plan keeps the floor, and the knee is looked for only in the tail past it:
 * the gaps from rank {@code floor-1} on. The bar is three times the tail's mean gap, its rise over its
 * {@code n-floor} gaps, so each case states that rise and the gap that does or does not clear the bar it implies. A
 * floor of one makes the tail the whole curve.
 *
 * <p>The curves are chosen to cut the same way at a factor of four, so only the factor case and the arithmetic in the
 * comments depend on the constant.
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
     * Two breaks is not ambiguous: the plan ends at the first, so the second is never reached. Two gaps can only both
     * clear a bar of {@code 3 x rise / (n-1)} once {@code n-1} exceeds six; ten points leave room at four as well. Rise
     * 207 over 9 gaps bars at 69, and the gaps are 1, 100, 1, 1, 1, 100, 1, 1, 1.
     */
    @Test
    void testPrune_whenTheCurveBreaksTwice_thenCutsAtTheFirst() {
        // given
        float[] distances = { 0f, 1f, 101f, 102f, 103f, 104f, 204f, 205f, 206f, 207f };

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
     * the rise itself, and no one gap exceeds the total rise. Asserted because it bounds where this is useful; with a
     * floor the same limit applies to the tail past it, pinned below.
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
     * The floor is kept outright and the knee is looked for past it, so a cut can land anywhere between the floor and
     * the budget. Floor 3: the tail is the gaps from index 2 on, 1, 1, 1, 1, 56, 1, 1, a rise of 62 over 7 gaps that
     * bars at 26.6. The 56 gap keeps seven. The 100 head gaps are inside the floor and never measured, and they do not
     * inflate the bar either: against the whole curve's mean they would set it at 87.3, above every tail gap.
     */
    @Test
    void testPrune_whenTheTailBreaks_thenCutsBetweenTheFloorAndTheBudget() {
        // given
        float[] distances = { 0f, 100f, 200f, 201f, 202f, 203f, 204f, 260f, 261f, 262f };

        // when
        int keep = CentroidKneePruner.prune(distances, 3);

        // then
        assertEquals(7, keep, "the tail's own knee, above the floor and below the ten ranked");
    }

    /** A tail that climbs evenly past a steep head has no gap over three of its own mean gaps: the budget stays. */
    @Test
    void testPrune_whenTheTailIsFlat_thenKeepsTheBudget() {
        // given: tail gaps all 1, rise 7 over 7 gaps, bar 3
        float[] distances = { 0f, 100f, 200f, 201f, 202f, 203f, 204f, 205f, 206f, 207f };

        // when
        int keep = CentroidKneePruner.prune(distances, 3);

        // then
        assertEquals(distances.length, keep);
    }

    /**
     * A break inside the floor is not a knee. The floor already pays for those clusters, so the cut is decided by the
     * tail alone, and here the tail is even: floor 3, tail gaps all 1, bar 3. Clamping the 999 gap up to the floor
     * would throw the whole tail away for a break the floor had already stepped over.
     */
    @Test
    void testPrune_whenTheOnlyBreakIsInsideTheFloor_thenIgnoresIt() {
        // given
        float[] distances = { 0f, 1f, 1000f, 1001f, 1002f, 1003f, 1004f, 1005f };

        // when
        int keep = CentroidKneePruner.prune(distances, 3);

        // then
        assertEquals(distances.length, keep);
    }

    /**
     * The tail starts at the floor's last gap, so a knee right there keeps exactly the floor: the floor is a count the
     * plan may reach, not one it must exceed. Floor 3: tail rise 107 - 2 = 105 over 8 gaps bars at 39.4, and the 98 gap
     * sits between index 2 and 3.
     */
    @Test
    void testPrune_whenTheKneeIsAtTheFloor_thenCutsToTheFloor() {
        // given
        float[] distances = { 0f, 1f, 2f, 100f, 101f, 102f, 103f, 104f, 105f, 106f, 107f };

        // when
        int keep = CentroidKneePruner.prune(distances, 3);

        // then
        assertEquals(3, keep, "the cut lands on the floor, not above it");
    }

    /**
     * A knee above the floor is left where it is. Floor 2: tail rise 10000 - 4 = 9996 over 6 gaps bars at 4998, and
     * only the 9951 gap into the outlier clears it.
     */
    @Test
    void testPrune_whenTheKneeIsAboveTheFloor_thenCutsAtTheKnee() {
        // given
        float[] distances = { 1f, 4f, 9f, 16f, 25f, 36f, 49f, 10000f };

        // when
        int keep = CentroidKneePruner.prune(distances, 2);

        // then
        assertEquals(7, keep);
    }

    /**
     * A tail of three gaps cannot cut, however sharp: its bar is {@code 3 x rise / 3}, the rise itself, and no one gap
     * exceeds the whole rise. The floor raises where that regime starts, so it is worth pinning there too.
     */
    @Test
    void testPrune_whenTheTailHasTooFewGaps_thenCannotCut() {
        // given: floor 3, tail gaps 1, 1, 99996
        float[] distances = { 1f, 2f, 3f, 4f, 5f, 100001f };

        // when
        int keep = CentroidKneePruner.prune(distances, 3);

        // then
        assertEquals(distances.length, keep);
    }

    // ---------------------------------------------------------------- the factor

    /**
     * The factor is what a gap must beat. Ten gaps, nine of 1 and one of 5: rise 14, mean gap 1.4, so the 5 gap is
     * 3.57 mean gaps. That clears three, and the plan stops before it.
     */
    @Test
    void testPrune_whenAGapIsBetweenThreeAndFourMeanGaps_thenCutsThere() {
        // given
        float[] distances = { 0f, 1f, 2f, 3f, 4f, 9f, 10f, 11f, 12f, 13f, 14f };

        // when
        int keep = CentroidKneePruner.prune(distances, 1);

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
