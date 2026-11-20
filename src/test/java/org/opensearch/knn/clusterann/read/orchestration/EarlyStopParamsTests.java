/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** {@link EarlyStopParams} validation and derivation (JUnit 5). Ported from SearchServicesKnnVectorFormats CR-324504354. */
class EarlyStopParamsTests {

    // ---------------------------------------------------------------- validation

    @ParameterizedTest
    @CsvSource({ "-1, -1, -1", "-5, 1, 0" })
    void testConstructor_whenPatienceIsNegative_thenThrows(int patience, int epsilon, int floor) {
        assertThrows(IllegalArgumentException.class, () -> new EarlyStopParams(patience, epsilon, floor));
    }

    @ParameterizedTest
    @CsvSource({ "3, 0, -1", "3, -2, -1", "3, -100, 0" })
    void testConstructor_whenEpsilonIsNeitherPositiveNorAuto_thenThrows(int patience, int epsilon, int floor) {
        assertThrows(IllegalArgumentException.class, () -> new EarlyStopParams(patience, epsilon, floor));
    }

    @ParameterizedTest
    @CsvSource({ "3, -1, -2", "3, 1, -7" })
    void testConstructor_whenFloorIsNeitherNonNegativeNorAuto_thenThrows(int patience, int epsilon, int floor) {
        assertThrows(IllegalArgumentException.class, () -> new EarlyStopParams(patience, epsilon, floor));
    }

    @ParameterizedTest
    @CsvSource({ "1, 1, 0", "3, -1, -1", "7, 5, 20" })
    void testConstructor_whenValid_thenKeepsTheValues(int patience, int epsilon, int floor) {
        EarlyStopParams params = new EarlyStopParams(patience, epsilon, floor);
        assertEquals(patience, params.patience());
        assertEquals(epsilon, params.epsilon());
        assertEquals(floor, params.floor());
    }

    // ---------------------------------------------------------------- the constants

    /** Off is the default and stops nothing; it is patience 0. */
    @Test
    void testOff_thenIsDisabled() {
        assertFalse(EarlyStopParams.OFF.enabled());
        assertEquals(0, EarlyStopParams.OFF.patience());
        assertFalse(new EarlyStopParams(0, 4, 2).enabled(), "patience 0 is off whatever else is set");
        assertEquals(EarlyStopParams.OFF, new EarlyStopParams(0, 4, 2), "every disabled instance is OFF");
    }

    /** The suggested settings: P = 3, epsilon and floor derived per query. */
    @Test
    void testProvisional_thenIsThreeQuietClustersWithDerivedEpsilonAndFloor() {
        assertTrue(EarlyStopParams.PROVISIONAL.enabled());
        assertEquals(3, EarlyStopParams.PROVISIONAL.patience());
        assertEquals(EarlyStopParams.AUTO, EarlyStopParams.PROVISIONAL.epsilon());
        assertEquals(EarlyStopParams.AUTO, EarlyStopParams.PROVISIONAL.floor());
    }

    // ---------------------------------------------------------------- derivation

    /** AUTO epsilon is max(1, ceil(0.01 * k)). */
    @ParameterizedTest
    @CsvSource({ "0, 1", "1, 1", "10, 1", "100, 1", "101, 2", "200, 2", "250, 3", "1000, 10", "2147483647, 21474837" })
    void testEpsilonFor_whenAuto_thenOnePercentOfKRoundedUpAndAtLeastOne(int k, int expected) {
        assertEquals(expected, EarlyStopParams.PROVISIONAL.epsilonFor(k));
    }

    @Test
    void testEpsilonFor_whenSet_thenIgnoresK() {
        assertEquals(4, new EarlyStopParams(3, 4, 0).epsilonFor(1000));
    }

    /** Never more than k, so the cluster that fills the queue (k inserts) is always useful, at a per-segment k too. */
    @Test
    void testEpsilonFor_whenSetAboveK_thenClampedToK() {
        assertEquals(7, new EarlyStopParams(3, 20, 0).epsilonFor(7));
        assertEquals(1, new EarlyStopParams(3, 20, 0).epsilonFor(0));
    }

    /** AUTO floor is the minProbes PlanParams.of gives the field: max(10, sqrt(C)) capped by maxProbes. */
    @ParameterizedTest
    @CsvSource({ "1", "4", "12", "100", "400", "10000" })
    void testFloorFor_whenAuto_thenThePlansMinProbes(int numClusters) {
        assertEquals(PlanParams.of(numClusters).minProbes(), EarlyStopParams.PROVISIONAL.floorFor(numClusters));
    }

    @Test
    void testFloorFor_whenAuto_thenMatchesTheFormula() {
        assertEquals(10, EarlyStopParams.PROVISIONAL.floorFor(100), "max(10, 10)");
        assertEquals(20, EarlyStopParams.PROVISIONAL.floorFor(400), "max(10, 20)");
        assertEquals(9, EarlyStopParams.PROVISIONAL.floorFor(12), "max(10, 3) capped by maxProbes 9");
    }

    @Test
    void testFloorFor_whenSet_thenIgnoresTheField() {
        assertEquals(2, new EarlyStopParams(3, -1, 2).floorFor(10_000));
    }
}
