/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.transport.client.Client;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * EXPERIMENT (do not merge): the live scan early-stop switches. Off by default, the shipped behaviour; on, the defaults
 * are P = 3, epsilon max(1, ceil(0.01 * k)) and floor the plan's minProbes, and each takes an override.
 */
class EarlyStopExperimentTests {

    private static final Setting<Boolean> ENABLED = KNNSettings.KNN_CLUSTERANN_EXPERIMENT_EARLY_STOP_ENABLED_SETTING;
    private static final Setting<Integer> PATIENCE = KNNSettings.KNN_CLUSTERANN_EXPERIMENT_EARLY_STOP_PATIENCE_SETTING;
    private static final Setting<Integer> EPSILON = KNNSettings.KNN_CLUSTERANN_EXPERIMENT_EARLY_STOP_EPSILON_SETTING;
    private static final Setting<Integer> FLOOR = KNNSettings.KNN_CLUSTERANN_EXPERIMENT_EARLY_STOP_FLOOR_SETTING;
    private static final Set<Setting<?>> ALL = Set.of(ENABLED, PATIENCE, EPSILON, FLOOR);

    @AfterEach
    void resetSwitches() {
        EarlyStopExperiment.reset();
        KneeExperiment.reset();
        ProbeCurveExperiment.reset();
    }

    // ---------------------------------------------------------------- defaults, as literals

    /** Off is the default and the live value before anything is applied. */
    @Test
    void testDefaults_thenTheStopIsOff() {
        assertFalse(EarlyStopExperiment.DEFAULT_ENABLED);
        assertSame(EarlyStopParams.OFF, EarlyStopExperiment.current());
        assertFalse(ENABLED.get(Settings.EMPTY));
        assertEquals(EarlyStopParams.OFF, KNNSettings.earlyStop(Settings.EMPTY));
    }

    /** The defaults once on: P = 3, epsilon and floor derived (-1). */
    @Test
    void testDefaults_thenPatienceThreeAndEpsilonAndFloorDerived() {
        assertEquals(3, PATIENCE.get(Settings.EMPTY));
        assertEquals(-1, EPSILON.get(Settings.EMPTY));
        assertEquals(-1, FLOOR.get(Settings.EMPTY));
        assertEquals(3, EarlyStopExperiment.DEFAULT_PATIENCE);
        assertEquals(EarlyStopParams.AUTO, EarlyStopExperiment.DEFAULT_EPSILON);
        assertEquals(EarlyStopParams.AUTO, EarlyStopExperiment.DEFAULT_FLOOR);
    }

    /** The keys, as literals, since the sweep driver sets them by name. */
    @Test
    void testKeys_thenAreTheSweepsNames() {
        assertEquals("knn.clusterann.experiment.early_stop_enabled", ENABLED.getKey());
        assertEquals("knn.clusterann.experiment.early_stop_patience", PATIENCE.getKey());
        assertEquals("knn.clusterann.experiment.early_stop_epsilon", EPSILON.getKey());
        assertEquals("knn.clusterann.experiment.early_stop_floor", FLOOR.getKey());
    }

    /** Every key is a dynamic node-scope setting the plugin registers, so a transient update reaches it. */
    @Test
    void testSettings_thenAreDynamicAndRegistered() {
        for (Setting<?> setting : ALL) {
            assertTrue(setting.isDynamic(), setting.getKey());
            assertTrue(setting.hasNodeScope(), setting.getKey());
            assertTrue(KNNSettings.state().getSettings().contains(setting), setting.getKey());
        }
    }

    // ---------------------------------------------------------------- what the switches mean

    /** Enabled alone is the provisional rule: at k = 100 epsilon is 1, and the floor is the plan's minProbes. */
    @Test
    void testEnabled_withDefaults_thenIsTheProvisionalRule() {
        final EarlyStopParams on = KNNSettings.earlyStop(Settings.builder().put(ENABLED.getKey(), true).build());
        assertEquals(EarlyStopParams.PROVISIONAL, on);
        assertTrue(on.enabled());
        assertEquals(1, on.epsilonFor(100), "max(1, ceil(0.01 * 100))");
        assertEquals(2, on.epsilonFor(150), "max(1, ceil(0.01 * 150))");
        assertEquals(1, on.epsilonFor(10), "never below 1");
        // 2601 clusters (the 1M fleet indexes): minProbes = min(max(10, 51), maxProbes) = 51 at the shipped curve.
        assertEquals(PlanParams.of(2601).minProbes(), on.floorFor(2601));
        assertEquals(51, on.floorFor(2601));
    }

    /** Each of patience, epsilon and floor takes an absolute override. */
    @Test
    void testEnabled_withOverrides_thenTakesThem() {
        final EarlyStopParams on = KNNSettings.earlyStop(
            Settings.builder()
                .put(ENABLED.getKey(), true)
                .put(PATIENCE.getKey(), 5)
                .put(EPSILON.getKey(), 4)
                .put(FLOOR.getKey(), 20)
                .build()
        );
        assertEquals(new EarlyStopParams(5, 4, 20), on);
        assertEquals(4, on.epsilonFor(100), "absolute, not 1% of k");
        assertEquals(20, on.floorFor(2601), "absolute, not minProbes");
        assertEquals(0, KNNSettings.earlyStop(Settings.builder().put(ENABLED.getKey(), true).put(FLOOR.getKey(), 0).build()).floor());
    }

    /** Off is OFF whatever the other three say, so a disabled node scans exactly as before. */
    @Test
    void testDisabled_withOverrides_thenIsOff() {
        final EarlyStopParams off = KNNSettings.earlyStop(
            Settings.builder()
                .put(ENABLED.getKey(), false)
                .put(PATIENCE.getKey(), 5)
                .put(EPSILON.getKey(), 4)
                .put(FLOOR.getKey(), 20)
                .build()
        );
        assertSame(EarlyStopParams.OFF, off);
    }

    // ---------------------------------------------------------------- the live wiring

    /** The production wiring: initialize registers the switches, a transient update turns the stop on, clearing it off. */
    @Test
    void testInitialize_whenTheSettingsAreApplied_thenTheLiveStopFollows() {
        final Set<Setting<?>> nodeScoped = new HashSet<>();
        KNNSettings.state().getSettings().stream().filter(Setting::hasNodeScope).forEach(nodeScoped::add);
        final ClusterSettings clusterSettings = new ClusterSettings(Settings.EMPTY, nodeScoped);
        final ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(Settings.EMPTY);
        KNNSettings.state().initialize(mock(Client.class), clusterService);
        assertSame(EarlyStopParams.OFF, EarlyStopExperiment.current());

        clusterSettings.applySettings(Settings.builder().put(ENABLED.getKey(), true).build());
        assertEquals(EarlyStopParams.PROVISIONAL, EarlyStopExperiment.current());

        clusterSettings.applySettings(
            Settings.builder().put(ENABLED.getKey(), true).put(PATIENCE.getKey(), 2).put(EPSILON.getKey(), 3).build()
        );
        assertEquals(new EarlyStopParams(2, 3, EarlyStopParams.AUTO), EarlyStopExperiment.current());

        clusterSettings.applySettings(Settings.EMPTY);
        assertSame(EarlyStopParams.OFF, EarlyStopExperiment.current());
    }

    /** A caller whose ClusterSettings does not carry the keys still registers, and the node keeps the stop off. */
    @Test
    void testRegister_whenTheClusterSettingsDoNotCarryTheKeys_thenStillRegistersAndKeepsItOff() {
        final ClusterSettings narrow = new ClusterSettings(Settings.EMPTY, Set.of(KNNSettings.QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING));
        EarlyStopExperiment.set(EarlyStopParams.PROVISIONAL);
        KNNSettings.registerEarlyStopExperiment(narrow, Settings.EMPTY);
        assertSame(EarlyStopParams.OFF, EarlyStopExperiment.current());
    }

    /** A node started with the keys set, or keys applied before registering, scans with them from the start. */
    @Test
    void testRegister_whenTheKeysAreAlreadySet_thenSeedsFromThem() {
        final Settings on = Settings.builder().put(ENABLED.getKey(), true).put(PATIENCE.getKey(), 1).build();
        KNNSettings.registerEarlyStopExperiment(new ClusterSettings(on, ALL), on);
        assertEquals(new EarlyStopParams(1, EarlyStopParams.AUTO, EarlyStopParams.AUTO), EarlyStopExperiment.current());

        EarlyStopExperiment.reset();
        final ClusterSettings live = new ClusterSettings(Settings.EMPTY, ALL);
        live.applySettings(on);
        KNNSettings.registerEarlyStopExperiment(live, Settings.EMPTY);
        assertEquals(new EarlyStopParams(1, EarlyStopParams.AUTO, EarlyStopParams.AUTO), EarlyStopExperiment.current());
    }

    /** Out-of-range values are refused, and a refused update leaves the live stop as it was. */
    @Test
    void testRegister_whenAValueIsOutOfRange_thenRejectedAndUnchanged() {
        final ClusterSettings clusterSettings = new ClusterSettings(Settings.EMPTY, ALL);
        KNNSettings.registerEarlyStopExperiment(clusterSettings, Settings.EMPTY);
        clusterSettings.applySettings(Settings.builder().put(ENABLED.getKey(), true).build());
        final EarlyStopParams before = EarlyStopExperiment.current();
        for (Settings bad : List.of(
            Settings.builder().put(ENABLED.getKey(), true).put(EPSILON.getKey(), 0).build(),
            Settings.builder().put(ENABLED.getKey(), true).put(EPSILON.getKey(), -2).build(),
            Settings.builder().put(ENABLED.getKey(), true).put(PATIENCE.getKey(), 0).build(),
            Settings.builder().put(ENABLED.getKey(), true).put(FLOOR.getKey(), -2).build(),
            Settings.builder().put(ENABLED.getKey(), "maybe").build(),
            Settings.builder().put(ENABLED.getKey(), true).put(PATIENCE.getKey(), "three").build()
        )) {
            assertThrows(IllegalArgumentException.class, () -> clusterSettings.applySettings(bad), bad.toString());
            assertSame(before, EarlyStopExperiment.current(), bad.toString());
        }
    }

    /** epsilon 0 is refused even with the stop off, so a bad value never waits to surface when it is turned on. */
    @Test
    void testOf_whenEpsilonIsZero_thenThrowsEvenWhenOff() {
        assertThrows(IllegalArgumentException.class, () -> EarlyStopExperiment.of(false, 3, 0, -1));
        assertThrows(IllegalArgumentException.class, () -> EarlyStopExperiment.of(true, 3, 0, -1));
    }

    /** epsilon 0 is refused by the setting itself, so a node started with it fails as a settings error, not in initialize. */
    @Test
    void testEpsilonSetting_whenZero_thenTheSettingRefusesIt() {
        for (String bad : List.of("0", "-2", "10001", "one")) {
            final Settings settings = Settings.builder().put(EPSILON.getKey(), bad).build();
            assertThrows(IllegalArgumentException.class, () -> EPSILON.get(settings), bad);
        }
        assertEquals(-1, EPSILON.get(Settings.builder().put(EPSILON.getKey(), "-1").build()));
        assertEquals(1, EPSILON.get(Settings.builder().put(EPSILON.getKey(), "1").build()));
        assertEquals(10_000, EPSILON.get(Settings.builder().put(EPSILON.getKey(), "10000").build()));
    }

    /** The stop's two stats, and the scored and planned counters it is compared with, are published in the knn stats. */
    @Test
    void testStats_thenThePluginPublishesTheEarlyStopCounters() {
        @SuppressWarnings("unchecked")
        final java.util.Map<String, Object> stats = (java.util.Map<String, Object>) new org.opensearch.knn.plugin.stats.KNNStats()
            .getStats()
            .get("cluster_ann_query_stats")
            .getValue();
        for (String key : List.of(
            "clusterann_scan_early_stop",
            "clusterann_scan_early_stop_scanned_bps",
            "clusterann_vectors_scored",
            "clusterann_probes_planned",
            "clusterann_clusters_scanned"
        )) {
            assertTrue(stats.containsKey(key), key);
            assertTrue(stats.containsKey(key + "_count"), key + "_count");
        }
    }
}
