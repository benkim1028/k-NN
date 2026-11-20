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
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * EXPERIMENT (do not merge): the live shard-plan switches. Defaults are the shard-plan branch's behaviour (on, rule H,
 * floor 8); transient cluster updates change them as one config, and a bad value leaves them unchanged.
 */
class ShardPlanExperimentTests {

    private static final Setting<Boolean> ENABLED = KNNSettings.KNN_CLUSTERANN_EXPERIMENT_SHARD_PLAN_ENABLED_SETTING;
    private static final Setting<ShardPlanner.Budget> BUDGET = KNNSettings.KNN_CLUSTERANN_EXPERIMENT_SHARD_PLAN_BUDGET_SETTING;
    private static final Setting<Double> FLOOR = KNNSettings.KNN_CLUSTERANN_EXPERIMENT_SHARD_PLAN_MIN_MULTIPLIER_SETTING;

    @AfterEach
    void resetSwitches() {
        ShardPlanExperiment.reset();
        ProbeCurveExperiment.reset();
        KneeExperiment.reset();
    }

    /** The defaults, as literals, for the holder and for each setting read from empty settings. */
    @Test
    void testDefaults_thenShardPlanOnCappedFloorEight() {
        assertEquals(new ShardPlanExperiment.Config(true, ShardPlanner.Budget.CAPPED, 8.0), ShardPlanExperiment.current());
        assertEquals(ShardPlanExperiment.DEFAULT, KNNSettings.shardPlan(Settings.EMPTY));
        assertEquals("knn.clusterann.experiment.shard_plan_enabled", ENABLED.getKey());
        assertEquals("knn.clusterann.experiment.shard_plan_budget", BUDGET.getKey());
        assertEquals("knn.clusterann.experiment.shard_plan_min_multiplier", FLOOR.getKey());
    }

    /** Each key is a dynamic node-scope setting the plugin registers, so a transient update reaches it. */
    @Test
    void testSettings_thenAreDynamicAndRegistered() {
        for (final Setting<?> setting : new Setting<?>[] { ENABLED, BUDGET, FLOOR }) {
            assertTrue(setting.isDynamic(), setting.getKey());
            assertTrue(setting.hasNodeScope(), setting.getKey());
            assertTrue(KNNSettings.state().getSettings().contains(setting), setting.getKey());
        }
    }

    /** The production wiring: initialize registers the switches, transient updates move them, clearing restores them. */
    @Test
    void testInitialize_whenTheSettingsAreApplied_thenTheSwitchesFollow() {
        final Set<Setting<?>> nodeScoped = new HashSet<>();
        KNNSettings.state().getSettings().stream().filter(Setting::hasNodeScope).forEach(nodeScoped::add);
        final ClusterSettings clusterSettings = new ClusterSettings(Settings.EMPTY, nodeScoped);
        final ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(Settings.EMPTY);
        KNNSettings.state().initialize(mock(Client.class), clusterService);
        assertEquals(ShardPlanExperiment.DEFAULT, ShardPlanExperiment.current());

        clusterSettings.applySettings(
            Settings.builder()
                .put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_SHARD_PLAN_BUDGET, "rule")
                .put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_SHARD_PLAN_MIN_MULTIPLIER, 0.0)
                .build()
        );
        assertEquals(new ShardPlanExperiment.Config(true, ShardPlanner.Budget.RULE, 0.0), ShardPlanExperiment.current());

        clusterSettings.applySettings(Settings.builder().put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_SHARD_PLAN_ENABLED, false).build());
        assertEquals(new ShardPlanExperiment.Config(false, ShardPlanner.Budget.CAPPED, 8.0), ShardPlanExperiment.current());

        clusterSettings.applySettings(Settings.EMPTY);
        assertEquals(ShardPlanExperiment.DEFAULT, ShardPlanExperiment.current());
    }

    /** A budget that is neither capped nor rule, or a floor out of range, rejects the whole update. */
    @Test
    void testRegister_whenAValueIsBad_thenRejectedAndUnchanged() {
        final ClusterSettings clusterSettings = new ClusterSettings(Settings.EMPTY, Set.of(ENABLED, BUDGET, FLOOR));
        KNNSettings.registerShardPlanExperiment(clusterSettings, Settings.EMPTY);
        for (final Settings bad : new Settings[] {
            Settings.builder().put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_SHARD_PLAN_BUDGET, "uncapped").build(),
            Settings.builder()
                .put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_SHARD_PLAN_ENABLED, false)
                .put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_SHARD_PLAN_MIN_MULTIPLIER, -1.0)
                .build(),
            Settings.builder().put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_SHARD_PLAN_MIN_MULTIPLIER, "NaN").build(),
            Settings.builder().put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_SHARD_PLAN_ENABLED, "maybe").build() }) {
            assertThrows(IllegalArgumentException.class, () -> clusterSettings.applySettings(bad), bad.toString());
            assertEquals(ShardPlanExperiment.DEFAULT, ShardPlanExperiment.current(), bad.toString());
        }
    }

    /** A caller whose ClusterSettings does not carry the keys still registers, and keeps the defaults. */
    @Test
    void testRegister_whenTheClusterSettingsDoNotCarryTheKeys_thenStillRegisters() {
        ShardPlanExperiment.set(new ShardPlanExperiment.Config(false, ShardPlanner.Budget.RULE, 1.0));
        KNNSettings.registerShardPlanExperiment(
            new ClusterSettings(Settings.EMPTY, Set.of(KNNSettings.QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING)),
            Settings.EMPTY
        );
        assertEquals(ShardPlanExperiment.DEFAULT, ShardPlanExperiment.current());
    }

    /**
     * The floor moves the shard ceiling: at the default 8 it is R; at 0, a static probe curve (coefficient 0, intercept
     * m) sets it alone, ceil(m * floor(sqrt C_total)), the baseline's static-m ceiling at the shard's cluster count.
     */
    @Test
    void testFloor_thenSetsTheShardCeiling() {
        assertEquals(80, ShardPlanner.shardParams(100, 1.0).maxProbes());
        ProbeCurveExperiment.set(ProbeCurveExperiment.DEFAULT_CURVE.withIntercept(2.0).withCoefficient(0.0));
        assertEquals(80, ShardPlanner.shardParams(100, 1.0).maxProbes());
        assertEquals(80, ShardPlanner.shardParams(100, 0.001).maxProbes());

        ShardPlanExperiment.set(new ShardPlanExperiment.Config(true, ShardPlanner.Budget.RULE, 0.0));
        for (final int m : new int[] { 1, 2, 4, 8, 16, 32 }) {
            ProbeCurveExperiment.set(ProbeCurveExperiment.DEFAULT_CURVE.withIntercept(m).withCoefficient(0.0));
            assertEquals(Math.min(2625, m * 51), ShardPlanner.shardParams(2625, 1.0).maxProbes(), "m=" + m);
            assertEquals(Math.min(2625, m * 51), ShardPlanner.shardParams(2625, 0.001).maxProbes(), "m=" + m);
            assertEquals(PlanParams.of(2625).maxProbes(), ShardPlanner.shardParams(2625, 1.0).maxProbes(), "m=" + m);
        }
    }

    /** Only capped or rule, in any case, name a budget. */
    @Test
    void testParseBudget() {
        assertEquals(ShardPlanner.Budget.RULE, ShardPlanExperiment.parseBudget(" RULE ", "k"));
        assertEquals(ShardPlanner.Budget.CAPPED, ShardPlanExperiment.parseBudget("capped", "k"));
        assertThrows(IllegalArgumentException.class, () -> ShardPlanExperiment.parseBudget("h", "k"));
        assertThrows(IllegalArgumentException.class, () -> ShardPlanExperiment.parseBudget(null, "k"));
    }
}
