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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * EXPERIMENT (do not merge): the live knee switch. Its default is knee pruning on, the shipped behaviour, and a
 * transient cluster update turns it off and back on.
 */
class KneeExperimentTests {

    private static final Setting<Boolean> KNEE = KNNSettings.KNN_CLUSTERANN_EXPERIMENT_KNEE_ENABLED_SETTING;

    @AfterEach
    void resetSwitches() {
        KneeExperiment.reset();
        ProbeCurveExperiment.reset();
    }

    /** The default is the shipped behaviour, as a literal: knee on. */
    @Test
    void testDefaults_thenKneeIsOn() {
        assertTrue(KneeExperiment.DEFAULT_ENABLED);
        assertTrue(KneeExperiment.enabled());
        assertTrue(KNEE.get(Settings.EMPTY));
        assertEquals("knn.clusterann.experiment.knee_enabled", KNEE.getKey());
    }

    /** The key is a dynamic node-scope setting the plugin registers, so a transient update reaches it. */
    @Test
    void testSetting_thenIsDynamicAndRegistered() {
        assertTrue(KNEE.isDynamic());
        assertTrue(KNEE.hasNodeScope());
        assertTrue(KNNSettings.state().getSettings().contains(KNEE));
    }

    /** The production wiring: initialize registers the switch, a transient update turns it off, clearing it turns it on. */
    @Test
    void testInitialize_whenTheSettingIsApplied_thenTheSwitchFollows() {
        final Set<Setting<?>> nodeScoped = new HashSet<>();
        KNNSettings.state().getSettings().stream().filter(Setting::hasNodeScope).forEach(nodeScoped::add);
        final ClusterSettings clusterSettings = new ClusterSettings(Settings.EMPTY, nodeScoped);
        final ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(Settings.EMPTY);
        KNNSettings.state().initialize(mock(Client.class), clusterService);
        assertTrue(KneeExperiment.enabled());

        clusterSettings.applySettings(Settings.builder().put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_KNEE_ENABLED, false).build());
        assertFalse(KneeExperiment.enabled());

        clusterSettings.applySettings(Settings.EMPTY);
        assertTrue(KneeExperiment.enabled());
    }

    /** A caller whose ClusterSettings does not carry the key still registers, and the node keeps the knee on. */
    @Test
    void testRegister_whenTheClusterSettingsDoNotCarryTheKey_thenStillRegistersAndKeepsTheDefault() {
        final ClusterSettings narrow = new ClusterSettings(Settings.EMPTY, Set.of(KNNSettings.QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING));
        KneeExperiment.setEnabled(false);
        KNNSettings.registerKneeExperiment(narrow, Settings.EMPTY);
        assertTrue(KneeExperiment.enabled());
    }

    /** A node started with the key set, or a key applied before registering, is planned with from the start. */
    @Test
    void testRegister_whenTheKeyIsAlreadySet_thenSeedsFromIt() {
        final Settings off = Settings.builder().put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_KNEE_ENABLED, false).build();
        KNNSettings.registerKneeExperiment(new ClusterSettings(off, Set.of(KNEE)), off);
        assertFalse(KneeExperiment.enabled());

        KneeExperiment.reset();
        final ClusterSettings live = new ClusterSettings(Settings.EMPTY, Set.of(KNEE));
        live.applySettings(off);
        KNNSettings.registerKneeExperiment(live, Settings.EMPTY);
        assertFalse(KneeExperiment.enabled());
    }

    /** Only true or false is accepted; a rejected update leaves the switch as it was. */
    @Test
    void testRegister_whenTheValueIsNotABoolean_thenRejectedAndUnchanged() {
        final ClusterSettings clusterSettings = new ClusterSettings(Settings.EMPTY, Set.of(KNEE));
        KNNSettings.registerKneeExperiment(clusterSettings, Settings.EMPTY);
        final Settings bad = Settings.builder().put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_KNEE_ENABLED, "maybe").build();
        assertThrows(IllegalArgumentException.class, () -> clusterSettings.applySettings(bad));
        assertTrue(KneeExperiment.enabled());
    }
}
