/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.transport.client.Client;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * EXPERIMENT (do not merge): the live probe curve. Its defaults are the shipped constants, each setting overrides
 * exactly its own constant, and {@link PlanParams} follows the live curve.
 */
class ProbeCurveExperimentTests {

    private static final List<Setting<Double>> CURVE_SETTINGS = List.of(
        KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_INTERCEPT_SETTING,
        KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_COEFFICIENT_SETTING,
        KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_EXPONENT_SETTING,
        KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_CAP_SETTING
    );

    @AfterEach
    void resetCurve() {
        ProbeCurveExperiment.reset();
    }

    // ---------------------------------------------------------------- defaults

    /** The defaults are the shipped curve, as literals: m(s) = min(32, 1 + 2 / s^(1/3)). */
    @Test
    void testDefaults_thenEqualTheShippedConstants() {
        assertEquals(1.0, ProbeCurveExperiment.DEFAULT_INTERCEPT);
        assertEquals(2.0, ProbeCurveExperiment.DEFAULT_COEFFICIENT);
        assertEquals(1.0 / 3.0, ProbeCurveExperiment.DEFAULT_EXPONENT);
        assertEquals(32.0, ProbeCurveExperiment.DEFAULT_CAP);
        assertEquals(PlanParams.CAP_MULTIPLIER, ProbeCurveExperiment.DEFAULT_CAP);
        assertEquals(PlanParams.SELECTIVITY_EXPONENT, ProbeCurveExperiment.DEFAULT_EXPONENT);
        assertEquals(ProbeCurveExperiment.DEFAULT_CURVE, ProbeCurveExperiment.current());
    }

    /**
     * Bit for bit, not within a tolerance: the default curve computes exactly the shipped expression at every
     * selectivity, so no ceil(m * root) can land on the other side of an integer.
     */
    @Test
    void testMultiplier_whenTheCurveIsTheDefault_thenBitIdenticalToTheShippedExpression() {
        final java.util.Random random = new java.util.Random(20261118L);
        for (int i = 0; i <= 200_000; i++) {
            final double s = i <= 100_000 ? i / 100_000.0 : random.nextDouble();
            final double shipped = Math.min(32.0, 1.0 + 2.0 / Math.pow(s, 1.0 / 3.0));
            assertEquals(Double.doubleToRawLongBits(shipped), Double.doubleToRawLongBits(PlanParams.multiplier(s)), "s = " + s);
            for (int root : new int[] { 45, 50, 51 }) {
                assertEquals((int) Math.ceil(shipped * root), (int) Math.ceil(PlanParams.multiplier(s) * root), "s = " + s);
            }
        }
    }

    /** A node that never sets the keys reads the shipped constants, bit for bit, including 1/3 through its string. */
    @Test
    void testSettings_whenUnset_thenDefaultToTheShippedConstants() {
        assertEquals(1.0, KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_INTERCEPT_SETTING.get(Settings.EMPTY));
        assertEquals(2.0, KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_COEFFICIENT_SETTING.get(Settings.EMPTY));
        assertEquals(1.0 / 3.0, KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_EXPONENT_SETTING.get(Settings.EMPTY));
        assertEquals(32.0, KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_CAP_SETTING.get(Settings.EMPTY));
    }

    /** Each key is a dynamic node-scope setting the plugin registers, so a transient cluster update reaches it. */
    @Test
    void testSettings_thenAreDynamicAndRegistered() {
        final List<Setting<?>> registered = KNNSettings.state().getSettings();
        for (Setting<Double> setting : CURVE_SETTINGS) {
            assertTrue(setting.isDynamic(), setting.getKey());
            assertTrue(setting.hasNodeScope(), setting.getKey());
            assertTrue(registered.contains(setting), setting.getKey() + " not registered");
        }
    }

    /** With the defaults, the multiplier is unchanged at every point the shipped tests pin. */
    @ParameterizedTest(name = "s = {0}")
    @CsvSource({ "1.0, 3.0", "0.25, 4.174802103936399", "0.01, 10.283177667225557", "0.001, 21.0", "0.0001, 32.0", "0.0, 32.0" })
    void testMultiplier_whenTheCurveIsTheDefault_thenUnchanged(double selectivity, double multiplier) {
        assertEquals(multiplier, ProbeCurveExperiment.DEFAULT_CURVE.multiplier(selectivity), 1e-12);
        assertEquals(multiplier, PlanParams.multiplier(selectivity), 1e-12);
    }

    // ---------------------------------------------------------------- overrides

    /** Each with-er moves its own constant and leaves the other three alone; set publishes the whole curve. */
    @Test
    void testSetters_thenEachMovesOnlyItsOwnConstant() {
        ProbeCurveExperiment.set(ProbeCurveExperiment.current().withIntercept(0.5));
        assertEquals(new ProbeCurveExperiment.Curve(0.5, 2.0, 1.0 / 3.0, 32.0), ProbeCurveExperiment.current());
        ProbeCurveExperiment.set(ProbeCurveExperiment.current().withCoefficient(1.5));
        assertEquals(new ProbeCurveExperiment.Curve(0.5, 1.5, 1.0 / 3.0, 32.0), ProbeCurveExperiment.current());
        ProbeCurveExperiment.set(ProbeCurveExperiment.current().withExponent(0.5));
        assertEquals(new ProbeCurveExperiment.Curve(0.5, 1.5, 0.5, 32.0), ProbeCurveExperiment.current());
        ProbeCurveExperiment.set(ProbeCurveExperiment.current().withCap(8.0));
        assertEquals(new ProbeCurveExperiment.Curve(0.5, 1.5, 0.5, 8.0), ProbeCurveExperiment.current());
        ProbeCurveExperiment.reset();
        assertEquals(ProbeCurveExperiment.DEFAULT_CURVE, ProbeCurveExperiment.current());
    }

    /**
     * The ceiling on the benchmarked field (2601 clusters, root 51) follows the live curve. Today: 153 at s = 1, 1632
     * (32 x 51) at s = 0.0001. Intercept 0 makes m(1) = 2 (102); coefficient 1 makes m(1) = 2 (102); exponent 1/2 at
     * s = 0.25 makes m = 5 (255); cap 8 at s = 0.0001 makes m = 8 (408).
     */
    @Test
    void testForSelectivity_whenTheCurveMoves_thenTheCeilingFollowsIt() {
        assertEquals(153, PlanParams.forSelectivity(2601, 1.0).maxProbes());
        assertEquals(1632, PlanParams.forSelectivity(2601, 0.0001).maxProbes());

        ProbeCurveExperiment.set(ProbeCurveExperiment.current().withIntercept(0.0));
        assertEquals(102, PlanParams.forSelectivity(2601, 1.0).maxProbes(), "intercept");
        assertEquals(102, PlanParams.of(2601).maxProbes(), "the unfiltered ceiling follows too");
        ProbeCurveExperiment.reset();

        ProbeCurveExperiment.set(ProbeCurveExperiment.current().withCoefficient(1.0));
        assertEquals(102, PlanParams.forSelectivity(2601, 1.0).maxProbes(), "coefficient");
        ProbeCurveExperiment.reset();

        ProbeCurveExperiment.set(ProbeCurveExperiment.current().withExponent(0.5));
        assertEquals(255, PlanParams.forSelectivity(2601, 0.25).maxProbes(), "exponent");
        ProbeCurveExperiment.reset();

        ProbeCurveExperiment.set(ProbeCurveExperiment.current().withCap(8.0));
        assertEquals(408, PlanParams.forSelectivity(2601, 0.0001).maxProbes(), "cap");
        assertEquals(153, PlanParams.forSelectivity(2601, 1.0).maxProbes(), "a cap above m(1) does not bind at s = 1");
    }

    /** A curve that asks for nothing still leaves a usable range: one probe, floor and ceiling equal. */
    @Test
    void testForSelectivity_whenTheCurveIsZero_thenTheBoundsAreStillUsable() {
        ProbeCurveExperiment.set(ProbeCurveExperiment.current().withIntercept(0.0));
        ProbeCurveExperiment.set(ProbeCurveExperiment.current().withCoefficient(0.0));
        final PlanParams params = PlanParams.forSelectivity(2601, 1.0);
        assertEquals(1, params.maxProbes());
        assertEquals(1, params.minProbes());
    }

    // ---------------------------------------------------------------- wiring

    /**
     * The real KNNSettings wiring: registering seeds the holder from the node's settings, a transient update moves the
     * curve on the next plan, and clearing it puts the shipped curve back.
     */
    @Test
    void testRegister_whenTheClusterSettingsChange_thenTheLiveCurveFollows() {
        final ClusterSettings clusterSettings = new ClusterSettings(Settings.EMPTY, Set.copyOf(CURVE_SETTINGS));
        ProbeCurveExperiment.set(ProbeCurveExperiment.current().withCap(5.0)); // stale value from an earlier run; registering must
                                                                               // overwrite it
        KNNSettings.registerProbeCurveExperiment(clusterSettings, Settings.EMPTY);
        assertEquals(ProbeCurveExperiment.DEFAULT_CURVE, ProbeCurveExperiment.current());

        clusterSettings.applySettings(
            Settings.builder()
                .put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_INTERCEPT, "0.25")
                .put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_CAP, "12")
                .build()
        );
        assertEquals(new ProbeCurveExperiment.Curve(0.25, 2.0, 1.0 / 3.0, 12.0), ProbeCurveExperiment.current());
        assertEquals(612, PlanParams.forSelectivity(2601, 0.0001).maxProbes(), "cap 12 x root 51");

        clusterSettings.applySettings(
            Settings.builder()
                .put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_COEFFICIENT, "1")
                .put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_EXPONENT, "0.5")
                .build()
        );
        assertEquals(new ProbeCurveExperiment.Curve(1.0, 1.0, 0.5, 32.0), ProbeCurveExperiment.current(), "the keys left out reset");

        clusterSettings.applySettings(Settings.EMPTY);
        assertEquals(ProbeCurveExperiment.DEFAULT_CURVE, ProbeCurveExperiment.current());
    }

    /**
     * The production wiring: KNNSettings.initialize, as the plugin calls it, registers the curve, so a transient
     * update applied to the node's ClusterSettings reaches PlanParams and clearing it brings the shipped curve back.
     */
    @Test
    void testInitialize_whenAClusterSettingIsApplied_thenPlanParamsFollowsIt() {
        // Every node-scope setting the plugin registers, as the node's ClusterSettings carries them.
        final Set<Setting<?>> nodeScoped = new java.util.HashSet<>();
        KNNSettings.state().getSettings().stream().filter(Setting::hasNodeScope).forEach(nodeScoped::add);
        final ClusterSettings clusterSettings = new ClusterSettings(Settings.EMPTY, nodeScoped);
        final ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        when(clusterService.getSettings()).thenReturn(Settings.EMPTY);
        KNNSettings.state().initialize(mock(Client.class), clusterService);
        assertEquals(153, PlanParams.of(2601).maxProbes());

        clusterSettings.applySettings(
            Settings.builder()
                .put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_INTERCEPT, "0")
                .put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_CAP, "8")
                .build()
        );
        assertEquals(102, PlanParams.of(2601).maxProbes(), "intercept 0: m(1) = 2");
        assertEquals(408, PlanParams.forSelectivity(2601, 0.0001).maxProbes(), "cap 8");

        clusterSettings.applySettings(Settings.EMPTY);
        assertEquals(153, PlanParams.of(2601).maxProbes());
    }

    /**
     * The grouped validator refuses a bad value before any updater runs: another setting's consumer registered
     * ahead of the curve, in the same update, does not run either. Without the validator the curve's consumer
     * would throw only in the apply phase, after that other consumer had already run.
     */
    @Test
    void testRegister_whenAnUpdateCarriesABadValue_thenNoUpdaterInItRuns() {
        final Setting<Integer> other = Setting.intSetting("test.other", 0, Setting.Property.NodeScope, Setting.Property.Dynamic);
        final java.util.List<Setting<?>> all = new java.util.ArrayList<>(CURVE_SETTINGS);
        all.add(other);
        final ClusterSettings clusterSettings = new ClusterSettings(Settings.EMPTY, Set.copyOf(all));
        final int[] otherSeen = { 0 };
        clusterSettings.addSettingsUpdateConsumer(other, v -> otherSeen[0] = v);
        KNNSettings.registerProbeCurveExperiment(clusterSettings, Settings.EMPTY);

        final Settings update = Settings.builder()
            .put("test.other", 5)
            .put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_CAP, "NaN")
            .build();
        assertThrows(IllegalArgumentException.class, () -> clusterSettings.applySettings(update));
        assertEquals(0, otherSeen[0], "the other updater ran: the bad value was caught only in the apply phase");
        assertEquals(ProbeCurveExperiment.DEFAULT_CURVE, ProbeCurveExperiment.current());
    }

    /** Seeding reads the live view: a key already applied before registering is planned with from the start. */
    @Test
    void testRegister_whenAKeyWasAppliedBeforeRegistering_thenSeedsFromTheLiveView() {
        final ClusterSettings clusterSettings = new ClusterSettings(Settings.EMPTY, Set.copyOf(CURVE_SETTINGS));
        clusterSettings.applySettings(Settings.builder().put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_CAP, "8").build());
        KNNSettings.registerProbeCurveExperiment(clusterSettings, Settings.EMPTY);
        assertEquals(ProbeCurveExperiment.DEFAULT_CURVE.withCap(8.0), ProbeCurveExperiment.current());
    }

    /** A node started with the keys already set plans with them from the first query. */
    @Test
    void testRegister_whenTheNodeStartsWithTheKeysSet_thenSeedsFromThem() {
        final Settings node = Settings.builder()
            .put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_INTERCEPT, "0.5")
            .put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_COEFFICIENT, "1.5")
            .put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_EXPONENT, "0.25")
            .put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_CAP, "16")
            .build();
        KNNSettings.registerProbeCurveExperiment(new ClusterSettings(node, Set.copyOf(CURVE_SETTINGS)), node);
        assertEquals(new ProbeCurveExperiment.Curve(0.5, 1.5, 0.25, 16.0), ProbeCurveExperiment.current());
    }

    // ---------------------------------------------------------------- validation

    /** Only a finite number in range is accepted; a rejected update leaves the live curve untouched. */
    @ParameterizedTest(name = "cap = {0}")
    @ValueSource(strings = { "NaN", "Infinity", "-Infinity", "0.5", "1000.5", "abc", "" })
    void testCapSetting_whenTheValueIsNotAFiniteNumberInRange_thenRejected(String value) {
        final Settings bad = Settings.builder().put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_CAP, value).build();
        assertThrows(IllegalArgumentException.class, () -> KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_CAP_SETTING.get(bad));

        final ClusterSettings clusterSettings = new ClusterSettings(Settings.EMPTY, Set.copyOf(CURVE_SETTINGS));
        KNNSettings.registerProbeCurveExperiment(clusterSettings, Settings.EMPTY);
        assertThrows(IllegalArgumentException.class, () -> clusterSettings.validate(bad, true));

        // An update that would reach the node anyway is refused whole: the good key beside the bad one is not applied.
        final Settings mixed = Settings.builder().put(bad).put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_INTERCEPT, "0.5").build();
        clusterSettings.applySettings(Settings.builder().put(KNNSettings.KNN_CLUSTERANN_EXPERIMENT_PROBE_CURVE_COEFFICIENT, "1").build());
        final ProbeCurveExperiment.Curve before = ProbeCurveExperiment.current();
        assertEquals(1.0, before.coefficient());
        assertThrows(IllegalArgumentException.class, () -> clusterSettings.applySettings(mixed));
        assertEquals(before, ProbeCurveExperiment.current());
    }

    /**
     * A caller whose ClusterSettings carries only its own keys (QuantizationStateCacheTests builds one) still
     * initializes KNNSettings: the probe-curve keys are optional there, and the node runs the shipped curve.
     */
    @Test
    void testRegister_whenTheClusterSettingsDoNotCarryTheKeys_thenStillRegistersAndKeepsTheDefault() {
        final ClusterSettings narrow = new ClusterSettings(Settings.EMPTY, Set.of(KNNSettings.QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING));
        ProbeCurveExperiment.set(ProbeCurveExperiment.current().withCap(5.0));
        KNNSettings.registerProbeCurveExperiment(narrow, Settings.EMPTY);
        assertEquals(ProbeCurveExperiment.DEFAULT_CURVE, ProbeCurveExperiment.current());
    }

    /** A zero coefficient drops the selectivity term, so even s = 0 plans the intercept, not a NaN one-probe plan. */
    @Test
    void testMultiplier_whenTheCoefficientIsZero_thenIsTheInterceptAtEverySelectivity() {
        final ProbeCurveExperiment.Curve flat = ProbeCurveExperiment.DEFAULT_CURVE.withCoefficient(0.0);
        assertEquals(1.0, flat.multiplier(0.0));
        assertEquals(1.0, flat.multiplier(0.001));
        assertEquals(1.0, flat.multiplier(1.0));
        ProbeCurveExperiment.set(flat);
        assertEquals(51, PlanParams.forSelectivity(2601, 0.0).maxProbes());
    }

    /** The lower bounds: a negative intercept, coefficient or exponent, or an exponent above 1, is refused. */
    @ParameterizedTest(name = "{0} = {1}")
    @CsvSource({
        "knn.clusterann.experiment.probe_curve_intercept, -0.1",
        "knn.clusterann.experiment.probe_curve_coefficient, -1",
        "knn.clusterann.experiment.probe_curve_exponent, -0.01",
        "knn.clusterann.experiment.probe_curve_exponent, 1.01" })
    void testSettings_whenOutOfRange_thenRejected(String key, String value) {
        final Setting<Double> setting = CURVE_SETTINGS.stream().filter(s -> s.getKey().equals(key)).findFirst().orElseThrow();
        final Settings bad = Settings.builder().put(key, value).build();
        final IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> setting.get(bad));
        assertTrue(e.getMessage().contains(key), e.getMessage());
    }

    /** The edges of each range are accepted. */
    @ParameterizedTest(name = "{0} = {1}")
    @CsvSource({
        "knn.clusterann.experiment.probe_curve_intercept, 0",
        "knn.clusterann.experiment.probe_curve_coefficient, 0",
        "knn.clusterann.experiment.probe_curve_exponent, 0",
        "knn.clusterann.experiment.probe_curve_exponent, 1",
        "knn.clusterann.experiment.probe_curve_cap, 1",
        "knn.clusterann.experiment.probe_curve_cap, 1000" })
    void testSettings_whenAtTheEdgeOfTheRange_thenAccepted(String key, String value) {
        final Setting<Double> setting = CURVE_SETTINGS.stream().filter(s -> s.getKey().equals(key)).findFirst().orElseThrow();
        assertEquals(Double.parseDouble(value), setting.get(Settings.builder().put(key, value).build()));
    }
}
