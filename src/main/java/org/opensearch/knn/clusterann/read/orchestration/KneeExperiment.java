/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

/**
 * EXPERIMENT (do not merge): knee pruning as a live on/off switch, for the static-multiplier sweep.
 *
 * <p>Fed by the dynamic cluster setting {@code knn.clusterann.experiment.knee_enabled} through {@code KNNSettings}
 * (the mechanism the knee-factor sweep used), and read by {@link CentroidPlanner} on every plan. On, the default, is
 * the shipped behaviour: every plan is cut at the knee. Off plans every cluster the ranking kept, so a plan holds
 * exactly the probe ceiling, or more where holding {@code k} accepted vectors needs more.
 */
public final class KneeExperiment {

    /** The shipped behaviour: knee pruning on. */
    public static final boolean DEFAULT_ENABLED = true;

    private static volatile boolean enabled = DEFAULT_ENABLED;

    private KneeExperiment() {}

    /** Whether every later plan is cut at the knee. */
    public static boolean enabled() {
        return enabled;
    }

    /** Sets whether every later plan is cut at the knee. */
    public static void setEnabled(final boolean on) {
        enabled = on;
    }

    /** Puts the shipped behaviour back. */
    public static void reset() {
        enabled = DEFAULT_ENABLED;
    }
}
