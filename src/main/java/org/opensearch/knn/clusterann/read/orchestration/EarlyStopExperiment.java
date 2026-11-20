/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import java.util.Objects;

/**
 * EXPERIMENT (do not merge): the scan early stop ({@link EarlyStopParams}) as live switches, for the early-stop sweep.
 *
 * <p>Fed by the dynamic cluster settings {@code knn.clusterann.experiment.early_stop_enabled}, {@code ..._patience},
 * {@code ..._epsilon} and {@code ..._floor} through {@code KNNSettings}, and read by the reader once per segment search.
 * Off, the default, is the shipped behaviour: every scan runs its whole plan, exactly as before the stop existed. On,
 * the defaults are P = {@value EarlyStopParams#DEFAULT_PATIENCE}, epsilon {@code max(1, ceil(0.01 * k))} and floor the
 * plan's minProbes; epsilon and floor take an absolute override, and {@code -1} ({@link EarlyStopParams#AUTO}) means
 * derived.
 *
 * <p>The live value is one immutable {@link EarlyStopParams} behind one volatile field, and {@code KNNSettings}
 * publishes a whole one per cluster-settings update (one grouped consumer over the four keys), so a scan that reads it
 * once never mixes the old value of one switch with the new value of another.
 */
public final class EarlyStopExperiment {

    /** The shipped behaviour: no early stop. */
    public static final boolean DEFAULT_ENABLED = false;
    /** Patience once enabled. */
    public static final int DEFAULT_PATIENCE = EarlyStopParams.DEFAULT_PATIENCE;
    /** Epsilon once enabled: derived, max(1, ceil(0.01 * k)). */
    public static final int DEFAULT_EPSILON = EarlyStopParams.AUTO;
    /** Floor once enabled: derived, the plan's minProbes. */
    public static final int DEFAULT_FLOOR = EarlyStopParams.AUTO;

    /** Bounds each setting accepts; finite and wide enough for any sweep. */
    public static final int MIN_PATIENCE = 1;
    public static final int MAX_PATIENCE = 10_000;
    public static final int MIN_EPSILON = EarlyStopParams.AUTO;
    public static final int MAX_EPSILON = 10_000;
    public static final int MIN_FLOOR = EarlyStopParams.AUTO;
    public static final int MAX_FLOOR = 100_000;

    private static volatile EarlyStopParams current = EarlyStopParams.OFF;

    private EarlyStopExperiment() {}

    /** The early stop every later scan reads: {@link EarlyStopParams#OFF} unless the switch is on. */
    public static EarlyStopParams current() {
        return current;
    }

    /**
     * The early stop the four switches describe.
     *
     * @param enabled whether the stop is on; off gives {@link EarlyStopParams#OFF} whatever the other three say
     * @param patience quiet clusters in a row that stop the scan, at least 1
     * @param epsilon fewest top-k inserts that make a cluster useful, or {@link EarlyStopParams#AUTO}
     * @param floor scanned clusters before any counts, or {@link EarlyStopParams#AUTO}
     * @throws IllegalArgumentException if a value is outside its range (0 for epsilon included)
     */
    public static EarlyStopParams of(final boolean enabled, final int patience, final int epsilon, final int floor) {
        checkRange("patience", patience, MIN_PATIENCE, MAX_PATIENCE);
        checkRange("epsilon", epsilon, MIN_EPSILON, MAX_EPSILON);
        checkRange("floor", floor, MIN_FLOOR, MAX_FLOOR);
        // The record refuses epsilon 0 (neither AUTO nor >= 1), whether or not the stop is on.
        final EarlyStopParams params = new EarlyStopParams(patience, epsilon, floor);
        return enabled ? params : EarlyStopParams.OFF;
    }

    /** Sets the early stop every later scan reads, in one write. */
    public static void set(final EarlyStopParams params) {
        current = Objects.requireNonNull(params);
    }

    /** Puts the shipped behaviour back: off. */
    public static void reset() {
        current = EarlyStopParams.OFF;
    }

    /**
     * Parses the epsilon setting: {@link EarlyStopParams#AUTO} or an integer in {@code [1, MAX_EPSILON]}, nothing else.
     *
     * @param value the setting's string value
     * @param key the setting's name, for the error
     */
    public static int parseEpsilon(final String value, final String key) {
        final int parsed;
        try {
            parsed = Integer.parseInt(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Failed to parse value [" + value + "] for setting [" + key + "]", e);
        }
        if (parsed != EarlyStopParams.AUTO && (parsed < 1 || parsed > MAX_EPSILON)) {
            throw new IllegalArgumentException(
                "Failed to parse value [" + value + "] for setting [" + key + "], must be -1 (derived) or in [1, " + MAX_EPSILON + "]"
            );
        }
        return parsed;
    }

    private static void checkRange(final String name, final int value, final int min, final int max) {
        if (value < min || value > max) {
            throw new IllegalArgumentException("early stop " + name + " must be in [" + min + ", " + max + "], got: " + value);
        }
    }
}
