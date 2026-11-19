/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import java.util.Locale;
import java.util.Objects;

/**
 * EXPERIMENT (do not merge): the probe-budget curve's four constants as live values, for the probe-budget retune.
 *
 * <p>{@link PlanParams} sizes a plan's ceiling as {@code ceil(m(s) * floor(sqrt(C)))} with
 * {@code m(s) = min(cap, intercept + coefficient / s^exponent)}. This holder is fed by the dynamic cluster settings
 * {@code knn.clusterann.experiment.probe_curve_intercept}, {@code ..._coefficient}, {@code ..._exponent} and
 * {@code ..._cap} through {@code KNNSettings}, and read on every plan, so one jar sweeps every curve. Each default is
 * the shipped constant (1, 2, 1/3, 32), so a node that never sets them plans exactly as before.
 *
 * <p>The four values live in one immutable {@link Curve} behind one volatile field, and {@code KNNSettings} publishes
 * a whole curve per cluster-settings update (one grouped consumer over the four keys), so a plan that reads the curve
 * once never mixes the old value of one constant with the new value of another.
 */
public final class ProbeCurveExperiment {

    /** The shipped intercept: m(1) = 1 + 2 = 3. */
    public static final double DEFAULT_INTERCEPT = 1.0;
    /** The shipped coefficient. */
    public static final double DEFAULT_COEFFICIENT = 2.0;
    /** The shipped exponent: the cube root. */
    public static final double DEFAULT_EXPONENT = PlanParams.SELECTIVITY_EXPONENT;
    /** The shipped cap on m(s). */
    public static final double DEFAULT_CAP = PlanParams.CAP_MULTIPLIER;

    /** Bounds each setting accepts. Finite, and wide enough for any sweep; NaN and infinities are refused. */
    public static final double MIN_INTERCEPT = 0.0;
    public static final double MAX_INTERCEPT = 1000.0;
    public static final double MIN_COEFFICIENT = 0.0;
    public static final double MAX_COEFFICIENT = 1000.0;
    public static final double MIN_EXPONENT = 0.0;
    public static final double MAX_EXPONENT = 1.0;
    public static final double MIN_CAP = 1.0;
    public static final double MAX_CAP = 1000.0;

    /**
     * One consistent set of the curve's constants.
     *
     * @param intercept the constant term of m(s)
     * @param coefficient the weight of the selectivity term
     * @param exponent the power selectivity is raised to
     * @param cap the most m(s) may be
     */
    public record Curve(double intercept, double coefficient, double exponent, double cap) {

        /**
         * m(s) = min(cap, intercept + coefficient / s^exponent). A zero coefficient drops the selectivity term
         * outright, so s = 0 gives the intercept rather than 0 / 0.
         */
        public double multiplier(final double selectivity) {
            final double term = coefficient == 0.0 ? 0.0 : coefficient / Math.pow(selectivity, exponent);
            return Math.min(cap, intercept + term);
        }

        /** This curve with another intercept. */
        public Curve withIntercept(final double value) {
            return new Curve(value, coefficient, exponent, cap);
        }

        /** This curve with another coefficient. */
        public Curve withCoefficient(final double value) {
            return new Curve(intercept, value, exponent, cap);
        }

        /** This curve with another exponent. */
        public Curve withExponent(final double value) {
            return new Curve(intercept, coefficient, value, cap);
        }

        /** This curve with another cap. */
        public Curve withCap(final double value) {
            return new Curve(intercept, coefficient, exponent, value);
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT, "m(s)=min(%s, %s + %s / s^%s)", cap, intercept, coefficient, exponent);
        }
    }

    /** The shipped curve. */
    public static final Curve DEFAULT_CURVE = new Curve(DEFAULT_INTERCEPT, DEFAULT_COEFFICIENT, DEFAULT_EXPONENT, DEFAULT_CAP);

    private static volatile Curve curve = DEFAULT_CURVE;

    private ProbeCurveExperiment() {}

    /** The curve every plan from now on reads. */
    public static Curve current() {
        return curve;
    }

    /** Sets the whole curve every later plan reads, in one write. */
    public static void set(final Curve newCurve) {
        curve = Objects.requireNonNull(newCurve);
    }

    /** Puts every constant back to its shipped value. */
    public static void reset() {
        curve = DEFAULT_CURVE;
    }

    /**
     * Parses one setting's value: a finite number inside {@code [min, max]}, nothing else.
     *
     * @param value the setting's string value
     * @param key the setting's name, for the error
     * @param min the smallest accepted value
     * @param max the largest accepted value
     */
    public static double parse(final String value, final String key, final double min, final double max) {
        final double parsed;
        try {
            parsed = Double.parseDouble(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Failed to parse value [" + value + "] for setting [" + key + "]", e);
        }
        if (Double.isFinite(parsed) == false || parsed < min || parsed > max) {
            throw new IllegalArgumentException(
                "Failed to parse value [" + value + "] for setting [" + key + "], must be a finite number in [" + min + ", " + max + "]"
            );
        }
        return parsed;
    }
}
