/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import java.util.Locale;
import java.util.Objects;

/**
 * EXPERIMENT (do not merge): shard-level planning as live switches, for the shard-budget benchmark.
 *
 * <p>Fed by three dynamic cluster settings through {@code KNNSettings} (the mechanism the knee and early-stop sweeps
 * use), published as one immutable {@link Config} behind a volatile, and read by {@code ClusterANNQuery} once per
 * weight and by {@link ShardPlanner} once per plan:
 *
 * <ul>
 *   <li>{@code knn.clusterann.experiment.shard_plan_enabled}, default true: off searches every segment without a
 *       shard plan, so each reader plans its own segment exactly as before shard planning;
 *   <li>{@code knn.clusterann.experiment.shard_plan_budget}, {@code capped} (rule H, the default) or {@code rule}
 *       (rule R uncapped);
 *   <li>{@code knn.clusterann.experiment.shard_plan_min_multiplier}, default 8: the floor on the shard ceiling's
 *       multiplier, maxProbes = ceil(max(floor, m(s)) * floor(sqrt(C_total))). 0 lets the live probe curve's m(s)
 *       (a static m when its coefficient is 0) set the shard ceiling on its own.
 * </ul>
 *
 * A node that never sets the keys plans exactly as the shard-plan branch does without them.
 */
public final class ShardPlanExperiment {

    /** The branch's behaviour: shard planning on. */
    public static final boolean DEFAULT_ENABLED = true;
    /** The branch's default budget: rule H. */
    public static final ShardPlanner.Budget DEFAULT_BUDGET = ShardPlanner.Budget.CAPPED;
    /** The branch's rule R floor. */
    public static final double DEFAULT_MIN_MULTIPLIER = ShardPlanner.RULE_MIN_MULTIPLIER;
    public static final double MIN_MIN_MULTIPLIER = 0.0;
    public static final double MAX_MIN_MULTIPLIER = 1000.0;

    /**
     * One consistent set of the three switches.
     *
     * @param enabled whether queries plan per shard
     * @param budget how the shard sets X
     * @param minMultiplier floor on the shard ceiling's multiplier, finite, in [0, 1000]
     */
    public record Config(boolean enabled, ShardPlanner.Budget budget, double minMultiplier) {
        public Config {
            Objects.requireNonNull(budget, "budget");
            if (!(minMultiplier >= MIN_MIN_MULTIPLIER && minMultiplier <= MAX_MIN_MULTIPLIER)) {
                throw new IllegalArgumentException(
                    "shard_plan_min_multiplier must be in [" + MIN_MIN_MULTIPLIER + ", " + MAX_MIN_MULTIPLIER + "], got: " + minMultiplier
                );
            }
        }
    }

    public static final Config DEFAULT = new Config(DEFAULT_ENABLED, DEFAULT_BUDGET, DEFAULT_MIN_MULTIPLIER);

    private static volatile Config current = DEFAULT;

    private ShardPlanExperiment() {}

    /** The switches every later query and plan reads. */
    public static Config current() {
        return current;
    }

    /** Sets the switches every later query and plan reads, in one write. */
    public static void set(final Config config) {
        current = Objects.requireNonNull(config);
    }

    /** Puts the branch's behaviour back. */
    public static void reset() {
        current = DEFAULT;
    }

    /**
     * The budget a setting names: {@code capped} or {@code rule}, case-insensitive.
     *
     * @throws IllegalArgumentException for any other value
     */
    public static ShardPlanner.Budget parseBudget(final String value, final String key) {
        if (value != null) {
            switch (value.trim().toLowerCase(Locale.ROOT)) {
                case "capped":
                    return ShardPlanner.Budget.CAPPED;
                case "rule":
                    return ShardPlanner.Budget.RULE;
                default:
                    break;
            }
        }
        throw new IllegalArgumentException("[" + key + "] must be [capped] or [rule], got [" + value + "]");
    }
}
