/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.opensearch.knn.clusterann.read.Clusters;

import java.util.Arrays;
import java.util.Objects;

/**
 * One segment's share of a {@link ShardPlanner.Plan}, handed to the reader through the per-segment collector's
 * {@link org.apache.lucene.search.KnnCollector#getSearchStrategy()} (Lucene's collector decorators forward it).
 *
 * <p>A reader that finds this strategy skips its own planning and scans exactly {@link #probes()}, in order; an empty
 * list means the shard gave the segment nothing, and the segment returns no hits. {@link #shardParams()} are the shard's
 * bounds, carried so scoring-time pruning can read them.
 *
 * <p>The plan was made against one segment's {@link Clusters}; {@link #plannedFor} lets the reader refuse a plan made
 * for another segment or field, whose ordinals would name the wrong clusters.
 */
public final class ShardPlanStrategy extends KnnSearchStrategy {

    private final Clusters clusters;
    private final int[] probes;
    private final PlanParams shardParams;

    /**
     * @param clusters the segment's clusters the plan was made against
     * @param probes centroid ordinals to scan, closest first; not copied, so the caller must not modify it
     * @param shardParams the shard's bounds
     */
    public ShardPlanStrategy(final Clusters clusters, final int[] probes, final PlanParams shardParams) {
        this.clusters = Objects.requireNonNull(clusters, "clusters");
        this.probes = Objects.requireNonNull(probes, "probes");
        this.shardParams = Objects.requireNonNull(shardParams, "shardParams");
    }

    /** Centroid ordinals to scan, closest first. Not a copy: callers must not modify it. */
    public int[] probes() {
        return probes;
    }

    /** The shard's bounds: {@code maxProbes} is R, {@code minProbes} the floor at the shard's cluster count. */
    public PlanParams shardParams() {
        return shardParams;
    }

    /** Whether this plan was made against exactly these clusters (the same segment and field). */
    public boolean plannedFor(final Clusters candidate) {
        return clusters == candidate;
    }

    /** Nothing to advance: the plan is fixed before the scan starts. */
    @Override
    public void nextVectorsBlock() {}

    @Override
    public boolean equals(final Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof ShardPlanStrategy o)) {
            return false;
        }
        return clusters == o.clusters && Arrays.equals(probes, o.probes) && shardParams.equals(o.shardParams);
    }

    @Override
    public int hashCode() {
        return Objects.hash(System.identityHashCode(clusters), Arrays.hashCode(probes), shardParams);
    }

    @Override
    public String toString() {
        return "ShardPlanStrategy[probes=" + probes.length + ", shardParams=" + shardParams + "]";
    }
}
