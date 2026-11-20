/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.opensearch.knn.clusterann.read.Clusters;

import java.util.Objects;

/**
 * One segment's share of a {@link ShardPlanner.Plan}, handed to the reader through the per-segment collector's
 * {@link org.apache.lucene.search.KnnCollector#getSearchStrategy()} (Lucene's collector decorators forward it).
 *
 * <p>It carries counts only, {@link #segmentParams()} = {@code (minProbes_s, maxProbes_s)}: no cluster list and no
 * ranked order. A reader that finds this strategy skips its own planning (no knee, no per-segment hold of k), re-ranks
 * its own centroids with the planner's keys and the filter's eligibility, and probes its first {@code maxProbes_s}
 * ({@link CentroidPlanner#planWithin}). {@code [minProbes_s, maxProbes_s]} is the window scoring-time pruning may act
 * in. {@code maxProbes_s = 0} means the shard gave the segment nothing, and the segment returns no hits.
 *
 * <p>The plan was made against one segment's {@link Clusters}; {@link #plannedFor} lets the reader refuse a plan made
 * for another segment or field, whose counts would mean nothing there.
 */
public final class ShardPlanStrategy extends KnnSearchStrategy {

    private final Clusters clusters;
    private final PlanParams segmentParams;
    private final PlanParams shardParams;

    /**
     * @param clusters the segment's clusters the plan was made against
     * @param segmentParams {@code (minProbes_s, maxProbes_s)}: this segment's clusters inside the shard's top
     *     {@code minProbes} and inside its cut
     * @param shardParams the shard's bounds
     */
    public ShardPlanStrategy(final Clusters clusters, final PlanParams segmentParams, final PlanParams shardParams) {
        this.clusters = Objects.requireNonNull(clusters, "clusters");
        this.segmentParams = Objects.requireNonNull(segmentParams, "segmentParams");
        this.shardParams = Objects.requireNonNull(shardParams, "shardParams");
    }

    /** {@code (minProbes_s, maxProbes_s)} for this segment; either may be 0. */
    public PlanParams segmentParams() {
        return segmentParams;
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
        return clusters == o.clusters && segmentParams.equals(o.segmentParams) && shardParams.equals(o.shardParams);
    }

    @Override
    public int hashCode() {
        return Objects.hash(System.identityHashCode(clusters), segmentParams, shardParams);
    }

    @Override
    public String toString() {
        return "ShardPlanStrategy[segmentParams=" + segmentParams + ", shardParams=" + shardParams + "]";
    }
}
