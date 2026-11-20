/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

/**
 * When a scan may stop short of its plan: once {@code patience} clusters in a row have each put fewer than
 * {@code epsilon} vectors into the collector's top k. Ported from SearchServicesKnnVectorFormats (CR-324504354).
 *
 * <p>The planner decides which clusters a query <em>may</em> need, from centroid distances alone. Whether the later
 * ones still matter is only visible during the scan: once the top k is full and cluster after cluster adds almost
 * nothing to it, the clusters still ahead in the plan, which are farther from the query, are unlikely to add more.
 * This is that soft stop. The plan's ceiling ({@link PlanParams#maxProbes()}) is still the hard bound.
 *
 * <p>A cluster counts toward {@code patience} only when all of these hold:
 *
 * <ul>
 *   <li>it is not empty;
 *   <li>under a filter, it held at least one candidate the filter and the SOAR dedup let through, so a cluster
 *       the filter emptied is no evidence that the query has run out of neighbours; unfiltered, every non-empty
 *       cluster counts. A segment with deletions is filtered by its live docs, so there the rule applies too, which
 *       can only delay a stop;
 *   <li>at least {@code floor} clusters have been scanned, this one included (every scanned cluster, counted or not);
 *   <li>the collector is full (it holds k results), since before that every vector is an insert and quiet means
 *       nothing.
 * </ul>
 *
 * A cluster that counts and puts at least {@code epsilon} vectors into the top k resets the run, so a stop means
 * {@code patience} quiet clusters in a row. The earliest a stop can come is therefore after
 * {@code max(floor, 1) + patience - 1} clusters. While enabled, the scan also stops once the collector reports
 * {@link org.apache.lucene.search.KnnCollector#earlyTerminated()}.
 *
 * <p><b>Off by default.</b> {@link #OFF} leaves the scan exactly as it was before this existed. In k-NN the stop is
 * turned on only by the EXPERIMENT switches in {@link EarlyStopExperiment}.
 *
 * <p>Immutable, so one instance may be shared by every query that uses it.
 *
 * @param patience quiet clusters in a row that stop the scan; {@code 0} disables the stop entirely, and then
 *     {@code epsilon} and {@code floor} are normalised to {@link #AUTO} so every disabled instance equals {@link #OFF}
 * @param epsilon fewest top-k inserts that make a cluster useful (resetting the run), or {@link #AUTO} for
 *     {@code max(1, ceil(k / 100))}; never more than k, so a cluster that fills the queue is always useful (an
 *     absolute override above k is clamped to k). k is the collector's: for ClusterANNQuery its candidateK, the query's
 *     k or k x oversample when a rescore follows; under Lucene's optimistic multi-segment search, the per-segment k
 * @param floor scanned clusters, the current one included, at which clusters start to count toward {@code patience},
 *     or {@link #AUTO} for the minProbes that {@link PlanParams#of(int)} gives the field, which is the floor the plan
 *     itself was cut against. Re-derived at scan time, so a probe-curve EXPERIMENT update landing between plan and scan
 *     can shift it for that one search
 *
 * <p>Known asymmetries, kept from the formats rule: unfiltered, a non-empty cluster counts even when every vector in it
 * is a SOAR copy already scored (it scores nothing, so it is quiet), while under a filter or live-docs bits such a
 * cluster does not count; and a stop counts as one even when only empty probes were left after it.
 */
public record EarlyStopParams(int patience, int epsilon, int floor) {

    /** Marks {@code epsilon} or {@code floor} as derived per query rather than fixed. */
    public static final int AUTO = -1;

    /** Suggested patience. */
    public static final int DEFAULT_PATIENCE = 3;

    /** Suggested epsilon is {@code max(1, ceil(k / EPSILON_K_DIVISOR))}, one percent of k. */
    public static final int EPSILON_K_DIVISOR = 100;

    /** No early stop: the scan visits every probe in the plan. The default. */
    public static final EarlyStopParams OFF = new EarlyStopParams(0, AUTO, AUTO);

    /** The suggested settings to start from once the stop is enabled: P = 3, epsilon and floor derived per query. */
    public static final EarlyStopParams PROVISIONAL = new EarlyStopParams(DEFAULT_PATIENCE, AUTO, AUTO);

    public EarlyStopParams {
        if (patience < 0) {
            throw new IllegalArgumentException("patience must be >= 0 (0 disables the stop), got: " + patience);
        }
        if (epsilon != AUTO && epsilon < 1) {
            throw new IllegalArgumentException("epsilon must be >= 1 or AUTO, got: " + epsilon);
        }
        if (floor != AUTO && floor < 0) {
            throw new IllegalArgumentException("floor must be >= 0 or AUTO, got: " + floor);
        }
        if (patience == 0) {
            // Every disabled instance is OFF, so two queries that both leave the stop off compare equal.
            epsilon = AUTO;
            floor = AUTO;
        }
    }

    /** Whether the scan may stop early at all. */
    public boolean enabled() {
        return patience > 0;
    }

    /**
     * The epsilon to apply for a query wanting {@code k} results.
     *
     * @param k results the collector holds when full
     * @return {@code epsilon}, or the AUTO value, clamped to {@code [1, k]}
     */
    public int epsilonFor(final int k) {
        final int derived = epsilon != AUTO ? epsilon : Math.ceilDiv(k, EPSILON_K_DIVISOR);
        return Math.max(1, Math.min(derived, k));
    }

    /**
     * The floor to apply on a field with {@code numClusters} clusters.
     *
     * @param numClusters the field's cluster count, from which {@link PlanParams#of(int)} derives the plan's floor
     */
    public int floorFor(final int numClusters) {
        return floor != AUTO ? floor : PlanParams.of(numClusters).minProbes();
    }
}
