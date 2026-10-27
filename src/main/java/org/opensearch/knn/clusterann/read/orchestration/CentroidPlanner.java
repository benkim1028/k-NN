/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.ConjunctionUtils;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.LongValues;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.opensearch.knn.clusterann.read.CentroidVectorValues;
import org.opensearch.knn.clusterann.read.Clusters;
import org.opensearch.knn.plugin.stats.ClusterANNQueryValue;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Chooses which clusters a query should visit.
 *
 * <p>Sweeps every centroid, keeps the {@code maxProbes} nearest non-empty clusters, and returns their ordinals
 * closest-first. Under a filter whose accepted vectors all fit in {@code k}, it keeps every cluster holding one
 * instead. The scan relies on that order and does not restore it.
 *
 * <p>Ranks on a distance-like key (smaller is closer)
 *
 * <p>The ceiling bounds what a query may spend; {@link CentroidKneePruner} cuts what it should not. A filtered plan
 * is cut over its eligible clusters too, at any selectivity, unless it is planning every accepted vector, and never
 * short of the clusters it takes to hold {@code k} accepted vectors (or of everything ranked, when even that holds
 * fewer).
 *
 * <p>Stateless and thread-safe: the only cursor it uses is a private copy taken per call.
 */
public final class CentroidPlanner {

    private static final int[] NO_PROBES = new int[0];

    private CentroidPlanner() {}

    /**
     * Rank this field's centroids against the query and name the clusters to visit, considering only clusters reached
     * by the document-space filter when one is present.
     *
     * <p>A filter that is null or has null {@link AcceptDocs#bits()} takes the unfiltered path without reading
     * ordinal-to-centroid assignments. Lucene drops a filter that rewrites to match-all and hands each leaf its live
     * docs, so a match-all filter on a segment without deletions arrives here with null bits. Any other filter takes
     * the filtered path, even if it happens to accept every document.
     *
     * <p>When every accepted vector fits in {@code k}, the caller wants all of them, so every centroid holding one is
     * probed rather than only the {@code maxProbes} nearest.
     *
     * @param clusters the field's clusters
     * @param query the query vector, in the space it arrived in; not modified
     * @param params the bounds to choose within; a selective filter may raise the ceiling above {@code maxProbes}
     * @param k the number of results the caller wants
     * @param acceptDocs document-space filter, or null for an unfiltered search
     * @param vectorDocsSupplier a new iterator over the documents that have this vector field, exposing each one's
     *     vector ordinal; called only when a sparse field is filtered
     * @return centroid ordinals to probe, ordered closest-first; empty if no non-empty cluster matches
     */
    public static int[] plan(
        Clusters clusters,
        float[] query,
        PlanParams params,
        int k,
        AcceptDocs acceptDocs,
        IOSupplier<KnnVectorValues.DocIndexIterator> vectorDocsSupplier
    ) throws IOException {
        if (clusters.numClusters() == 0) {
            return NO_PROBES;
        }
        if (acceptDocs == null || acceptDocs.bits() == null) {
            return plan(clusters, query, params);
        }

        final LongValues ordToCentroid = clusters.ordToCentroid();
        final FixedBitSet acceptedCentroids = new FixedBitSet(clusters.numClusters());
        final int[] acceptedPerCluster = new int[clusters.numClusters()];
        final DocIdSetIterator acceptedDocs = acceptDocs.iterator();
        int acceptedVectors = 0;
        if (clusters.hasDenseVectorValues()) {
            for (int doc = acceptedDocs.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = acceptedDocs.nextDoc()) {
                accept(acceptedCentroids, acceptedPerCluster, (int) ordToCentroid.get(doc));
                acceptedVectors++;
            }
        } else {
            final KnnVectorValues.DocIndexIterator vectorDocs = vectorDocsSupplier.get();
            final DocIdSetIterator acceptedVectorDocs = ConjunctionUtils.intersectIterators(List.of(acceptedDocs, vectorDocs));
            while (acceptedVectorDocs.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                accept(acceptedCentroids, acceptedPerCluster, (int) ordToCentroid.get(vectorDocs.index()));
                acceptedVectors++;
            }
        }
        final int eligibleClusters = acceptedCentroids.cardinality();
        ClusterANNQueryValue.FILTERED_SEGMENT_SCANS.increment();
        ClusterANNQueryValue.ELIGIBLE_CLUSTERS.incrementBy(eligibleClusters);
        if (eligibleClusters == 0) {
            return NO_PROBES;
        }
        final double selectivity = (double) acceptedVectors / clusters.numVectors();
        final int budget = PlanParams.forSelectivity(clusters.numClusters(), selectivity).maxProbes();
        final boolean everyAccepted = acceptedVectors <= k;
        final int maxProbes = everyAccepted ? eligibleClusters : Math.max(budget, params.maxProbes());
        final NeighborQueue nearest = rank(clusters, query, maxProbes, acceptedCentroids);
        // Never cut when the caller wants every accepted vector: that would drop results the plan exists to return.
        if (everyAccepted) {
            return closestFirst(nearest);
        }
        return closestFirstPruned(nearest, params.minProbes(), acceptedPerCluster, k);
    }

    /** Marks a centroid eligible and counts one more accepted vector assigned to it. */
    private static void accept(final FixedBitSet acceptedCentroids, final int[] acceptedPerCluster, final int centroid) {
        acceptedCentroids.set(centroid);
        acceptedPerCluster[centroid]++;
    }

    /**
     * Rank every centroid in this field against the query and name the clusters to visit.
     *
     * <p>Costs one pass over the centroid region ({@code numClusters × dimension} floats), before any posting is
     * touched. Empty clusters are dropped during the sweep so they never take up a probe slot.
     *
     * @param clusters the field's clusters
     * @param query the query vector, in the space it arrived in; not modified
     * @param params the bounds to choose within
     * @return centroid ordinals to probe, ordered closest-first; empty if the field has no non-empty cluster
     */
    static int[] plan(Clusters clusters, float[] query, PlanParams params) throws IOException {
        if (clusters.numClusters() == 0) {
            return NO_PROBES;
        }
        return closestFirstPruned(rank(clusters, query, params.maxProbes(), null), params.minProbes(), null, 0);
    }

    /**
     * Rank the candidate centroids, keeping the {@code maxProbes} nearest non-empty ones in a heap.
     *
     * @param eligibleCentroids ordinals to consider, or null when nothing is filtered out
     */
    private static NeighborQueue rank(Clusters clusters, float[] query, int maxProbes, FixedBitSet eligibleCentroids) throws IOException {
        final int numClusters = clusters.numClusters();
        final VectorSimilarityFunction similarity = clusters.clusterMeta().similarityFunction();
        final CentroidVectorValues centroids = clusters.centroids();
        final float queryNormSq = similarity == VectorSimilarityFunction.EUCLIDEAN ? VectorUtil.dotProduct(query, query) : 0f;

        final NeighborQueue nearest = new NeighborQueue(maxProbes, true);
        for (int ordinal = nextCandidate(eligibleCentroids, 0); ordinal < numClusters; ordinal = nextCandidate(
            eligibleCentroids,
            ordinal + 1
        )) {
            if (clusters.clusterSize(ordinal) != 0) {
                final float[] centroid = centroids.vectorValue(ordinal);
                nearest.insertWithOverflow(ordinal, distanceKey(similarity, query, queryNormSq, centroid, centroids.norm()));
            }
        }

        return nearest;
    }

    /** The first candidate ordinal at or after {@code from}: {@code from} itself when nothing is filtered out. */
    private static int nextCandidate(final FixedBitSet eligible, final int from) {
        if (eligible == null) {
            return from;
        }
        return from < eligible.length() ? eligible.nextSetBit(from) : DocIdSetIterator.NO_MORE_DOCS;
    }

    /**
     * Distance-like key from the query to a centroid, smaller meaning closer. Only EUCLIDEAN is a true distance:
     *
     * <ul>
     *   <li>EUCLIDEAN — {@code ‖q‖² − 2⟨q,c⟩ + ‖c‖²}, i.e. {@code d²}.
     *   <li>DOT_PRODUCT / MAXIMUM_INNER_PRODUCT — {@code −⟨q,c⟩}.
     *   <li>COSINE — {@code −⟨q,c⟩ / ‖c‖}; the norm matters because centroids of unit vectors are not unit norm.
     * </ul>
     */
    private static float distanceKey(
        VectorSimilarityFunction similarity,
        float[] query,
        float queryNormSq,
        float[] centroid,
        float centroidNorm
    ) {
        float dot = VectorUtil.dotProduct(query, centroid);
        return switch (similarity) {
            case EUCLIDEAN -> queryNormSq - 2 * dot + centroidNorm;
            case MAXIMUM_INNER_PRODUCT -> -dot;
            case DOT_PRODUCT -> throw new IllegalStateException("ClusterANN does not support DOT_PRODUCT; use MAXIMUM_INNER_PRODUCT");
            case COSINE -> -dot / (float) Math.sqrt(centroidNorm);
        };
    }

    /** Drain the heap into ordinals, closest-first. {@code pop} yields farthest first, so the array fills from the back. */
    private static int[] closestFirst(final NeighborQueue nearest) {
        final int[] probes = new int[nearest.size()];
        for (int i = probes.length - 1; i >= 0; i--) {
            probes[i] = nearest.pop();
        }
        return probes;
    }

    /**
     * Drain the heap into ordinals, closest-first, cutting at the knee. {@code pop} yields farthest first, so both
     * arrays fill from the back; the keys ride along because the heap is the only place the curve is.
     *
     * @param acceptedPerCluster accepted vectors assigned to each cluster, or null when nothing is filtered out
     * @param k the number of results the caller wants; read only with {@code acceptedPerCluster}
     */
    private static int[] closestFirstPruned(final NeighborQueue nearest, final int minProbes, final int[] acceptedPerCluster, final int k) {
        final int ranked = nearest.size();
        final int[] probes = new int[ranked];
        final float[] distances = new float[ranked];
        for (int i = ranked - 1; i >= 0; i--) {
            distances[i] = nearest.topScore();
            probes[i] = nearest.pop();
        }

        int keep = CentroidKneePruner.prune(distances, minProbes);
        if (acceptedPerCluster != null && keep < ranked) {
            keep = holdingK(probes, keep, acceptedPerCluster, k);
        }
        return keep == ranked ? probes : Arrays.copyOf(probes, keep);
    }

    /**
     * Extends a cut until the clusters it keeps hold {@code k} accepted vectors between them, or until nothing ranked
     * is left. The knee reads distances alone, so under a filter that sits away from the query the clusters before it
     * can hold too few accepted vectors to answer {@code k}. Counts primary assignments only, which a cluster's SOAR
     * copies can only add to, so the extension never stops short.
     */
    private static int holdingK(final int[] probes, final int keep, final int[] acceptedPerCluster, final int k) {
        long held = 0;
        for (int i = 0; i < keep; i++) {
            held += acceptedPerCluster[probes[i]];
        }
        int extended = keep;
        while (held < k && extended < probes.length) {
            held += acceptedPerCluster[probes[extended++]];
        }
        return extended;
    }
}
