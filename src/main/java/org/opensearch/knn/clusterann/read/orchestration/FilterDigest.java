/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.ConjunctionUtils;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.LongValues;
import org.opensearch.knn.clusterann.read.Clusters;
import org.opensearch.knn.clusterann.stats.ClusterANNStat;

import java.io.IOException;
import java.util.List;

/**
 * What a document-space filter leaves of one field's clusters: which clusters hold an accepted vector, and how many
 * accepted vectors each holds (primary assignments only, so SOAR copies are not counted twice).
 *
 * <p>Both planners read it: {@link CentroidPlanner} for one segment, {@link ShardPlanner} for every segment of a shard
 * at once. Computing it walks the filter's accepted documents once.
 *
 * @param eligible clusters holding at least one accepted vector
 * @param acceptedPerCluster accepted vectors per cluster, indexed by centroid ordinal
 * @param accepted accepted vectors in the field
 * @param eligibleClusters clusters holding at least one accepted vector ({@code eligible.cardinality()})
 */
public record FilterDigest(FixedBitSet eligible, int[] acceptedPerCluster, int accepted, int eligibleClusters) {

    /**
     * The digest of {@code acceptDocs} over {@code clusters}, or {@code null} when nothing is filtered out: a null
     * filter, or one with null {@link AcceptDocs#bits()}, which is how Lucene hands a segment without deletions a
     * match-all filter. Records {@link ClusterANNStat#FILTER_ELIGIBLE_CLUSTERS}.
     *
     * @param vectorDocsSupplier a new iterator over the documents that have this vector field, exposing each one's
     *     vector ordinal; called only when the field is sparse
     */
    public static FilterDigest of(
        final Clusters clusters,
        final AcceptDocs acceptDocs,
        final IOSupplier<KnnVectorValues.DocIndexIterator> vectorDocsSupplier
    ) throws IOException {
        if (acceptDocs == null || acceptDocs.bits() == null) {
            return null;
        }
        final int numClusters = clusters.numClusters();
        final LongValues ordToCentroid = clusters.ordToCentroid();
        final FixedBitSet eligible = new FixedBitSet(Math.max(1, numClusters));
        final int[] counts = new int[Math.max(1, numClusters)];
        final DocIdSetIterator acceptedDocs = acceptDocs.iterator();
        int accepted = 0;
        if (clusters.hasDenseVectorValues()) {
            for (int doc = acceptedDocs.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = acceptedDocs.nextDoc()) {
                final int centroid = (int) ordToCentroid.get(doc);
                eligible.set(centroid);
                counts[centroid]++;
                accepted++;
            }
        } else {
            final KnnVectorValues.DocIndexIterator vectorDocs = vectorDocsSupplier.get();
            final DocIdSetIterator acceptedVectorDocs = ConjunctionUtils.intersectIterators(List.of(acceptedDocs, vectorDocs));
            while (acceptedVectorDocs.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                final int centroid = (int) ordToCentroid.get(vectorDocs.index());
                eligible.set(centroid);
                counts[centroid]++;
                accepted++;
            }
        }
        final int eligibleClusters = eligible.cardinality();
        // True count of clusters containing at least one accepted doc (the filter's eligible set), recorded here
        // where it is known; distinct from the planned probe count the reader records as PROBES_PLANNED.
        ClusterANNStat.FILTER_ELIGIBLE_CLUSTERS.record(eligibleClusters);
        return new FilterDigest(eligible, counts, accepted, eligibleClusters);
    }

    /** Fraction of the field's vectors the filter accepts. */
    public double selectivity(final int numVectors) {
        return (double) accepted / Math.max(1, numVectors);
    }
}
