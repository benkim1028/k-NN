/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.codecs.lucene90.IndexedDISI;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.IOSupplier;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.format.ClusterANNFieldMeta;
import org.opensearch.knn.clusterann.format.ClusterANNFormatConstants;
import org.opensearch.knn.clusterann.format.rotation.RotationFormats;
import org.opensearch.knn.clusterann.read.ClusterANNFieldMetaEncoder;
import org.opensearch.knn.clusterann.read.Clusters;
import org.opensearch.knn.clusterann.read.block.scalar.JavaAdcScoreComputer;
import org.opensearch.knn.clusterann.write.CentroidsWriter;
import org.opensearch.knn.clusterann.write.CentroidsWriter.CentroidData;
import org.opensearch.knn.clusterann.write.CentroidsWriter.CentroidOffsets;
import org.opensearch.knn.plugin.stats.ClusterANNQueryValue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;

class CentroidPlannerFilteringTest {

    private static final int DIMENSION = 2;
    private static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;
    private static final int[] CLUSTER_SIZES = { 4, 4, 4 };
    /** Too small to cover the accepted vectors wherever the probe cap matters, so those tests still see the cap. */
    private static final int K = 1;
    /** Wide enough for the probe budget to bind: root 20, so 60 probes unfiltered and more under a filter. */
    private static final int WIDE_CLUSTERS = 400;
    private static final int WIDE_CLUSTER_SIZE = 25;
    private static final int WIDE_VECTORS = WIDE_CLUSTERS * WIDE_CLUSTER_SIZE;
    private static final IOSupplier<KnnVectorValues.DocIndexIterator> DENSE_FIELD = () -> {
        throw new AssertionError("a dense field maps documents to ordinals directly and must not read its vector docs");
    };

    private final List<Directory> directories = new ArrayList<>();

    @AfterEach
    void tearDown() throws IOException {
        for (Directory directory : directories) {
            directory.close();
        }
    }

    @Test
    void testPlanWithFilter_whenNull_thenMatchesUnfilteredPath() throws IOException {
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 3f, 0f } };
        Clusters clusters = spy(clusters(centroids));
        PlanParams params = new PlanParams(1, centroids.length);
        float[] query = { 0f, 0f };

        int[] actual = CentroidPlanner.plan(clusters, query, params, K, null, DENSE_FIELD);
        int[] expected = CentroidPlanner.plan(clusters, query, params);

        assertArrayEquals(expected, actual);
        verify(clusters, never()).ordToCentroid();
    }

    @Test
    void testPlanWithFilter_whenRestrictive_thenPlansEligibleCentroids() throws IOException {
        float[][] centroids = { { 1f, 0f }, { (float) Math.sqrt(2d), 0f }, { 0.1f, 0f } };
        Clusters clusters = clusters(centroids);
        FixedBitSet acceptedOrds = acceptedOrds(0, CLUSTER_SIZES[0]);
        PlanParams params = new PlanParams(1, centroids.length);
        float[] query = { 0f, 0f };

        int[] actual = CentroidPlanner.plan(clusters, query, params, K, acceptDocs(acceptedOrds), DENSE_FIELD);

        assertArrayEquals(new int[] { 0, 1 }, actual);
    }

    @Test
    void testPlanWithFilter_whenBitsAcceptAll_thenTakesFilteredPathWithUnfilteredProbes() throws IOException {
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 3f, 0f } };
        Clusters clusters = spy(clusters(centroids));
        FixedBitSet acceptedOrds = new FixedBitSet(Arrays.stream(CLUSTER_SIZES).sum());
        acceptedOrds.set(0, acceptedOrds.length());
        PlanParams params = new PlanParams(1, centroids.length);
        float[] query = { 0f, 0f };

        int[] actual = CentroidPlanner.plan(clusters, query, params, K, acceptDocs(acceptedOrds), DENSE_FIELD);
        int[] expected = CentroidPlanner.plan(clusters, query, params);

        assertArrayEquals(expected, actual);
        // Only null bits mean match-all; a filter whose bits happen to accept everything is still a filter.
        verify(clusters).ordToCentroid();
    }

    @Test
    void testPlanWithFilter_whenBitsAreNull_thenMatchesUnfilteredPath() throws IOException {
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 3f, 0f } };
        Clusters clusters = spy(clusters(centroids));
        PlanParams params = new PlanParams(1, centroids.length);
        float[] query = { 0f, 0f };

        int[] actual = CentroidPlanner.plan(
            clusters,
            query,
            params,
            K,
            AcceptDocs.fromLiveDocs(null, Arrays.stream(CLUSTER_SIZES).sum()),
            DENSE_FIELD
        );
        int[] expected = CentroidPlanner.plan(clusters, query, params);

        assertArrayEquals(expected, actual);
        verify(clusters, never()).ordToCentroid();
    }

    @Test
    void testPlanWithFilter_whenFieldHasNoClusters_thenPlansNothing() throws IOException {
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 3f, 0f } };
        Clusters clusters = spy(clusters(centroids));
        doReturn(0).when(clusters).numClusters();

        int[] probes = CentroidPlanner.plan(
            clusters,
            new float[] { 0f, 0f },
            new PlanParams(1, centroids.length),
            K,
            acceptDocs(acceptedOrds(0)),
            DENSE_FIELD
        );

        assertArrayEquals(new int[0], probes);
        verify(clusters, never()).ordToCentroid();
        verify(clusters, never()).centroids();
    }

    @Test
    void testPlanWithFilter_whenNoClusterMatches_thenPlansNothing() throws IOException {
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 3f, 0f } };
        Clusters clusters = clusters(centroids);
        FixedBitSet acceptedOrds = new FixedBitSet(Arrays.stream(CLUSTER_SIZES).sum());
        PlanParams params = new PlanParams(1, centroids.length);
        float[] query = { 0f, 0f };

        int[] actual = CentroidPlanner.plan(clusters, query, params, K, acceptDocs(acceptedOrds), DENSE_FIELD);

        assertArrayEquals(new int[0], actual);
    }

    @Test
    void testPlanWithFilter_whenMaximumInnerProduct_thenRanksByPlainInnerProduct() throws IOException {
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 0.1f, 0f } };
        FixedBitSet acceptedOrds = acceptedOrds(0, CLUSTER_SIZES[0]);

        int[] probes = CentroidPlanner.plan(
            clusters(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, centroids),
            new float[] { 1f, 0f },
            new PlanParams(1, centroids.length),
            K,
            acceptDocs(acceptedOrds),
            DENSE_FIELD
        );

        assertArrayEquals(new int[] { 1, 0 }, probes);
    }

    @Test
    void testPlanFiltered_whenCosine_thenMatchesUnfilteredPlainOrder() throws IOException {
        float[][] centroids = { { 5f, 5f }, { 1f, 0f }, { -1f, 0f } };
        Clusters clusters = clusters(VectorSimilarityFunction.COSINE, centroids);
        FixedBitSet acceptedOrds = new FixedBitSet(Arrays.stream(CLUSTER_SIZES).sum());
        acceptedOrds.set(0, acceptedOrds.length());
        PlanParams params = new PlanParams(1, centroids.length);
        float[] query = { 1f, 0f };

        int[] filtered = CentroidPlanner.plan(clusters, query, params, K, acceptDocs(acceptedOrds), DENSE_FIELD);
        int[] unfiltered = CentroidPlanner.plan(clusters, query, params);

        assertArrayEquals(unfiltered, filtered);
        assertArrayEquals(new int[] { 1, 0, 2 }, filtered);
    }

    @Test
    void testPlanFiltered_whenAllClustersMatch_thenUsesUnfilteredMaxProbesAndOrder() throws IOException {
        float[][] centroids = new float[64][DIMENSION];
        int[] clusterSizes = new int[centroids.length];
        for (int ordinal = 0; ordinal < centroids.length; ordinal++) {
            centroids[ordinal][0] = ordinal + 1;
            clusterSizes[ordinal] = 1;
        }
        Clusters clusters = clusters(VectorSimilarityFunction.EUCLIDEAN, centroids, clusterSizes);
        FixedBitSet acceptedOrds = new FixedBitSet(centroids.length);
        acceptedOrds.set(0, acceptedOrds.length());
        PlanParams params = PlanParams.of(centroids.length);

        int[] filtered = CentroidPlanner.plan(clusters, new float[DIMENSION], params, K, acceptDocs(acceptedOrds), DENSE_FIELD);
        int[] unfiltered = CentroidPlanner.plan(clusters, new float[DIMENSION], params);

        assertArrayEquals(unfiltered, filtered);
        assertEquals(params.maxProbes(), filtered.length);
    }

    @Test
    void testPlanFiltered_whenEligibleClustersExceedProbeCap_thenUsesExactlyMaxProbes() throws IOException {
        float[][] centroids = new float[129][DIMENSION];
        int[] clusterSizes = new int[centroids.length];
        for (int ordinal = 0; ordinal < centroids.length; ordinal++) {
            centroids[ordinal][0] = ordinal + 1;
            clusterSizes[ordinal] = 1;
        }
        FixedBitSet acceptedOrds = new FixedBitSet(centroids.length);
        acceptedOrds.set(0, centroids.length - 1);
        PlanParams params = PlanParams.of(centroids.length);

        int[] filtered = CentroidPlanner.plan(
            clusters(VectorSimilarityFunction.EUCLIDEAN, centroids, clusterSizes),
            new float[DIMENSION],
            params,
            K,
            acceptDocs(acceptedOrds),
            DENSE_FIELD
        );

        // s = 128/129, budget (34) exceeds params.maxProbes (33)
        assertEquals(34, filtered.length);
    }

    @Test
    void testPlanFiltered_whenKCoversEveryAcceptedVector_thenPlansEveryEligibleCentroidClosestFirst() throws IOException {
        float[][] centroids = new float[129][DIMENSION];
        int[] clusterSizes = new int[centroids.length];
        for (int ordinal = 0; ordinal < centroids.length; ordinal++) {
            centroids[ordinal][0] = ordinal + 1;
            clusterSizes[ordinal] = 1;
        }
        FixedBitSet acceptedOrds = new FixedBitSet(centroids.length);
        acceptedOrds.set(0, centroids.length - 1);

        int[] filtered = CentroidPlanner.plan(
            clusters(VectorSimilarityFunction.EUCLIDEAN, centroids, clusterSizes),
            new float[DIMENSION],
            PlanParams.of(centroids.length),
            centroids.length - 1,
            acceptDocs(acceptedOrds),
            DENSE_FIELD
        );

        assertArrayEquals(IntStream.range(0, centroids.length - 1).toArray(), filtered);
    }

    /** Filtered plans are not knee-pruned, so a cluster past the knee survives. The curve above has no knee. */
    @Test
    void testPlanFiltered_whenTheCurveHasAKnee_thenStillPlansEveryEligibleCentroid() throws IOException {
        float[][] centroids = new float[129][DIMENSION];
        int[] clusterSizes = new int[centroids.length];
        for (int ordinal = 0; ordinal < centroids.length; ordinal++) {
            centroids[ordinal][0] = ordinal + 1;
            clusterSizes[ordinal] = 1;
        }
        // The last eligible centroid sits far beyond the rest, so the gap into it dwarfs the mean gap and is a knee.
        centroids[centroids.length - 2][0] = 10_000f;
        FixedBitSet acceptedOrds = new FixedBitSet(centroids.length);
        acceptedOrds.set(0, centroids.length - 1);

        int[] filtered = CentroidPlanner.plan(
            clusters(VectorSimilarityFunction.EUCLIDEAN, centroids, clusterSizes),
            new float[DIMENSION],
            PlanParams.of(centroids.length),
            centroids.length - 1,
            acceptDocs(acceptedOrds),
            DENSE_FIELD
        );

        assertArrayEquals(IntStream.range(0, centroids.length - 1).toArray(), filtered);
    }

    @Test
    void testPlanFiltered_whenKIsOneBelowTheAcceptedVectors_thenUsesExactlyMaxProbes() throws IOException {
        float[][] centroids = new float[129][DIMENSION];
        int[] clusterSizes = new int[centroids.length];
        for (int ordinal = 0; ordinal < centroids.length; ordinal++) {
            centroids[ordinal][0] = ordinal + 1;
            clusterSizes[ordinal] = 1;
        }
        FixedBitSet acceptedOrds = new FixedBitSet(centroids.length);
        acceptedOrds.set(0, centroids.length - 1);
        PlanParams params = PlanParams.of(centroids.length);

        int[] filtered = CentroidPlanner.plan(
            clusters(VectorSimilarityFunction.EUCLIDEAN, centroids, clusterSizes),
            new float[DIMENSION],
            params,
            centroids.length - 2,
            acceptDocs(acceptedOrds),
            DENSE_FIELD
        );

        // s = 128/129, m = 3.005, budget = ceil(3.005 * 11) = 34; exceeds params.maxProbes (33)
        assertArrayEquals(IntStream.range(0, 34).toArray(), filtered);
    }

    @Test
    void testPlanFiltered_whenSparseKCoversAcceptedVectorsButNotAcceptedDocuments_thenPlansEveryEligibleCentroid() throws IOException {
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 3f, 0f } };
        int[] ordToCentroid = { 0, 1, 2 };
        int[] vectorDocs = { 2, 5, 8 };
        FixedBitSet acceptedDocs = new FixedBitSet(11);
        // Two accepted vector documents, in clusters 0 and 1, and two accepted documents without a vector.
        acceptedDocs.set(0);
        acceptedDocs.set(2);
        acceptedDocs.set(5);
        acceptedDocs.set(6);

        SparseField sparse = sparseField(centroids, ordToCentroid, vectorDocs, acceptedDocs.length());

        int[] probes = CentroidPlanner.plan(
            sparse.clusters(),
            new float[] { 0f, 0f },
            new PlanParams(1, 1),
            2,
            acceptDocs(acceptedDocs),
            sparse.vectorDocs()
        );

        assertArrayEquals(new int[] { 0, 1 }, probes);
    }

    @Test
    void testPlanFiltered_whenSparseKIsBelowTheAcceptedVectors_thenUsesExactlyMaxProbes() throws IOException {
        SparseField sparse = wideSparseField();
        FixedBitSet acceptedDocs = new FixedBitSet(2 * WIDE_VECTORS);
        for (int cluster = 0; cluster < 200; cluster++) {
            acceptedDocs.set(2 * cluster * WIDE_CLUSTER_SIZE);
            acceptedDocs.set(2 * cluster + 1);
        }

        int[] probes = CentroidPlanner.plan(
            sparse.clusters(),
            new float[DIMENSION],
            PlanParams.of(WIDE_CLUSTERS),
            199,
            acceptDocs(acceptedDocs),
            sparse.vectorDocs()
        );

        // s = 200/10000 = 0.02, m(0.02) = 8.37, ceil(8.37 * 20) = 168
        assertArrayEquals(IntStream.range(0, 168).toArray(), probes);
    }

    @Test
    void testPlanFiltered_whenDense_thenBudgetsByTheExactAcceptedVectorCount() throws IOException {
        FixedBitSet acceptedOrds = new FixedBitSet(WIDE_VECTORS);
        for (int ordinal = 0; ordinal < WIDE_VECTORS; ordinal += 10) {
            acceptedOrds.set(ordinal);
        }

        int[] probes = CentroidPlanner.plan(
            wideDenseField(),
            new float[DIMENSION],
            PlanParams.of(WIDE_CLUSTERS),
            K,
            acceptDocs(acceptedOrds),
            DENSE_FIELD
        );

        // s = 1000/10000 = 0.1, m(0.1) = 5.31, ceil(5.31 * 20) = 107
        assertArrayEquals(IntStream.range(0, 107).toArray(), probes);

        // Halve the accepted set (simulates deleted docs): s = 500/10000 = 0.05, ceil(6.43 * 20) = 129
        FixedBitSet fewerAccepted = new FixedBitSet(WIDE_VECTORS);
        for (int ordinal = 0; ordinal < WIDE_VECTORS; ordinal += 20) {
            fewerAccepted.set(ordinal);
        }
        int[] fewerProbes = CentroidPlanner.plan(
            wideDenseField(),
            new float[DIMENSION],
            PlanParams.of(WIDE_CLUSTERS),
            K,
            acceptDocs(fewerAccepted),
            DENSE_FIELD
        );
        assertArrayEquals(IntStream.range(0, 129).toArray(), fewerProbes);
    }

    @Test
    void testPlanWithFilter_whenUnfiltered_thenCountsNoFilteredScan() throws IOException {
        resetStats();
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 3f, 0f } };
        Clusters clusters = clusters(centroids);
        PlanParams params = new PlanParams(1, centroids.length);

        CentroidPlanner.plan(clusters, new float[DIMENSION], params, K, null, DENSE_FIELD);
        CentroidPlanner.plan(clusters, new float[DIMENSION], params, K, AcceptDocs.fromLiveDocs(null, 12), DENSE_FIELD);

        assertEquals(0L, ClusterANNQueryValue.FILTERED_SEGMENT_SCANS.getValue());
        assertEquals(0L, ClusterANNQueryValue.ELIGIBLE_CLUSTERS.getValue());
    }

    @Test
    void testPlanFiltered_whenEligibleClustersExceedTheProbes_thenCountsEveryEligibleCluster() throws IOException {
        resetStats();
        float[][] centroids = new float[129][DIMENSION];
        int[] clusterSizes = new int[centroids.length];
        for (int ordinal = 0; ordinal < centroids.length; ordinal++) {
            centroids[ordinal][0] = ordinal + 1;
            clusterSizes[ordinal] = 1;
        }
        FixedBitSet acceptedOrds = new FixedBitSet(centroids.length);
        acceptedOrds.set(0, centroids.length - 1);

        int[] probes = CentroidPlanner.plan(
            clusters(VectorSimilarityFunction.EUCLIDEAN, centroids, clusterSizes),
            new float[DIMENSION],
            PlanParams.of(centroids.length),
            K,
            acceptDocs(acceptedOrds),
            DENSE_FIELD
        );

        assertEquals(34, probes.length);
        assertEquals(1L, ClusterANNQueryValue.FILTERED_SEGMENT_SCANS.getValue());
        assertEquals(128L, ClusterANNQueryValue.ELIGIBLE_CLUSTERS.getValue());
    }

    @Test
    void testPlanFiltered_whenNoClusterMatches_thenCountsTheScanWithNoEligibleCluster() throws IOException {
        resetStats();
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 3f, 0f } };
        FixedBitSet acceptedOrds = new FixedBitSet(Arrays.stream(CLUSTER_SIZES).sum());

        int[] probes = CentroidPlanner.plan(
            clusters(centroids),
            new float[DIMENSION],
            new PlanParams(1, centroids.length),
            K,
            acceptDocs(acceptedOrds),
            DENSE_FIELD
        );

        assertEquals(0, probes.length);
        assertEquals(1L, ClusterANNQueryValue.FILTERED_SEGMENT_SCANS.getValue());
        assertEquals(0L, ClusterANNQueryValue.ELIGIBLE_CLUSTERS.getValue());
    }

    @Test
    void testPlanFiltered_whenLastOrdinalAcceptedAtMultipleOf64_thenIncludesItsPrimaryCentroid() throws IOException {
        float[][] centroids = { { 2f, 0f }, { 1f, 0f }, { 3f, 0f } };
        int[] ordToCentroid = new int[128];
        ordToCentroid[ordToCentroid.length - 1] = 2;
        int[] clusterSizes = { 127, 0, 1 };
        FixedBitSet acceptedOrds = new FixedBitSet(ordToCentroid.length);
        acceptedOrds.set(ordToCentroid.length - 1);

        int[] probes = CentroidPlanner.plan(
            clusters(VectorSimilarityFunction.EUCLIDEAN, centroids, clusterSizes, ordToCentroid, open("clap", 3000)),
            new float[] { 0f, 0f },
            new PlanParams(1, centroids.length),
            K,
            acceptDocs(acceptedOrds),
            DENSE_FIELD
        );

        assertArrayEquals(new int[] { 2 }, probes);
    }

    @Test
    void testPlanFiltered_thenGetsEligibilityFromPrimaryRegion1Assignments() throws IOException {
        float[][] centroids = { { 2f, 0f }, { 3f, 0f }, { 1f, 0f } };
        int[] ordToCentroid = { 0, 1, 2, 1, 0, 2, 2, 1, 0, 2, 1, 0, 0, 1, 2, 2, 1, 0, 1 };
        FixedBitSet acceptedOrds = new FixedBitSet(ordToCentroid.length);
        acceptedOrds.set(1);
        acceptedOrds.set(2);
        acceptedOrds.set(3);
        acceptedOrds.set(4);
        acceptedOrds.set(6);
        acceptedOrds.set(12);
        acceptedOrds.set(14);

        int[] probes = CentroidPlanner.plan(
            clusters(VectorSimilarityFunction.EUCLIDEAN, centroids, new int[] { 7, 6, 6 }, ordToCentroid, open("clap", 3000)),
            new float[] { 0f, 0f },
            new PlanParams(1, centroids.length),
            K,
            acceptDocs(acceptedOrds),
            DENSE_FIELD
        );

        assertArrayEquals(new int[] { 2, 0, 1 }, probes);
    }

    @Test
    void testPlanFiltered_whenVectorFieldIsSparse_thenMapsAcceptedDocumentsToVectorOrdinals() throws IOException {
        float[][] centroids = { { 3f, 0f }, { 2f, 0f }, { 1f, 0f } };
        int[] ordToCentroid = { 0, 2, 1 };
        int[] vectorDocs = { 1, 4, 7 };
        FixedBitSet acceptedDocs = new FixedBitSet(8);
        acceptedDocs.set(4);

        SparseField sparse = sparseField(centroids, ordToCentroid, vectorDocs, acceptedDocs.length());

        int[] probes = CentroidPlanner.plan(
            sparse.clusters(),
            new float[] { 0f, 0f },
            new PlanParams(1, centroids.length),
            K,
            acceptDocs(acceptedDocs),
            sparse.vectorDocs()
        );

        assertArrayEquals(new int[] { 2 }, probes);
    }

    @Test
    void testPlanFiltered_whenAcceptedDocumentsHaveNoVector_thenMarksOnlyClustersOfAcceptedVectorDocuments() throws IOException {
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 3f, 0f } };
        int[] ordToCentroid = { 0, 1, 2 };
        int[] vectorDocs = { 2, 5, 8 };
        FixedBitSet acceptedDocs = new FixedBitSet(11);
        // Before the first vector doc, between vector docs (one past a vector doc, one just before the next), on the
        // last vector doc, and after the last vector doc.
        acceptedDocs.set(0);
        acceptedDocs.set(3);
        acceptedDocs.set(4);
        acceptedDocs.set(8);
        acceptedDocs.set(9);
        acceptedDocs.set(10);

        SparseField sparse = sparseField(centroids, ordToCentroid, vectorDocs, acceptedDocs.length());

        int[] probes = CentroidPlanner.plan(
            sparse.clusters(),
            new float[] { 0f, 0f },
            new PlanParams(1, centroids.length),
            K,
            acceptDocs(acceptedDocs),
            sparse.vectorDocs()
        );

        assertArrayEquals(new int[] { 2 }, probes);
    }

    @Test
    void testPlanFiltered_whenSparseAcceptedDocumentsInterleaveWithVectorDocuments_thenMarksEachMatchedCluster() throws IOException {
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 3f, 0f } };
        int[] ordToCentroid = { 0, 1, 2 };
        int[] vectorDocs = { 2, 5, 8 };
        FixedBitSet acceptedDocs = new FixedBitSet(11);
        acceptedDocs.set(1);
        acceptedDocs.set(2);
        acceptedDocs.set(6);
        acceptedDocs.set(8);

        SparseField sparse = sparseField(centroids, ordToCentroid, vectorDocs, acceptedDocs.length());

        int[] probes = CentroidPlanner.plan(
            sparse.clusters(),
            new float[] { 0f, 0f },
            new PlanParams(1, centroids.length),
            K,
            acceptDocs(acceptedDocs),
            sparse.vectorDocs()
        );

        assertArrayEquals(new int[] { 0, 2 }, probes);
    }

    @Test
    void testPlanFiltered_whenMatchIsOnlyASecondary_thenDoesNotMakeThatCentroidEligible() throws IOException {
        float[][] centroids = { { 2f, 0f }, { 3f, 0f }, { 1f, 0f } };
        int[] ordToCentroid = new int[19];
        FixedBitSet acceptedOrds = new FixedBitSet(ordToCentroid.length);
        acceptedOrds.set(0);
        Clusters clusters = spy(
            clusters(VectorSimilarityFunction.EUCLIDEAN, centroids, new int[] { 10, 6, 3 }, ordToCentroid, clapWithSecondary())
        );

        int[] probes = CentroidPlanner.plan(
            clusters,
            new float[] { 0f, 0f },
            new PlanParams(1, centroids.length),
            K,
            acceptDocs(acceptedOrds),
            DENSE_FIELD
        );

        assertArrayEquals(new int[] { 0 }, probes);
        verify(clusters).ordToCentroid();
        verify(clusters, never()).get(anyInt());
    }

    /** {@link #WIDE_CLUSTERS} clusters of {@link #WIDE_CLUSTER_SIZE}; centroid {@code c} sits at {@code (c + 1, 0)}. */
    private Clusters wideDenseField() throws IOException {
        int[] clusterSizes = new int[WIDE_CLUSTERS];
        Arrays.fill(clusterSizes, WIDE_CLUSTER_SIZE);
        return clusters(VectorSimilarityFunction.EUCLIDEAN, wideCentroids(), clusterSizes);
    }

    /** The same clusters on a sparse field: only the even documents have a vector. */
    private SparseField wideSparseField() throws IOException {
        int[] ordToCentroid = new int[WIDE_VECTORS];
        int[] vectorDocs = new int[WIDE_VECTORS];
        for (int ordinal = 0; ordinal < WIDE_VECTORS; ordinal++) {
            ordToCentroid[ordinal] = ordinal / WIDE_CLUSTER_SIZE;
            vectorDocs[ordinal] = 2 * ordinal;
        }
        return sparseField(wideCentroids(), ordToCentroid, vectorDocs, 2 * WIDE_VECTORS);
    }

    private static float[][] wideCentroids() {
        float[][] centroids = new float[WIDE_CLUSTERS][DIMENSION];
        for (int ordinal = 0; ordinal < WIDE_CLUSTERS; ordinal++) {
            centroids[ordinal][0] = ordinal + 1;
        }
        return centroids;
    }

    private static FixedBitSet acceptedOrds(int... ordinals) {
        FixedBitSet acceptedOrds = new FixedBitSet(Arrays.stream(CLUSTER_SIZES).sum());
        for (int ordinal : ordinals) {
            acceptedOrds.set(ordinal);
        }
        return acceptedOrds;
    }

    private static AcceptDocs acceptDocs(FixedBitSet acceptedDocs) {
        return AcceptDocs.fromLiveDocs(acceptedDocs, acceptedDocs.length());
    }

    private Clusters clusters(float[][] centroids) throws IOException {
        return clusters(VectorSimilarityFunction.EUCLIDEAN, centroids, CLUSTER_SIZES);
    }

    private Clusters clusters(VectorSimilarityFunction similarity, float[][] centroids) throws IOException {
        return clusters(similarity, centroids, CLUSTER_SIZES);
    }

    private Clusters clusters(VectorSimilarityFunction similarity, float[][] centroids, int[] clusterSizes) throws IOException {
        int[] ordToCentroid = new int[Arrays.stream(clusterSizes).sum()];
        int vectorOrdinal = 0;
        for (int centroidOrdinal = 0; centroidOrdinal < clusterSizes.length; centroidOrdinal++) {
            Arrays.fill(ordToCentroid, vectorOrdinal, vectorOrdinal + clusterSizes[centroidOrdinal], centroidOrdinal);
            vectorOrdinal += clusterSizes[centroidOrdinal];
        }
        return clusters(similarity, centroids, clusterSizes, ordToCentroid, open("clap", 3000));
    }

    private Clusters clusters(
        VectorSimilarityFunction similarity,
        float[][] centroids,
        int[] clusterSizes,
        int[] ordToCentroid,
        IndexInput postings
    ) throws IOException {
        ClacFixture clac = clac(centroids, ordToCentroid);
        return new Clusters(
            postings,
            clac.input(),
            null,
            fieldMeta(similarity, ordToCentroid.length, clac.length(), clac.offsets().clacCentroidsOffset(), clusterSizes),
            JavaAdcScoreComputer.INSTANCE
        );
    }

    private SparseField sparseField(float[][] centroids, int[] ordToCentroid, int[] vectorDocs, int maxDoc) throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        DocsWithFieldSet docsWithField = new DocsWithFieldSet();
        for (int doc : vectorDocs) {
            docsWithField.add(doc);
        }
        try (
            IndexOutput meta = directory.createOutput("sparse-meta", IOContext.DEFAULT);
            IndexOutput postings = directory.createOutput("sparse-clap", IOContext.DEFAULT)
        ) {
            OrdToDocDISIReaderConfiguration.writeStoredMeta(
                DIRECT_MONOTONIC_BLOCK_SHIFT,
                meta,
                postings,
                vectorDocs.length,
                maxDoc,
                docsWithField
            );
        }
        final OrdToDocDISIReaderConfiguration ordToDoc;
        try (IndexInput meta = directory.openInput("sparse-meta", IOContext.DEFAULT)) {
            ordToDoc = OrdToDocDISIReaderConfiguration.fromStoredMeta(meta, vectorDocs.length);
        }
        ClacFixture clac = clac(centroids, ordToCentroid);
        IndexInput postings = directory.openInput("sparse-clap", IOContext.DEFAULT);
        Clusters clusters = new Clusters(
            postings,
            clac.input(),
            null,
            fieldMeta(
                VectorSimilarityFunction.EUCLIDEAN,
                ordToCentroid.length,
                clac.length(),
                clac.offsets().clacCentroidsOffset(),
                clusterSizes(ordToCentroid, centroids.length),
                ordToDoc
            ),
            JavaAdcScoreComputer.INSTANCE
        );
        return new SparseField(clusters, () -> IndexedDISI.asDocIndexIterator(ordToDoc.getIndexedDISI(postings)));
    }

    private static int[] clusterSizes(int[] ordToCentroid, int numClusters) {
        int[] clusterSizes = new int[numClusters];
        for (int centroid : ordToCentroid) {
            clusterSizes[centroid]++;
        }
        return clusterSizes;
    }

    private IndexInput clapWithSecondary() throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        try (IndexOutput out = directory.createOutput("clap-secondary", IOContext.DEFAULT)) {
            writePosting(out, new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, -1, 1000);
            writePosting(out, new int[] { 10, 11, 12, 13, 14, 15 }, -1, 1000);
            writePosting(out, new int[] { 0, 17, 18 }, 0, 1000);
        }
        return directory.openInput("clap-secondary", IOContext.DEFAULT);
    }

    private static void writePosting(IndexOutput out, int[] ordinals, int secondaryPosition, int length) throws IOException {
        final long start = out.getFilePointer();
        for (int ordinal : ordinals) {
            out.writeInt(ordinal);
        }
        final FixedBitSet secondaries = new FixedBitSet(ordinals.length);
        if (secondaryPosition >= 0) {
            secondaries.set(secondaryPosition);
        }
        for (long bits : secondaries.getBits()) {
            out.writeLong(bits);
        }
        for (int ignored : ordinals) {
            out.writeInt(0);
        }
        final int padding = Math.toIntExact(length - (out.getFilePointer() - start));
        out.writeBytes(new byte[padding], 0, padding);
    }

    private ClacFixture clac(float[][] centroids, int[] ordToCentroid) throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        CentroidOffsets offsets;
        try (IndexOutput out = directory.createOutput("clac", IOContext.DEFAULT)) {
            offsets = CentroidsWriter.write(
                out,
                new CentroidData(ordToCentroid, centroids),
                RotationFormats.create(ClusterANNFormatConstants.ROTATION_NONE, DIMENSION)
            );
        }
        IndexInput input = directory.openInput("clac", IOContext.DEFAULT);
        return new ClacFixture(input, input.length(), offsets);
    }

    private static ClusterANNFieldMeta fieldMeta(
        VectorSimilarityFunction similarity,
        int vectorCount,
        long clacLength,
        long clacCentroidsOffset,
        int[] clusterSizes
    ) throws IOException {
        return fieldMeta(
            similarity,
            vectorCount,
            clacLength,
            clacCentroidsOffset,
            clusterSizes,
            ClusterANNFieldMetaEncoder.denseOrdToDoc(vectorCount)
        );
    }

    private static ClusterANNFieldMeta fieldMeta(
        VectorSimilarityFunction similarity,
        int vectorCount,
        long clacLength,
        long clacCentroidsOffset,
        int[] clusterSizes,
        OrdToDocDISIReaderConfiguration ordToDoc
    ) {
        int centroidCount = clusterSizes.length;
        return new ClusterANNFieldMeta(
            32,
            DIMENSION,
            vectorCount,
            centroidCount,
            similarity,
            1,
            ClusterANNFormatConstants.ROTATION_NONE,
            ClusterANNFormatConstants.QUANTIZER_OPTIMIZED_SQ,
            new byte[0],
            0L,
            clacLength,
            clacCentroidsOffset,
            -1L,
            0L,
            3000L,
            new long[centroidCount],
            new int[centroidCount],
            clusterSizes,
            -1L,
            -1L,
            ordToDoc
        );
    }

    private IndexInput open(String name, int bytes) throws IOException {
        Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        try (IndexOutput out = directory.createOutput(name, IOContext.DEFAULT)) {
            out.writeBytes(new byte[bytes], 0, bytes);
        }
        return directory.openInput(name, IOContext.DEFAULT);
    }

    private record ClacFixture(IndexInput input, long length, CentroidOffsets offsets) {
    }

    private record SparseField(Clusters clusters, IOSupplier<KnnVectorValues.DocIndexIterator> vectorDocs) {
    }

    private static void resetStats() {
        for (ClusterANNQueryValue value : ClusterANNQueryValue.values()) {
            value.set(0);
        }
    }
}
