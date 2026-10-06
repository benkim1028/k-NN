/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.FixedBitSet;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.format.ClusterANNFieldMeta;
import org.opensearch.knn.clusterann.format.ClusterANNFormatConstants;
import org.opensearch.knn.clusterann.format.rotation.RotationFormats;
import org.opensearch.knn.clusterann.read.ClusterANNFieldMetaEncoder;
import org.opensearch.knn.clusterann.read.Clusters;
import org.opensearch.knn.clusterann.write.CentroidsWriter;
import org.opensearch.knn.clusterann.write.CentroidsWriter.CentroidData;
import org.opensearch.knn.clusterann.write.CentroidsWriter.CentroidOffsets;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;

class CentroidPlannerFilteringTest {

    private static final int DIMENSION = 2;
    private static final int[] CLUSTER_SIZES = { 4, 4, 4 };

    private final List<Directory> directories = new ArrayList<>();

    @AfterEach
    void tearDown() throws IOException {
        for (Directory directory : directories) {
            directory.close();
        }
    }

    @Test
    void testPlanWithFilter_whenNull_thenMatchesUnfilteredPath() throws IOException {
        // given
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 3f, 0f } };
        Clusters clusters = clusters(centroids);
        PlanParams params = new PlanParams(1, centroids.length);
        float[] query = { 0f, 0f };

        // when
        int[] actual = CentroidPlanner.plan(clusters, query, params, null);
        int[] expected = CentroidPlanner.plan(clusters, query, params);

        // then
        assertArrayEquals(expected, actual);
    }

    @Test
    void testPlanWithFilter_whenRestrictive_thenMatchesTwoStepPath() throws IOException {
        // given
        float[][] centroids = { { 1f, 0f }, { (float) Math.sqrt(2d), 0f }, { 0.1f, 0f } };
        FixedBitSet acceptedCentroids = acceptedCentroids(0, 1);
        int[] matchCounts = { 1, 1, 0 };
        Clusters clusters = clustersWithMatches(clusters(centroids), acceptedCentroids, matchCounts);
        FixedBitSet acceptedOrds = acceptedOrds(0, CLUSTER_SIZES[0]);
        PlanParams params = new PlanParams(1, centroids.length);
        float[] query = { 0f, 0f };

        // when
        int[] actual = CentroidPlanner.plan(clusters, query, params, acceptedOrds);
        int[] expected = twoStepPlan(clusters, query, params, acceptedOrds);

        // then
        assertArrayEquals(expected, actual);
        assertArrayEquals(new int[] { 0, 1 }, actual);
    }

    @Test
    void testPlanWithFilter_whenAcceptAll_thenMatchesUnfilteredPath() throws IOException {
        // given
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 3f, 0f } };
        Clusters clusters = spy(clusters(centroids));
        FixedBitSet acceptedOrds = new FixedBitSet(Arrays.stream(CLUSTER_SIZES).sum());
        acceptedOrds.set(0, acceptedOrds.length());
        PlanParams params = new PlanParams(1, centroids.length);
        float[] query = { 0f, 0f };

        // when
        int[] actual = CentroidPlanner.plan(clusters, query, params, acceptedOrds);
        int[] expected = CentroidPlanner.plan(clusters, query, params);

        // then
        assertArrayEquals(expected, actual);
        verify(clusters, never()).centroidMatches(acceptedOrds);
    }

    @Test
    void testPlanWithFilter_whenNoClusterMatches_thenMatchesTwoStepPath() throws IOException {
        // given
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 3f, 0f } };
        Clusters clusters = clustersWithMatches(clusters(centroids), new FixedBitSet(centroids.length), new int[centroids.length]);
        FixedBitSet acceptedOrds = new FixedBitSet(Arrays.stream(CLUSTER_SIZES).sum());
        PlanParams params = new PlanParams(1, centroids.length);
        float[] query = { 0f, 0f };

        // when
        int[] actual = CentroidPlanner.plan(clusters, query, params, acceptedOrds);
        int[] expected = twoStepPlan(clusters, query, params, acceptedOrds);

        // then
        assertArrayEquals(expected, actual);
        assertArrayEquals(new int[0], actual);
    }

    @Test
    void testPlanWithFilter_whenMaximumInnerProduct_thenRanksByPlainInnerProduct() throws IOException {
        // given
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 0.1f, 0f } };
        FixedBitSet acceptedCentroids = acceptedCentroids(0, 1);
        int[] matchCounts = { 1, 1, 0 };
        FixedBitSet acceptedOrds = acceptedOrds(0, CLUSTER_SIZES[0]);

        // when
        int[] probes = CentroidPlanner.plan(
            clustersWithMatches(clusters(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, centroids), acceptedCentroids, matchCounts),
            new float[] { 1f, 0f },
            new PlanParams(1, centroids.length),
            acceptedOrds
        );

        // then
        assertArrayEquals(new int[] { 1, 0 }, probes);
    }

    @Test
    void testPlanFiltered_whenCosine_thenMatchesUnfilteredPlainOrder() throws IOException {
        // given
        float[][] centroids = { { 5f, 5f }, { 1f, 0f }, { -1f, 0f } };
        Clusters clusters = clusters(VectorSimilarityFunction.COSINE, centroids);
        FixedBitSet acceptedOrds = new FixedBitSet(Arrays.stream(CLUSTER_SIZES).sum());
        acceptedOrds.set(0, acceptedOrds.length());
        PlanParams params = new PlanParams(1, centroids.length);
        float[] query = { 1f, 0f };

        // when
        int[] filtered = CentroidPlanner.plan(clusters, query, params, acceptedOrds);
        int[] unfiltered = CentroidPlanner.plan(clusters, query, params);

        // then
        assertArrayEquals(unfiltered, filtered);
        assertArrayEquals(new int[] { 1, 0, 2 }, filtered);
    }

    @Test
    void testPlanFiltered_whenAllClustersMatch_thenUsesUnfilteredMaxProbesAndOrder() throws IOException {
        // given
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

        // when
        int[] filtered = CentroidPlanner.plan(clusters, new float[DIMENSION], params, acceptedOrds);
        int[] unfiltered = CentroidPlanner.plan(clusters, new float[DIMENSION], params);

        // then
        assertArrayEquals(unfiltered, filtered);
        assertEquals(params.maxProbes(), filtered.length);
    }

    @Test
    void testPlanFiltered_whenLastCentroidAcceptedAtMultipleOf64_thenIncludesItWithoutThrowing() throws IOException {
        // given
        float[][] centroids = new float[64][DIMENSION];
        int[] clusterSizes = new int[centroids.length];
        Arrays.fill(clusterSizes, 1);
        FixedBitSet acceptedOrds = new FixedBitSet(centroids.length);
        acceptedOrds.set(centroids.length - 1);
        FixedBitSet acceptedCentroids = acceptedCentroids(centroids.length - 1);
        int[] matchCounts = new int[centroids.length];
        matchCounts[centroids.length - 1] = 1;

        // when
        int[] probes = CentroidPlanner.plan(
            clustersWithMatches(clusters(centroids, clusterSizes), acceptedCentroids, matchCounts),
            new float[] { 0f, 0f },
            new PlanParams(1, centroids.length),
            acceptedOrds
        );

        // then
        assertArrayEquals(new int[] { centroids.length - 1 }, probes);
    }

    private static int[] twoStepPlan(Clusters clusters, float[] query, PlanParams params, FixedBitSet acceptedOrds) throws IOException {
        Clusters.CentroidMatches matches = clusters.centroidMatches(acceptedOrds);
        return CentroidPlanner.plan(clusters, query, params, matches.acceptedCentroids(), matches.matchCounts());
    }

    private static Clusters clustersWithMatches(Clusters clusters, FixedBitSet acceptedCentroids, int[] matchCounts) throws IOException {
        Clusters stubbed = spy(clusters);
        doReturn(new Clusters.CentroidMatches(acceptedCentroids, matchCounts)).when(stubbed)
            .centroidMatches(org.mockito.ArgumentMatchers.any());
        return stubbed;
    }

    private static FixedBitSet acceptedCentroids(int... ordinals) {
        int length = Arrays.stream(ordinals).max().orElse(-1) + 1;
        FixedBitSet acceptedCentroids = new FixedBitSet(Math.max(CLUSTER_SIZES.length, length));
        for (int ordinal : ordinals) {
            acceptedCentroids.set(ordinal);
        }
        return acceptedCentroids;
    }

    private static FixedBitSet acceptedOrds(int... ordinals) {
        FixedBitSet acceptedOrds = new FixedBitSet(Arrays.stream(CLUSTER_SIZES).sum());
        for (int ordinal : ordinals) {
            acceptedOrds.set(ordinal);
        }
        return acceptedOrds;
    }

    private Clusters clusters(float[][] centroids) throws IOException {
        return clusters(VectorSimilarityFunction.EUCLIDEAN, centroids, CLUSTER_SIZES);
    }

    private Clusters clusters(float[][] centroids, int[] clusterSizes) throws IOException {
        return clusters(VectorSimilarityFunction.EUCLIDEAN, centroids, clusterSizes);
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
        ClacFixture clac = clac(centroids, ordToCentroid);
        return new Clusters(
            open("clap", 3000),
            clac.input(),
            null,
            fieldMeta(similarity, ordToCentroid.length, clac.length(), clac.offsets().clacCentroidsOffset(), clusterSizes)
        );
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
            ClusterANNFieldMetaEncoder.denseOrdToDoc(vectorCount)
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
}
