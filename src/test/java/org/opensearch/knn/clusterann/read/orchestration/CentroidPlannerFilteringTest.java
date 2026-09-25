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
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

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
    void testPlanFiltered_thenSkipsZeroMatchCentroidsAndRanksByDensityWeightedDistance() throws IOException {
        // given
        float[][] centroids = { { 1f, 0f }, { (float) Math.sqrt(2d), 0f }, { 0.1f, 0f } };
        FixedBitSet acceptedCentroids = new FixedBitSet(centroids.length);
        acceptedCentroids.set(0);
        acceptedCentroids.set(1);
        int[] matchCounts = { 1, 10, 0 };

        // when
        int[] probes = CentroidPlanner.plan(
            clusters(centroids),
            new float[] { 0f, 0f },
            new PlanParams(1, centroids.length),
            acceptedCentroids,
            matchCounts
        );

        // then
        assertArrayEquals(new int[] { 1, 0 }, probes);
    }

    private Clusters clusters(float[][] centroids) throws IOException {
        int[] ordToCentroid = { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2 };
        ClacFixture clac = clac(centroids, ordToCentroid);
        return new Clusters(
            open("clap", 3000),
            clac.input(),
            null,
            fieldMeta(ordToCentroid.length, clac.length(), clac.offsets().clacCentroidsOffset(), centroids.length)
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

    private static ClusterANNFieldMeta fieldMeta(int vectorCount, long clacLength, long clacCentroidsOffset, int centroidCount)
        throws IOException {
        return new ClusterANNFieldMeta(
            32,
            DIMENSION,
            vectorCount,
            centroidCount,
            VectorSimilarityFunction.EUCLIDEAN,
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
            CLUSTER_SIZES,
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
