/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
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
import org.opensearch.knn.plugin.stats.ClusterANNQueryValue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;

class CentroidPlannerFilteringTest {

    private static final int DIMENSION = 2;
    private static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;
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
        Clusters clusters = spy(clusters(centroids));
        PlanParams params = new PlanParams(1, centroids.length);
        float[] query = { 0f, 0f };

        // when
        int[] actual = CentroidPlanner.plan(clusters, query, params, null);
        int[] expected = CentroidPlanner.plan(clusters, query, params);

        // then
        assertArrayEquals(expected, actual);
        verify(clusters, never()).ordToCentroid();
    }

    @Test
    void testPlanWithFilter_whenRestrictive_thenPlansEligibleCentroids() throws IOException {
        // given
        float[][] centroids = { { 1f, 0f }, { (float) Math.sqrt(2d), 0f }, { 0.1f, 0f } };
        Clusters clusters = clusters(centroids);
        FixedBitSet acceptedOrds = acceptedOrds(0, CLUSTER_SIZES[0]);
        PlanParams params = new PlanParams(1, centroids.length);
        float[] query = { 0f, 0f };

        // when
        int[] actual = CentroidPlanner.plan(clusters, query, params, acceptDocs(acceptedOrds));

        // then
        assertArrayEquals(new int[] { 0, 1 }, actual);
    }

    @Test
    void testPlanWithFilter_whenAcceptAll_thenMatchesUnfilteredPath() throws IOException {
        // given
        resetStats();
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 3f, 0f } };
        Clusters clusters = spy(clusters(centroids));
        FixedBitSet acceptedOrds = new FixedBitSet(Arrays.stream(CLUSTER_SIZES).sum());
        acceptedOrds.set(0, acceptedOrds.length());
        PlanParams params = new PlanParams(1, centroids.length);
        float[] query = { 0f, 0f };

        // when
        int[] actual = CentroidPlanner.plan(clusters, query, params, acceptDocs(acceptedOrds));
        int[] expected = CentroidPlanner.plan(clusters, query, params);

        // then
        assertArrayEquals(expected, actual);
        verify(clusters, never()).ordToCentroid();
        assertEquals(0L, ClusterANNQueryValue.FILTERED_SEGMENT_SCANS.getValue());
        assertEquals(0L, ClusterANNQueryValue.ELIGIBLE_CLUSTERS.getValue());
    }

    @Test
    void testPlanWithFilter_whenNoClusterMatches_thenPlansNothing() throws IOException {
        // given
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 3f, 0f } };
        Clusters clusters = clusters(centroids);
        FixedBitSet acceptedOrds = new FixedBitSet(Arrays.stream(CLUSTER_SIZES).sum());
        PlanParams params = new PlanParams(1, centroids.length);
        float[] query = { 0f, 0f };

        // when
        int[] actual = CentroidPlanner.plan(clusters, query, params, acceptDocs(acceptedOrds));

        // then
        assertArrayEquals(new int[0], actual);
    }

    @Test
    void testPlanWithFilter_whenMaximumInnerProduct_thenRanksByPlainInnerProduct() throws IOException {
        // given
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 0.1f, 0f } };
        FixedBitSet acceptedOrds = acceptedOrds(0, CLUSTER_SIZES[0]);

        // when
        int[] probes = CentroidPlanner.plan(
            clusters(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, centroids),
            new float[] { 1f, 0f },
            new PlanParams(1, centroids.length),
            acceptDocs(acceptedOrds)
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
        int[] filtered = CentroidPlanner.plan(clusters, query, params, acceptDocs(acceptedOrds));
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
        int[] filtered = CentroidPlanner.plan(clusters, new float[DIMENSION], params, acceptDocs(acceptedOrds));
        int[] unfiltered = CentroidPlanner.plan(clusters, new float[DIMENSION], params);

        // then
        assertArrayEquals(unfiltered, filtered);
        assertEquals(params.maxProbes(), filtered.length);
    }

    @Test
    void testPlanFiltered_whenEligibleClustersExceedBudget_thenCountsEveryEligibleCluster() throws IOException {
        // given
        resetStats();
        float[][] centroids = { { 1f, 0f }, { 2f, 0f }, { 3f, 0f }, { 4f, 0f } };
        int[] clusterSizes = { 1, 1, 1, 1 };
        FixedBitSet acceptedOrds = new FixedBitSet(clusterSizes.length);
        acceptedOrds.set(0, 3);
        PlanParams params = new PlanParams(1, 2);

        // when
        int[] probes = CentroidPlanner.plan(
            clusters(VectorSimilarityFunction.EUCLIDEAN, centroids, clusterSizes),
            new float[] { 0f, 0f },
            params,
            acceptDocs(acceptedOrds)
        );

        // then
        assertArrayEquals(new int[] { 0, 1 }, probes);
        assertEquals(1L, ClusterANNQueryValue.FILTERED_SEGMENT_SCANS.getValue());
        assertEquals(3L, ClusterANNQueryValue.ELIGIBLE_CLUSTERS.getValue());
    }

    @Test
    void testPlanFiltered_whenLastOrdinalAcceptedAtMultipleOf64_thenIncludesItsPrimaryCentroid() throws IOException {
        // given
        float[][] centroids = { { 2f, 0f }, { 1f, 0f }, { 3f, 0f } };
        int[] ordToCentroid = new int[128];
        ordToCentroid[ordToCentroid.length - 1] = 2;
        int[] clusterSizes = { 127, 0, 1 };
        FixedBitSet acceptedOrds = new FixedBitSet(ordToCentroid.length);
        acceptedOrds.set(ordToCentroid.length - 1);

        // when
        int[] probes = CentroidPlanner.plan(
            clusters(VectorSimilarityFunction.EUCLIDEAN, centroids, clusterSizes, ordToCentroid, open("clap", 3000)),
            new float[] { 0f, 0f },
            new PlanParams(1, centroids.length),
            acceptDocs(acceptedOrds)
        );

        // then
        assertArrayEquals(new int[] { 2 }, probes);
    }

    @Test
    void testPlanFiltered_thenGetsEligibilityFromPrimaryRegion1Assignments() throws IOException {
        // given
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

        // when
        int[] probes = CentroidPlanner.plan(
            clusters(VectorSimilarityFunction.EUCLIDEAN, centroids, new int[] { 7, 6, 6 }, ordToCentroid, open("clap", 3000)),
            new float[] { 0f, 0f },
            new PlanParams(1, centroids.length),
            acceptDocs(acceptedOrds)
        );

        // then
        assertArrayEquals(new int[] { 2, 0, 1 }, probes);
    }

    @Test
    void testPlanFiltered_whenVectorFieldIsSparse_thenMapsAcceptedDocumentsToVectorOrdinals() throws IOException {
        // given
        float[][] centroids = { { 3f, 0f }, { 2f, 0f }, { 1f, 0f } };
        int[] ordToCentroid = { 0, 2, 1 };
        int[] vectorDocs = { 1, 4, 7 };
        FixedBitSet acceptedDocs = new FixedBitSet(8);
        acceptedDocs.set(4);

        // when
        int[] probes = CentroidPlanner.plan(
            sparseClusters(centroids, ordToCentroid, vectorDocs, acceptedDocs.length()),
            new float[] { 0f, 0f },
            new PlanParams(1, centroids.length),
            acceptDocs(acceptedDocs)
        );

        // then
        assertArrayEquals(new int[] { 2 }, probes);
    }

    @Test
    void testPlanFiltered_whenMatchIsOnlyASecondary_thenDoesNotMakeThatCentroidEligible() throws IOException {
        // given
        float[][] centroids = { { 2f, 0f }, { 3f, 0f }, { 1f, 0f } };
        int[] ordToCentroid = new int[19];
        FixedBitSet acceptedOrds = new FixedBitSet(ordToCentroid.length);
        acceptedOrds.set(0);

        // when
        int[] probes = CentroidPlanner.plan(
            clusters(VectorSimilarityFunction.EUCLIDEAN, centroids, new int[] { 10, 6, 3 }, ordToCentroid, clapWithSecondary()),
            new float[] { 0f, 0f },
            new PlanParams(1, centroids.length),
            acceptDocs(acceptedOrds)
        );

        // then
        assertArrayEquals(new int[] { 0 }, probes);
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

    private static void resetStats() {
        for (ClusterANNQueryValue value : ClusterANNQueryValue.values()) {
            value.set(0);
        }
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
            fieldMeta(similarity, ordToCentroid.length, clac.length(), clac.offsets().clacCentroidsOffset(), clusterSizes)
        );
    }

    private Clusters sparseClusters(float[][] centroids, int[] ordToCentroid, int[] vectorDocs, int maxDoc) throws IOException {
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
        return new Clusters(
            directory.openInput("sparse-clap", IOContext.DEFAULT),
            clac.input(),
            null,
            fieldMeta(
                VectorSimilarityFunction.EUCLIDEAN,
                ordToCentroid.length,
                clac.length(),
                clac.offsets().clacCentroidsOffset(),
                new int[] { 1, 1, 1 },
                ordToDoc
            )
        );
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
}
