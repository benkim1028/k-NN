/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.vectorformat1030;

import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.Mockito;
import org.opensearch.knn.clusterann.format.QuantizationParams;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Fast, isolated unit tests for {@link KNN1030ClusterANNVectorsFormat} (JUnit 5 + Mockito).
 *
 * <p>These do not build a Lucene index; they cover the format's deterministic contract
 * (name and max-dimensions) and its writer/reader factory methods. The factory methods eagerly
 * delegate to Lucene's flat vectors format, which needs a fully-initialized segment state, so
 * they are driven here with Mockito mocks to confirm they fail fast on an empty state rather than
 * silently returning.
 */
class KNN1030ClusterANNVectorsFormatTest {

    private final KNN1030ClusterANNVectorsFormat format = new KNN1030ClusterANNVectorsFormat(KNN1030ClusterANNVectorsFormat.FORMAT_NAME);

    @Test
    void formatName_isStable() {
        assertEquals("KNN1030ClusterANNVectorsFormat", KNN1030ClusterANNVectorsFormat.FORMAT_NAME);
    }

    @Test
    void getName_returnsFormatName() {
        assertEquals(KNN1030ClusterANNVectorsFormat.FORMAT_NAME, format.getName());
    }

    @Test
    void getMaxDimensions_returnsConfiguredLimit_independentOfField() {
        assertEquals(16000, format.getMaxDimensions("float_field"));
        assertEquals(16000, format.getMaxDimensions("some_other_field"));
    }

    @ParameterizedTest(name = "storageBlockBytes {0}")
    @ValueSource(ints = { 0, -1 })
    void constructor_throws_whenStorageBlockBytesIsNotPositive(int storageBlockBytes) {
        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> new KNN1030ClusterANNVectorsFormat(
                KNN1030ClusterANNVectorsFormat.FORMAT_NAME,
                QuantizationParams.DEFAULT,
                storageBlockBytes
            )
        );
        assertEquals("storageBlockBytes must be positive, got: " + storageBlockBytes, e.getMessage());
    }

    @Test
    void fieldsWriter_delegatesToFlatFormat_soRequiresARealSegmentState() {
        SegmentWriteState writeState = Mockito.mock(SegmentWriteState.class);
        // The format immediately calls the flat vectors writer, which reads directory/segment
        // info off the state; an unpopulated mock makes that fail fast.
        assertThrows(Exception.class, () -> format.fieldsWriter(writeState));
    }

    @Test
    void fieldsReader_delegatesToFlatFormat_soRequiresARealSegmentState() {
        SegmentReadState readState = Mockito.mock(SegmentReadState.class);
        assertThrows(Exception.class, () -> format.fieldsReader(readState));
    }
}
