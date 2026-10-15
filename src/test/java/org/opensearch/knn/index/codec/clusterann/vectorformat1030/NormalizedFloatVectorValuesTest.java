/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.vectorformat1030;

import org.apache.lucene.index.FloatVectorValues;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** Verifies {@link NormalizedFloatVectorValues} hands out unit vectors without touching the vectors it views. */
class NormalizedFloatVectorValuesTest {

    private static final int DIMENSION = 2;
    private static final float DELTA = 1e-6f;

    @Test
    void vectorValue_isTheSourceVectorAtUnitLength() throws IOException {
        final FloatVectorValues view = new NormalizedFloatVectorValues(source(new float[] { 3f, 4f }, new float[] { 0f, 2f }));

        assertArrayEquals(new float[] { 0.6f, 0.8f }, view.vectorValue(0), DELTA);
        assertArrayEquals(new float[] { 0f, 1f }, view.vectorValue(1), DELTA);
    }

    @Test
    void vectorValue_leavesTheSourceUntouched() throws IOException {
        final float[] raw = { 3f, 4f };
        final FloatVectorValues view = new NormalizedFloatVectorValues(source(raw));

        view.vectorValue(0);

        assertArrayEquals(new float[] { 3f, 4f }, raw, "the flat vector must keep its indexed value");
    }

    @Test
    void vectorValue_keepsAZeroVectorAtZero() throws IOException {
        final FloatVectorValues view = new NormalizedFloatVectorValues(source(new float[] { 0f, 0f }));

        assertArrayEquals(new float[] { 0f, 0f }, view.vectorValue(0), DELTA);
    }

    @Test
    void copy_hasItsOwnScratch() throws IOException {
        final FloatVectorValues view = new NormalizedFloatVectorValues(source(new float[] { 3f, 4f }, new float[] { 0f, 2f }));
        final FloatVectorValues copy = view.copy();

        final float[] first = view.vectorValue(0);
        copy.vectorValue(1);

        assertArrayEquals(new float[] { 0.6f, 0.8f }, first, DELTA, "reading the copy must not overwrite this view's vector");
    }

    @Test
    void shapeAndOrdinals_comeFromTheSource() {
        final FloatVectorValues source = source(new float[] { 3f, 4f }, new float[] { 0f, 2f });
        final FloatVectorValues view = new NormalizedFloatVectorValues(source);

        assertEquals(DIMENSION, view.dimension());
        assertEquals(2, view.size());
        assertEquals(source.ordToDoc(1), view.ordToDoc(1));
    }

    private static FloatVectorValues source(final float[]... vectors) {
        return FloatVectorValues.fromFloats(List.of(vectors), DIMENSION);
    }
}
