/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.vectorformat1030;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.util.VectorUtil;

import java.io.IOException;

/**
 * A unit-length view of a cosine field's merged vectors, for clustering and quantization: the scalar quantizer
 * requires unit vectors under cosine. The source is the flat file's reopened vectors, so each vector is copied into
 * a per-view scratch buffer and normalized there, which is what {@link #vectorValue(int)} returns; successive calls
 * overwrite it. The source is never mutated.
 *
 * <p>Copied from Lucene's {@code Lucene103ScalarQuantizedVectorsWriter.NormalizedFloatVectorValues}, which is
 * package-private there and so cannot be reused, with one change: a zero vector is left at zero rather than
 * rejected.
 */
final class NormalizedFloatVectorValues extends FloatVectorValues {

    private final FloatVectorValues source;
    private final float[] scratch;

    NormalizedFloatVectorValues(final FloatVectorValues source) {
        this.source = source;
        this.scratch = new float[source.dimension()];
    }

    @Override
    public int dimension() {
        return source.dimension();
    }

    @Override
    public int size() {
        return source.size();
    }

    @Override
    public int ordToDoc(final int ord) {
        return source.ordToDoc(ord);
    }

    @Override
    public float[] vectorValue(final int ord) throws IOException {
        System.arraycopy(source.vectorValue(ord), 0, scratch, 0, scratch.length);
        return VectorUtil.l2normalize(scratch, false); // a zero vector stays zero rather than failing the merge
    }

    @Override
    public DocIndexIterator iterator() {
        return source.iterator();
    }

    @Override
    public NormalizedFloatVectorValues copy() throws IOException {
        return new NormalizedFloatVectorValues(source.copy());
    }
}
