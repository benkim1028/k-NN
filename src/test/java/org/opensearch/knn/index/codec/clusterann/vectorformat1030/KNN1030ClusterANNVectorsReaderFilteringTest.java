/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.vectorformat1030;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.tests.store.MockDirectoryWrapper;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.opensearch.knn.clusterann.format.ClusterANNFieldMeta;
import org.opensearch.knn.clusterann.format.ClusterANNFormatConstants;
import org.opensearch.knn.clusterann.read.ClusterANNFieldMetaEncoder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

class KNN1030ClusterANNVectorsReaderFilteringTest {

    private static final String FIELD = "float_field";
    private static final String SEGMENT = "_0";
    private static final int FIELD_NUMBER = 0;
    private static final int DIMENSION = 8;
    private static final int BLOCK_SIZE = 32;
    private static final int MAX_DOC = 100;
    private static final byte DOC_BITS = 1;
    private static final int DATA_FILE_BYTES = 4096;

    private final List<Directory> directories = new ArrayList<>();
    private final List<KNN1030ClusterANNVectorsReader> readers = new ArrayList<>();

    @AfterEach
    void tearDown() throws IOException {
        IOUtils.close(readers);
        IOUtils.close(directories);
    }

    @Test
    void search_whenSelectiveFilterOnSparseSegment_thenExactRouteUsesOrdToDocMapping() throws Exception {
        RecordingRandomVectorScorer scorer = new RecordingRandomVectorScorer(Map.of(0, 0.9f, 2, 0.7f));
        StubFlatVectorsReader flatReader = new StubFlatVectorsReader(scorer);
        KNN1030ClusterANNVectorsReader reader = openSparseReader(new int[] { 2, 5, 9 }, 12, flatReader);

        FixedBitSet acceptedDocs = new FixedBitSet(12);
        acceptedDocs.set(2);
        acceptedDocs.set(9);

        RecordingCollector collector = new RecordingCollector();
        reader.search(FIELD, query(), collector, new StaticAcceptDocs(acceptedDocs, 2));

        assertEquals(List.of(0, 2), scorer.scoredOrds);
        assertEquals(List.of(2, 9), collector.collectedDocs);
        assertEquals(1, flatReader.floatRandomScorerRequests);
    }

    @Test
    void search_whenFilterCostCoversAllVectors_thenSkipsExactRoute() throws Exception {
        StubFlatVectorsReader flatReader = new StubFlatVectorsReader(new RecordingRandomVectorScorer(Map.of()));
        KNN1030ClusterANNVectorsReader reader = openDenseReader(flatReader);

        FixedBitSet acceptedDocs = new FixedBitSet(MAX_DOC);
        acceptedDocs.set(0, 30);

        reader.search(FIELD, query(), new RecordingCollector(), new StaticAcceptDocs(acceptedDocs, 30));

        assertEquals(0, flatReader.floatRandomScorerRequests);
    }

    private KNN1030ClusterANNVectorsReader openDenseReader(StubFlatVectorsReader flatReader) throws IOException {
        return openDenseReader(flatReader, floatVectorField(DIMENSION, VectorSimilarityFunction.EUCLIDEAN), validEntry(), FIELD_NUMBER);
    }

    private KNN1030ClusterANNVectorsReader openDenseReader(
        StubFlatVectorsReader flatReader,
        FieldInfo info,
        ClusterANNFieldMetaEncoder entry,
        int entryFieldNumber
    ) throws IOException {
        MockDirectoryWrapper directory = new MockDirectoryWrapper(new Random(), new ByteBuffersDirectory());
        directory.setCheckIndexOnClose(false);
        directories.add(directory);

        SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            null,
            SEGMENT,
            MAX_DOC,
            false,
            false,
            null,
            Map.of(),
            StringHelper.randomId(),
            Map.of(),
            null
        );
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { info });
        SegmentReadState state = new SegmentReadState(directory, segmentInfo, fieldInfos, IOContext.DEFAULT);

        writeDenseMeta(state, entry, entryFieldNumber);
        writeBlank(state, KNN1030ClusterANNVectorsFormat.POSTINGS_EXTENSION);
        writeBlank(state, KNN1030ClusterANNVectorsFormat.CENTROIDS_EXTENSION);

        KNN1030ClusterANNVectorsReader opened = new KNN1030ClusterANNVectorsReader(state, flatReader);
        readers.add(opened);
        return opened;
    }

    private KNN1030ClusterANNVectorsReader openSparseReader(int[] docsWithVectors, int maxDoc, StubFlatVectorsReader flatReader)
        throws IOException {
        MockDirectoryWrapper directory = new MockDirectoryWrapper(new Random(), new ByteBuffersDirectory());
        directory.setCheckIndexOnClose(false);
        directories.add(directory);

        SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            null,
            SEGMENT,
            maxDoc,
            false,
            false,
            null,
            Map.of(),
            StringHelper.randomId(),
            Map.of(),
            null
        );
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { floatVectorField(DIMENSION, VectorSimilarityFunction.EUCLIDEAN) });
        SegmentReadState state = new SegmentReadState(directory, segmentInfo, fieldInfos, IOContext.DEFAULT);

        writeSparseMeta(state, sparseEntry(docsWithVectors.length), docs(docsWithVectors), maxDoc);
        writeBlank(state, KNN1030ClusterANNVectorsFormat.CENTROIDS_EXTENSION);

        KNN1030ClusterANNVectorsReader opened = new KNN1030ClusterANNVectorsReader(state, flatReader);
        readers.add(opened);
        return opened;
    }

    private static void writeDenseMeta(SegmentReadState state, ClusterANNFieldMetaEncoder entry, int entryFieldNumber) throws IOException {
        String name = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            KNN1030ClusterANNVectorsFormat.META_EXTENSION
        );
        try (IndexOutput out = state.directory.createOutput(name, IOContext.DEFAULT)) {
            CodecUtil.writeIndexHeader(
                out,
                KNN1030ClusterANNVectorsFormat.META_CODEC_NAME,
                KNN1030ClusterANNVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            out.writeVInt(BLOCK_SIZE);
            out.writeInt(entryFieldNumber);
            entry.write(out);
            out.writeInt(-1);
            CodecUtil.writeFooter(out);
        }
    }

    private static void writeSparseMeta(SegmentReadState state, ClusterANNFieldMeta entry, DocsWithFieldSet docsWithField, int maxDoc)
        throws IOException {
        String metaName = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            KNN1030ClusterANNVectorsFormat.META_EXTENSION
        );
        String postingsName = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            KNN1030ClusterANNVectorsFormat.POSTINGS_EXTENSION
        );

        try (
            IndexOutput metaOut = state.directory.createOutput(metaName, IOContext.DEFAULT);
            IndexOutput postingsOut = state.directory.createOutput(postingsName, IOContext.DEFAULT)
        ) {
            CodecUtil.writeIndexHeader(
                metaOut,
                KNN1030ClusterANNVectorsFormat.META_CODEC_NAME,
                KNN1030ClusterANNVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            metaOut.writeVInt(BLOCK_SIZE);
            metaOut.writeInt(FIELD_NUMBER);
            entry.write(metaOut, postingsOut, maxDoc, docsWithField);
            metaOut.writeInt(-1);
            CodecUtil.writeFooter(metaOut);

            long bytesToPad = DATA_FILE_BYTES - postingsOut.getFilePointer();
            if (bytesToPad > 0) {
                postingsOut.writeBytes(new byte[(int) bytesToPad], 0, (int) bytesToPad);
            }
            CodecUtil.writeFooter(postingsOut);
        }
    }

    private static void writeBlank(SegmentReadState state, String extension) throws IOException {
        String name = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, extension);
        try (IndexOutput out = state.directory.createOutput(name, IOContext.DEFAULT)) {
            out.writeBytes(new byte[DATA_FILE_BYTES], 0, DATA_FILE_BYTES);
            CodecUtil.writeFooter(out);
        }
    }

    private static ClusterANNFieldMetaEncoder validEntry() {
        return new ClusterANNFieldMetaEncoder().dimension(DIMENSION)
            .similarityFunction(ClusterANNFieldMetaEncoder.SIMILARITY_L2)
            .rotationId((byte) ClusterANNFormatConstants.ROTATION_NONE)
            .docBits(DOC_BITS);
    }

    private static ClusterANNFieldMeta sparseEntry(int vectorCount) {
        return new ClusterANNFieldMeta(
            BLOCK_SIZE,
            DIMENSION,
            vectorCount,
            1,
            VectorSimilarityFunction.EUCLIDEAN,
            DOC_BITS,
            ClusterANNFormatConstants.ROTATION_NONE,
            ClusterANNFormatConstants.QUANTIZER_OPTIMIZED_SQ,
            new byte[0],
            0L,
            512L,
            64L,
            -1L,
            0L,
            900L,
            new long[] { 0L },
            new int[] { 100 },
            new int[] { vectorCount },
            -1L,
            -1L,
            null
        );
    }

    private static DocsWithFieldSet docs(int... docIds) throws IOException {
        DocsWithFieldSet docs = new DocsWithFieldSet();
        for (int docId : docIds) {
            docs.add(docId);
        }
        return docs;
    }

    private static float[] query() {
        float[] query = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            query[i] = 0.1f * (i + 1);
        }
        return query;
    }

    private static FieldInfo floatVectorField(int dimension, VectorSimilarityFunction similarity) {
        return new FieldInfo(
            FIELD,
            FIELD_NUMBER,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Map.of(),
            0,
            0,
            0,
            dimension,
            VectorEncoding.FLOAT32,
            similarity,
            false,
            false
        );
    }

    private static final class StubFlatVectorsReader extends FlatVectorsReader {

        private final RandomVectorScorer floatScorer;
        private int floatRandomScorerRequests;

        private StubFlatVectorsReader(RandomVectorScorer floatScorer) {
            super(DefaultFlatVectorScorer.INSTANCE);
            this.floatScorer = floatScorer;
        }

        @Override
        public void checkIntegrity() {}

        @Override
        public FloatVectorValues getFloatVectorValues(String fieldName) {
            return null;
        }

        @Override
        public ByteVectorValues getByteVectorValues(String fieldName) {
            return null;
        }

        @Override
        public RandomVectorScorer getRandomVectorScorer(String fieldName, float[] target) {
            floatRandomScorerRequests++;
            return floatScorer;
        }

        @Override
        public RandomVectorScorer getRandomVectorScorer(String fieldName, byte[] target) {
            throw new UnsupportedOperationException("byte vectors not used in this test");
        }

        @Override
        public long ramBytesUsed() {
            return 0L;
        }

        @Override
        public void close() {}
    }

    private static final class RecordingRandomVectorScorer implements RandomVectorScorer {

        private final Map<Integer, Float> scores;
        private final List<Integer> scoredOrds = new ArrayList<>();

        private RecordingRandomVectorScorer(Map<Integer, Float> scores) {
            this.scores = new HashMap<>(scores);
        }

        @Override
        public float score(int ord) {
            scoredOrds.add(ord);
            return scores.getOrDefault(ord, 0.0f);
        }

        @Override
        public int maxOrd() {
            return Integer.MAX_VALUE;
        }
    }

    private static final class StaticAcceptDocs extends AcceptDocs {

        private final FixedBitSet bits;
        private final int cost;

        private StaticAcceptDocs(FixedBitSet bits, int cost) {
            this.bits = bits;
            this.cost = cost;
        }

        @Override
        public FixedBitSet bits() {
            return bits;
        }

        @Override
        public DocIdSetIterator iterator() {
            return new BitSetIterator(bits, cost);
        }

        @Override
        public int cost() {
            return cost;
        }
    }

    private static final class RecordingCollector implements KnnCollector {

        private final List<Integer> collectedDocs = new ArrayList<>();

        @Override
        public boolean earlyTerminated() {
            return false;
        }

        @Override
        public void incVisitedCount(int count) {}

        @Override
        public long visitedCount() {
            return collectedDocs.size();
        }

        @Override
        public long visitLimit() {
            return Long.MAX_VALUE;
        }

        @Override
        public int k() {
            return 10;
        }

        @Override
        public boolean collect(int docId, float similarity) {
            collectedDocs.add(docId);
            return true;
        }

        @Override
        public float minCompetitiveSimilarity() {
            return Float.NEGATIVE_INFINITY;
        }

        @Override
        public TopDocs topDocs() {
            throw new UnsupportedOperationException("not needed by these tests");
        }

        @Override
        public KnnSearchStrategy getSearchStrategy() {
            return null;
        }
    }
}
