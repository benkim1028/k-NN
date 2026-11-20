/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.opensearch.knn.clusterann.read.orchestration.ShardPlanner;
import org.opensearch.knn.index.codec.clusterann.ClusterANN1030TestCodec;

import java.io.IOException;
import java.util.Random;

/**
 * A one-segment shard under the default budget returns exactly what per-segment planning returned: the same hits with
 * the same scores, filtered or not. Per-segment planning is reached as the reader is searched without a shard plan.
 */
public class ClusterANNShardPlanSingleSegmentTests extends LuceneTestCase {

    private static final String FIELD = "v";
    private static final String BUCKET = "bucket";
    private static final int DIM = 8;
    private static final int NUM_DOCS = 30_000;

    private Directory directory;
    private DirectoryReader reader;
    private IndexSearcher searcher;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        directory = newFSDirectory(createTempDir("shard-plan-one"));
        final Random random = new Random(5);
        final IndexWriterConfig config = new IndexWriterConfig().setCodec(new ClusterANN1030TestCodec(4))
            .setMergePolicy(NoMergePolicy.INSTANCE)
            .setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setRAMBufferSizeMB(256)
            .setUseCompoundFile(false);
        try (IndexWriter writer = new IndexWriter(directory, config)) {
            for (int i = 0; i < NUM_DOCS; i++) {
                final float[] vector = new float[DIM];
                for (int d = 0; d < DIM; d++) {
                    vector[d] = random.nextFloat();
                }
                final Document document = new Document();
                document.add(new KnnFloatVectorField(FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                document.add(new StringField(BUCKET, Integer.toString(random.nextInt(100)), Field.Store.NO));
                writer.addDocument(document);
            }
        }
        reader = DirectoryReader.open(directory);
        searcher = new IndexSearcher(reader);
        assertEquals(1, reader.leaves().size());
    }

    @Override
    public void tearDown() throws Exception {
        reader.close();
        directory.close();
        super.tearDown();
    }

    public void testOneSegment_sameTopKAsPerSegmentPlanning() throws IOException {
        final Random random = new Random(9);
        final Query[] filters = { null, new TermQuery(new Term(BUCKET, "7")), new TermQuery(new Term(BUCKET, "42")) };
        for (int q = 0; q < 30; q++) {
            final float[] query = new float[DIM];
            for (int d = 0; d < DIM; d++) {
                query[d] = random.nextFloat();
            }
            final int k = q % 2 == 0 ? 10 : 100;
            for (final Query filter : filters) {
                final TopDocs planned = searcher.search(new ClusterANNQuery(FIELD, query, k, filter, null, 0), k);
                ClusterANNShardPlanTests.assertSameHits("query " + q + " k " + k + " filter " + filter, today(query, k, filter), planned);
            }
        }
    }

    /** Today's path: the reader searched with no shard plan on the collector, so it plans the segment itself. */
    private TopDocs today(final float[] query, final int k, final Query filter) throws IOException {
        final LeafReaderContext leaf = reader.leaves().get(0);
        final ClusterANNQuery q = new ClusterANNQuery(FIELD, query, k, filter, null, 0, ShardPlanner.Budget.CAPPED);
        final ClusterANNWeight weight = (ClusterANNWeight) searcher.createWeight(searcher.rewrite(q), ScoreMode.COMPLETE, 1f);
        final AcceptDocs acceptDocs = weight.shardPlan(leaf).forLeaf(leaf.ord).acceptDocs();
        final TopKnnCollector collector = new TopKnnCollector(k, Integer.MAX_VALUE);
        leaf.reader().searchNearestVectors(FIELD, query, collector, acceptDocs);
        return collector.topDocs();
    }
}
