/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clusterann;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexReaderContext;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.ReaderUtil;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TimeLimitingKnnCollectorManager;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.util.ThreadInterruptedException;
import org.opensearch.common.Nullable;
import org.opensearch.knn.clusterann.read.orchestration.ShardPlanStrategy;
import org.opensearch.knn.clusterann.read.orchestration.ShardPlanner;
import org.opensearch.knn.index.codec.clusterann.vectorformat1030.KNN1030ClusterANNVectorsReader;
import org.opensearch.knn.index.query.KNNScorer;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;

/**
 * Scans one segment per {@link #scorerSupplier}, rather than every segment up front.
 *
 * <p>Lazy because a segment's scan is only worth paying for when someone asks for that segment's hits, and because the
 * caller is better placed to parallelise than this class is: a rescorer already fans out per segment, so doing it here
 * as well would mean two fan-outs and an intermediate merge that the rescorer immediately takes apart again.
 *
 * <p>All the segments share one competitive threshold through the collector manager, which is what makes the
 * parallelism pay for itself — a segment that starts late learns the score it must beat from the ones already done, and
 * prunes from its first cluster instead of rediscovering the same bar.
 *
 * <p><b>The probe budget is planned per shard, not per segment.</b> The first segment asked for its hits plans the
 * whole shard ({@link ShardPlanner}): every segment's centroids ranked against the query together, the global top X
 * kept, and each segment handed its own share through the collector's search strategy ({@link ShardPlanStrategy}). The
 * reader then scans exactly that list. A segment the plan gave no cluster returns no hits; there is no fall back to
 * exact search for a segment that returns fewer than k, which a shard plan makes routine. Every other segment waits
 * for that one plan, which is the price of comparing their centroids.
 */
final class ClusterANNWeight extends Weight {

    private final String field;
    private final float[] queryVector;
    private final int candidateK;
    private final float boost;

    @Nullable
    private final Weight filterWeight;

    private final TimeLimitingKnnCollectorManager collectorManager;

    private final ShardPlanner.Budget budget;

    /**
     * The shard plan, made once per top-level reader by whichever segment asks first; the others wait on it. Keyed on
     * the top-level context so a weight is never handed a plan made for another reader's leaves.
     */
    private final Map<IndexReaderContext, FutureTask<LeafPlans>> shardPlans = new ConcurrentHashMap<>();

    /**
     * Each segment's hits, kept after the first scan.
     *
     * <p>Lucene may ask for a segment's scorer more than once — {@code count()}, {@code explain()}, a caching layer —
     * and a scan is far too expensive to repeat for an answer that cannot have changed.
     */
    private final Map<Integer, TopDocs> perLeafResults = new ConcurrentHashMap<>();

    ClusterANNWeight(
        final Query query,
        final String field,
        final float[] queryVector,
        final int candidateK,
        @Nullable final Weight filterWeight,
        final TimeLimitingKnnCollectorManager collectorManager,
        final float boost,
        final ShardPlanner.Budget budget
    ) {
        super(query);
        this.field = field;
        this.queryVector = queryVector;
        this.candidateK = candidateK;
        this.filterWeight = filterWeight;
        this.collectorManager = collectorManager;
        this.boost = boost;
        this.budget = budget;
    }

    @Override
    public ScorerSupplier scorerSupplier(final LeafReaderContext context) {
        return new ScorerSupplier() {
            @Override
            public Scorer get(final long leadCost) throws IOException {
                final TopDocs topDocs = searchLeaf(context);
                if (topDocs.scoreDocs.length == 0) {
                    return KNNScorer.emptyScorer();
                }
                return new KNNScorer(topDocs, boost);
            }

            /**
             * Before the scan there is nothing to count, so this is what the scan is allowed to return rather than what
             * it will. Afterwards the real number is known and worth reporting, since a rescorer sizes its queue from
             * it.
             */
            @Override
            public long cost() {
                final TopDocs done = perLeafResults.get(context.ord);
                return done == null ? candidateK : done.scoreDocs.length;
            }
        };
    }

    /**
     * The approximate hits of one segment, in that segment's own doc space.
     *
     * <p>Doc ids are left segment-local. Whoever merges across segments shifts them, and doing it here would mean the
     * memoised results could not be handed out twice.
     */
    TopDocs searchLeaf(final LeafReaderContext context) throws IOException {
        final TopDocs cached = perLeafResults.get(context.ord);
        if (cached != null) {
            return cached;
        }

        final LeafPlan plan = shardPlan(context).forLeaf(context.ord);
        // The plan's filter, already evaluated for this segment, so the scan does not evaluate it again.
        final AcceptDocs acceptDocs = plan == null ? acceptDocs(context) : plan.acceptDocs();
        TopDocs results = TopDocsCollector.EMPTY_TOPDOCS;
        // The shard plan gave this segment no cluster: nothing to scan and, deliberately, no exact search either.
        final boolean nothingPlanned = plan != null && plan.strategy() == null;
        if (!nothingPlanned && acceptDocs.cost() > 0) {
            // TODO: short-circuit to exact search when the filter is selective enough that scanning postings cannot pay
            // for itself. The threshold belongs to the probed clusters' sizes, which the codec knows and this layer does
            // not, so it is left out rather than approximated with a bound that means nothing for a cluster scan.

            // No visit limit: the scan reads earlyTerminated() only while the EXPERIMENT scan early stop is on (where it
            // also honours the time-limiting wrapper's timeout), so a limit here would otherwise be quietly ignored. Said
            // outright rather than passed as a number that does nothing.
            // Null for a nested field's segment that holds no parents: there are no children to score there either, so
            // this is "nothing here", not "nothing matched".
            // A segment outside the shard plan (no ClusterANN reader reachable for it) keeps the default strategy, so
            // its reader does whatever it does without one.
            final KnnSearchStrategy strategy = plan == null ? KnnSearchStrategy.Hnsw.DEFAULT : plan.strategy();
            final KnnCollector collector = collectorManager.newCollector(Integer.MAX_VALUE, strategy, context);
            if (collector != null) {
                context.reader().searchNearestVectors(field, queryVector, collector, acceptDocs);
                results = collector.topDocs();
            }
        }

        perLeafResults.put(context.ord, results);
        return results;
    }

    /**
     * The shard plan for the reader {@code context} belongs to, made on first use. Thread-safe: concurrent segments
     * share one plan, and the planning runs once.
     */
    LeafPlans shardPlan(final LeafReaderContext context) throws IOException {
        final IndexReaderContext top = ReaderUtil.getTopLevelContext(context);
        final FutureTask<LeafPlans> task = new FutureTask<>(() -> planShard(top.leaves()));
        final FutureTask<LeafPlans> existing = shardPlans.putIfAbsent(top, task);
        final FutureTask<LeafPlans> plan = existing == null ? task : existing;
        if (existing == null) {
            task.run();
        }
        try {
            return plan.get();
        } catch (InterruptedException e) {
            throw new ThreadInterruptedException(e);
        } catch (ExecutionException e) {
            final Throwable cause = e.getCause();
            if (cause instanceof IOException io) {
                throw io;
            }
            if (cause instanceof RuntimeException runtime) {
                throw runtime;
            }
            if (cause instanceof Error error) {
                throw error;
            }
            throw new UncheckedIOException(new IOException(cause));
        }
    }

    /**
     * Plans every leaf at once. A leaf whose ClusterANN reader cannot be reached gets no entry, and is searched as it
     * would be without a shard plan; a reachable leaf without clusters for the field gets an entry with nothing to
     * scan, since its reader would return nothing anyway.
     */
    private LeafPlans planShard(final List<LeafReaderContext> leaves) throws IOException {
        final LeafPlan[] byOrd = new LeafPlan[leaves.size()];
        final List<ShardPlanner.Segment> segments = new ArrayList<>(leaves.size());
        final List<LeafReaderContext> planned = new ArrayList<>(leaves.size());
        final List<AcceptDocs> plannedAcceptDocs = new ArrayList<>(leaves.size());
        for (final LeafReaderContext leaf : leaves) {
            final KNN1030ClusterANNVectorsReader reader = clusterAnnReader(leaf.reader(), field);
            if (reader == null) {
                continue;
            }
            final AcceptDocs acceptDocs = acceptDocs(leaf);
            final ShardPlanner.Segment segment = acceptDocs.cost() > 0 ? reader.shardPlanSegment(field, acceptDocs) : null;
            if (segment == null) {
                byOrd[leaf.ord] = new LeafPlan(acceptDocs, null);
                continue;
            }
            segments.add(segment);
            planned.add(leaf);
            plannedAcceptDocs.add(acceptDocs);
        }
        final ShardPlanner.Plan plan = ShardPlanner.plan(segments, queryVector, candidateK, budget);
        for (int i = 0; i < planned.size(); i++) {
            final int[] probes = plan.probes()[i];
            final ShardPlanStrategy strategy = probes.length == 0
                ? null
                : new ShardPlanStrategy(segments.get(i).clusters(), probes, plan.params());
            byOrd[planned.get(i).ord] = new LeafPlan(plannedAcceptDocs.get(i), strategy);
        }
        return new LeafPlans(byOrd, plan);
    }

    /**
     * The ClusterANN reader behind a leaf for this field, or {@code null} when there is none to reach: through filter
     * readers to the codec reader, then through a per-field format to the field's own reader.
     */
    @Nullable
    static KNN1030ClusterANNVectorsReader clusterAnnReader(final LeafReader leaf, final String field) {
        if (!(FilterLeafReader.unwrap(leaf) instanceof CodecReader codecReader)) {
            return null;
        }
        KnnVectorsReader vectors = codecReader.getVectorReader();
        if (vectors instanceof PerFieldKnnVectorsFormat.FieldsReader perField) {
            vectors = perField.getFieldReader(field);
        }
        return vectors instanceof KNN1030ClusterANNVectorsReader clusterAnn ? clusterAnn : null;
    }

    /**
     * One leaf's part of the shard plan.
     *
     * @param acceptDocs the leaf's filter, already evaluated while planning
     * @param strategy what to hand the reader, or {@code null} when the plan gave the leaf no cluster
     */
    record LeafPlan(AcceptDocs acceptDocs, @Nullable ShardPlanStrategy strategy) {
    }

    /**
     * Every leaf's part of the shard plan, by leaf ord.
     *
     * @param byOrd per leaf ord; {@code null} for a leaf outside the plan
     * @param plan the plan itself, for its bounds and totals
     */
    record LeafPlans(LeafPlan[] byOrd, ShardPlanner.Plan plan) {
        @Nullable
        LeafPlan forLeaf(final int ord) {
            return ord < byOrd.length ? byOrd[ord] : null;
        }
    }

    /**
     * What this segment may return: the filter's matches if there is one, otherwise its live documents.
     *
     * <p>With no filter and no deletions the bits come back {@code null} — Lucene's own way of saying "everything" —
     * which is what lets the codec skip the per-vector membership test entirely instead of probing a set that would
     * answer true every time.
     */
    private AcceptDocs acceptDocs(final LeafReaderContext context) throws IOException {
        if (filterWeight == null) {
            return AcceptDocs.fromLiveDocs(context.reader().getLiveDocs(), context.reader().maxDoc());
        }
        return AcceptDocs.fromIteratorSupplier(() -> {
            final Scorer scorer = filterWeight.scorer(context);
            return scorer == null ? DocIdSetIterator.empty() : scorer.iterator();
        }, context.reader().getLiveDocs(), context.reader().maxDoc());
    }

    @Override
    public Explanation explain(final LeafReaderContext context, final int doc) throws IOException {
        final TopDocs topDocs = searchLeaf(context);
        for (final var scoreDoc : topDocs.scoreDocs) {
            if (scoreDoc.doc == doc) {
                return Explanation.match(scoreDoc.score * boost, "within top " + candidateK + " of the ClusterANN scan");
            }
        }
        return Explanation.noMatch("not among the top " + candidateK + " of the ClusterANN scan");
    }

    /**
     * Never cacheable.
     *
     * <p>A cache entry is keyed on the query, and a query vector is effectively unique — so caching would never hit,
     * while filling the cache costs an extra call into {@link #scorerSupplier} and therefore an extra segment scan.
     */
    @Override
    public boolean isCacheable(final LeafReaderContext context) {
        return false;
    }
}
