/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.clusterann.read.orchestration;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.LongValues;
import org.apache.lucene.util.VectorUtil;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.opensearch.knn.clusterann.format.ClusterANNFieldMeta;
import org.opensearch.knn.clusterann.format.ClusterANNFormatConstants;
import org.opensearch.knn.clusterann.read.CentroidVectorValues;
import org.opensearch.knn.clusterann.read.Clusters;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * {@link ShardPlanner}: the shard bounds, the global merge, the shard-level hold of k, the X policy, and the counts it
 * hands each segment, which {@link CentroidPlanner#planWithin} must turn back into exactly the clusters counted.
 *
 * <p>{@link Clusters} is mocked (it is final and owns file inputs), but the centroids are real bytes read by a real
 * {@link CentroidVectorValues}, so the keys are the planner's own.
 */
class ShardPlannerTests {

    private static final float[] ORIGIN = { 0f, 0f };

    private final List<Directory> directories = new ArrayList<>();

    @AfterEach
    void tearDown() throws IOException {
        for (Directory directory : directories) {
            directory.close();
        }
    }

    // ---------------------------------------------------------------- shard bounds

    /** R = ceil(max(8, m(s)) * floor(sqrt C)), floor = the shipped floor at C, both capped at C. */
    @Test
    void testShardParams_thenRuleRAndShippedFloor() {
        assertEquals(new PlanParams(10, 80), ShardPlanner.shardParams(100, 1.0));
        // m(0.01) = 1 + 2 / 0.01^(1/3) = 10.28 > 8, so R = ceil(102.8) = 103, capped at C.
        assertEquals(new PlanParams(10, 100), ShardPlanner.shardParams(100, 0.01));
        assertEquals(new PlanParams(100, 800), ShardPlanner.shardParams(10_000, 1.0));
        assertEquals(new PlanParams(4, 4), ShardPlanner.shardParams(4, 1.0));
        assertEquals(new PlanParams(1, 1), ShardPlanner.shardParams(1, 1.0));
        assertThrows(IllegalArgumentException.class, () -> ShardPlanner.shardParams(0, 1.0));
    }

    /** R and the shipped curve disagree: the curve's ceiling is 3x unfiltered, R's 8x. The plan keeps R. */
    @Test
    void testShardParams_whenUnfiltered_thenCeilingIsAboveTheShippedCurve() {
        assertEquals(30, PlanParams.forSelectivity(100, 1.0).maxProbes());
        assertEquals(80, ShardPlanner.shardParams(100, 1.0).maxProbes());
    }

    // ---------------------------------------------------------------- global merge

    /** Uneven segments, one with no clusters: the cut takes the globally closest, whichever segment they are in. */
    @Test
    void testMergeTake_whenUnevenWithAnEmptySegment_thenTakesTheGlobalPrefix() throws IOException {
        final CentroidPlanner.Ranked[] ranked = {
            ranked(new float[] { 0f, 1f, 2f, 3f }),
            ranked(new float[] { 0.5f, 1.5f }),
            ranked(new float[0]) };
        final List<ShardPlanner.Segment> segments = unfilteredSegments(3);

        assertArrayEquals(new int[] { 2, 2, 0 }, ShardPlanner.mergeTake(segments, ranked, 4, 0));
        assertArrayEquals(new int[] { 1, 0, 0 }, ShardPlanner.mergeTake(segments, ranked, 1, 0));
        assertArrayEquals(new int[] { 4, 2, 0 }, ShardPlanner.mergeTake(segments, ranked, 100, 0));
        assertArrayEquals(new int[] { 0, 0, 0 }, ShardPlanner.mergeTake(segments, ranked, 0, 0));
    }

    /** Equal keys across segments go to the lower segment first, so the counts are deterministic. */
    @Test
    void testMergeTake_whenKeysTie_thenLowerSegmentFirst() throws IOException {
        final CentroidPlanner.Ranked[] ranked = { ranked(new float[] { 1f, 1f }), ranked(new float[] { 1f, 1f }) };
        assertArrayEquals(new int[] { 2, 1 }, ShardPlanner.mergeTake(unfilteredSegments(2), ranked, 3, 0));
        assertArrayEquals(new int[] { 1, 0 }, ShardPlanner.mergeTake(unfilteredSegments(2), ranked, 1, 0));
    }

    /** Under a filter the cut grows past X until the shard, not each segment, holds k accepted vectors. */
    @Test
    void testMergeTake_whenFiltered_thenExtendsUntilTheShardHoldsK() throws IOException {
        final CentroidPlanner.Ranked[] ranked = {
            new CentroidPlanner.Ranked(new int[] { 0, 1, 2, 3 }, new float[] { 0f, 1f, 2f, 3f }),
            new CentroidPlanner.Ranked(new int[] { 0, 1 }, new float[] { 0.5f, 1.5f }) };
        final List<ShardPlanner.Segment> segments = List.of(
            filteredSegment(new int[] { 1, 1, 1, 1 }),
            filteredSegment(new int[] { 50, 50 })
        );
        // Order: a0 (1), b0 (51), a1 (52), b1 (102). X = 1 alone holds 1.
        assertArrayEquals(new int[] { 1, 0 }, ShardPlanner.mergeTake(segments, ranked, 1, 1));
        assertArrayEquals(new int[] { 1, 1 }, ShardPlanner.mergeTake(segments, ranked, 1, 51));
        assertArrayEquals(new int[] { 2, 2 }, ShardPlanner.mergeTake(segments, ranked, 1, 60));
        // More than the shard holds: everything.
        assertArrayEquals(new int[] { 4, 2 }, ShardPlanner.mergeTake(segments, ranked, 1, 1_000));
    }

    /**
     * End to end on a line: segment 0 at the even positions 0..158, segment 1 at the odd positions 1..79, segment 2
     * with no clusters. C_total = 120, so R = 80 = positions 0..79, half from each; the floor of 10 is 0..9.
     */
    @Test
    void testPlan_whenRuleOnALine_thenCountsTheGlobalCut() throws IOException {
        final List<ShardPlanner.Segment> segments = List.of(
            new ShardPlanner.Segment(line(VectorSimilarityFunction.EUCLIDEAN, 0, 2, 80), null),
            new ShardPlanner.Segment(line(VectorSimilarityFunction.EUCLIDEAN, 1, 2, 40), null),
            new ShardPlanner.Segment(emptyClusters(), null)
        );

        final ShardPlanner.Plan plan = ShardPlanner.plan(segments, ORIGIN, 10, ShardPlanner.Budget.RULE);

        assertEquals(new PlanParams(10, 80), plan.params());
        assertEquals(80, plan.budget());
        assertEquals(120, plan.totalClusters());
        assertEquals(new PlanParams(5, 40), plan.segmentParams()[0]);
        assertEquals(new PlanParams(5, 40), plan.segmentParams()[1]);
        assertEquals(new PlanParams(0, 0), plan.segmentParams()[2]);
        assertEquals(80, plan.probeCount());
    }

    /** A segment whose clusters are all beyond the cut gets (0, 0), and planWithin turns that into no probes. */
    @Test
    void testPlan_whenASegmentIsBeyondTheCut_thenItGetsNothing() throws IOException {
        final Clusters far = line(VectorSimilarityFunction.EUCLIDEAN, 1_000, 1, 4);
        final List<ShardPlanner.Segment> segments = List.of(
            new ShardPlanner.Segment(line(VectorSimilarityFunction.EUCLIDEAN, 0, 1, 100), null),
            new ShardPlanner.Segment(far, null)
        );

        final ShardPlanner.Plan plan = ShardPlanner.plan(segments, ORIGIN, 10, ShardPlanner.Budget.RULE);

        assertEquals(new PlanParams(0, 0), plan.segmentParams()[1]);
        assertEquals(0, CentroidPlanner.planWithin(far, ORIGIN, plan.segmentParams()[1], null).length);
    }

    /** No cluster anywhere: no bounds, and every segment gets nothing. */
    @Test
    void testPlan_whenNoClusters_thenNothing() throws IOException {
        final ShardPlanner.Plan plan = ShardPlanner.plan(
            List.of(new ShardPlanner.Segment(emptyClusters(), null)),
            ORIGIN,
            10,
            ShardPlanner.Budget.CAPPED
        );
        assertNull(plan.params());
        assertEquals(new PlanParams(0, 0), plan.segmentParams()[0]);
    }

    // ---------------------------------------------------------------- filtered hold at the shard

    /** The shard holds k or fewer accepted vectors: every eligible cluster of every segment, past R. */
    @Test
    void testPlan_whenTheShardHoldsAtMostK_thenEveryEligibleCluster() throws IOException {
        final Random random = new Random(3);
        final List<ShardPlanner.Segment> segments = new ArrayList<>();
        int eligible = 0;
        int accepted = 0;
        for (int s = 0; s < 3; s++) {
            final Fixture f = fixture(random, VectorSimilarityFunction.EUCLIDEAN, 150, 8, 0.002);
            segments.add(new ShardPlanner.Segment(f.clusters(), f.digest()));
            eligible += f.digest().eligibleClusters();
            accepted += f.digest().accepted();
        }
        for (final int k : new int[] { accepted, accepted + 1 }) {
            final ShardPlanner.Plan plan = ShardPlanner.plan(segments, query(random, 8), k, ShardPlanner.Budget.RULE);
            assertEquals(eligible, plan.probeCount(), "k=" + k + " accepted=" + accepted);
        }
    }

    /**
     * More than k accepted: the cut holds k across the shard, and is minimal once past X, so no segment holds k on
     * its own unless the global order put it there.
     */
    @ParameterizedTest
    @ValueSource(ints = { 1, 10, 100, 1000 })
    void testPlan_whenFiltered_thenTheCutHoldsKAtTheShard(final int k) throws IOException {
        for (int seed = 0; seed < 20; seed++) {
            final Random random = new Random(seed);
            final List<ShardPlanner.Segment> segments = new ArrayList<>();
            long accepted = 0;
            for (int s = 0; s < 4; s++) {
                final Fixture f = fixture(random, VectorSimilarityFunction.EUCLIDEAN, 20 + random.nextInt(150), 8, 0.05);
                segments.add(new ShardPlanner.Segment(f.clusters(), f.digest()));
                accepted += f.digest().accepted();
            }
            final ShardPlanner.Ranked planned = ShardPlanner.planRanked(segments, query(random, 8), k, ShardPlanner.Budget.RULE);
            long held = 0;
            for (int s = 0; s < segments.size(); s++) {
                final int[] ords = planned.rankings()[s].ords();
                for (int i = 0; i < planned.plan().segmentParams()[s].maxProbes(); i++) {
                    held += segments.get(s).digest().acceptedPerCluster()[ords[i]];
                }
            }
            assertTrue(held >= Math.min(k, accepted), "seed " + seed + ": held " + held + " < k " + k);
            assertTrue(planned.plan().probeCount() >= Math.min(planned.plan().budget(), eligible(segments)), "seed " + seed);
        }
    }

    /**
     * A filtered shard with an undigested segment (nothing filtered out there) of many small clusters: the cut must hold
     * k through that segment's own ranking rather than run its list dry, so the shard holds min(k, available), and the
     * segments' re-ranks still probe exactly what was counted.
     */
    @Test
    void testPlan_whenFilteredShardMixesAnUndigestedSegment_thenTheCutStillHoldsK() throws IOException {
        final Random random = new Random(42);
        final Fixture small = fixture(random, VectorSimilarityFunction.EUCLIDEAN, 300, 4, 1.0);
        final Fixture filtered = fixture(random, VectorSimilarityFunction.EUCLIDEAN, 60, 4, 0.5);
        final List<ShardPlanner.Segment> segments = List.of(
            new ShardPlanner.Segment(small.clusters(), null),
            new ShardPlanner.Segment(filtered.clusters(), filtered.digest())
        );
        final long available = small.clusters().numVectors() + filtered.digest().accepted();
        final float[] query = query(random, 4);
        for (final int k : new int[] { 1000, 3000, (int) available - 1 }) {
            final ShardPlanner.Ranked planned = ShardPlanner.planRanked(segments, query, k, ShardPlanner.Budget.RULE);
            final PlanParams[] params = planned.plan().segmentParams();
            long held = 0;
            for (int i = 0; i < params[0].maxProbes(); i++) {
                held += small.clusters().clusterSize(planned.rankings()[0].ords()[i]);
            }
            for (int i = 0; i < params[1].maxProbes(); i++) {
                held += filtered.digest().acceptedPerCluster()[planned.rankings()[1].ords()[i]];
            }
            assertTrue(held >= Math.min(k, available), "k=" + k + ": held " + held + " of " + available);
            // The cut is a global prefix: nothing it left out, in either segment, is closer than something it took.
            float farthestTaken = Float.NEGATIVE_INFINITY;
            float closestLeft = Float.POSITIVE_INFINITY;
            for (int s = 0; s < 2; s++) {
                final Clusters clusters = segments.get(s).clusters();
                final CentroidPlanner.Ranked all = CentroidPlanner.rankClosestFirst(
                    clusters,
                    query,
                    clusters.numClusters(),
                    segments.get(s).digest(),
                    0
                );
                if (params[s].maxProbes() > 0) {
                    farthestTaken = Math.max(farthestTaken, all.keys()[params[s].maxProbes() - 1]);
                }
                if (params[s].maxProbes() < all.size()) {
                    closestLeft = Math.min(closestLeft, all.keys()[params[s].maxProbes()]);
                }
            }
            assertTrue(farthestTaken <= closestLeft, "k=" + k + ": took " + farthestTaken + " but left " + closestLeft);
            for (int s = 0; s < 2; s++) {
                final int[] counted = Arrays.copyOf(planned.rankings()[s].ords(), params[s].maxProbes());
                assertArrayEquals(
                    counted,
                    CentroidPlanner.planWithin(segments.get(s).clusters(), query, params[s], segments.get(s).digest())
                );
            }
        }
    }

    // ---------------------------------------------------------------- X policy

    /**
     * H spends min(R, P), where P is what each segment's own plan, CentroidPlanner.plan under PlanParams.of(C_s), would
     * probe, read off the shard's rankings; R spends R. Checked against the real per-segment planner, filtered
     * (holds included) and unfiltered.
     */
    @ParameterizedTest
    @ValueSource(booleans = { false, true })
    void testPlan_whenCapped_thenXIsTheSmallerOfRAndThePerSegmentTotal(final boolean filtered) throws IOException {
        for (int seed = 0; seed < 30; seed++) {
            final Random random = new Random(100 + seed);
            final VectorSimilarityFunction similarity = seed % 2 == 0
                ? VectorSimilarityFunction.EUCLIDEAN
                : VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
            final int k = new int[] { 1, 10, 100 }[seed % 3];
            final List<ShardPlanner.Segment> segments = new ArrayList<>();
            final List<Fixture> fixtures = new ArrayList<>();
            for (int s = 0, segmentCount = 1 + random.nextInt(5); s < segmentCount; s++) {
                final Fixture f = fixture(
                    random,
                    similarity,
                    5 + random.nextInt(200),
                    4,
                    filtered ? 0.02 + random.nextDouble() * 0.3 : 1.0
                );
                fixtures.add(f);
                segments.add(new ShardPlanner.Segment(f.clusters(), filtered ? f.digest() : null));
            }
            final float[] query = query(random, 4);
            int today = 0;
            for (final Fixture f : fixtures) {
                today += CentroidPlanner.plan(
                    f.clusters(),
                    query,
                    PlanParams.of(f.clusters().numClusters()),
                    k,
                    filtered ? f.acceptDocs() : null,
                    () -> {
                        throw new AssertionError("dense");
                    }
                ).length;
            }
            final ShardPlanner.Plan capped = ShardPlanner.plan(segments, query, k, ShardPlanner.Budget.CAPPED);
            final ShardPlanner.Plan rule = ShardPlanner.plan(segments, query, k, ShardPlanner.Budget.RULE);
            assertEquals(Math.min(rule.params().maxProbes(), today), capped.budget(), "seed " + seed);
            assertEquals(rule.params().maxProbes(), rule.budget(), "seed " + seed);
            assertTrue(capped.probeCount() <= rule.probeCount(), "seed " + seed);
        }
    }

    // ---------------------------------------------------------------- EXPERIMENT switches (do not merge)

    /**
     * Knee off: rule H's cost cap is what each segment's own knee-off plan would probe (its ceiling, or the nearest that
     * hold k), checked against the real per-segment planner, so the shard applies no knee anywhere.
     */
    @ParameterizedTest
    @ValueSource(booleans = { false, true })
    void testPlan_whenKneeOff_thenTheCappedBudgetHasNoKnee(final boolean filtered) throws IOException {
        KneeExperiment.setEnabled(false);
        try {
            for (int seed = 0; seed < 30; seed++) {
                final Random random = new Random(900 + seed);
                final int k = new int[] { 1, 10, 100 }[seed % 3];
                final List<ShardPlanner.Segment> segments = new ArrayList<>();
                final List<Fixture> fixtures = new ArrayList<>();
                for (int s = 0, segmentCount = 1 + random.nextInt(5); s < segmentCount; s++) {
                    final Fixture f = fixture(
                        random,
                        VectorSimilarityFunction.EUCLIDEAN,
                        5 + random.nextInt(200),
                        4,
                        filtered ? 0.05 + random.nextDouble() * 0.3 : 1.0
                    );
                    fixtures.add(f);
                    segments.add(new ShardPlanner.Segment(f.clusters(), filtered ? f.digest() : null));
                }
                final float[] query = query(random, 4);
                int today = 0;
                for (final Fixture f : fixtures) {
                    today += CentroidPlanner.plan(
                        f.clusters(),
                        query,
                        PlanParams.of(f.clusters().numClusters()),
                        k,
                        filtered ? f.acceptDocs() : null,
                        () -> {
                            throw new AssertionError("dense");
                        }
                    ).length;
                }
                final ShardPlanner.Plan capped = ShardPlanner.plan(segments, query, k, ShardPlanner.Budget.CAPPED);
                assertEquals(Math.min(capped.params().maxProbes(), today), capped.budget(), "seed " + seed);
            }
        } finally {
            KneeExperiment.reset();
        }
    }

    /**
     * The benchmark's sanity point: knee off, a static probe curve (coefficient 0, intercept m), shard floor 0 and rule
     * R. A one-segment shard then probes exactly what the baseline's per-segment knee-off plan probes, filtered or not.
     */
    @ParameterizedTest
    @ValueSource(booleans = { false, true })
    void testPlan_whenOneSegmentStaticMKneeOffRuleFloorZero_thenSameProbesAsTheBaseline(final boolean filtered) throws IOException {
        KneeExperiment.setEnabled(false);
        ShardPlanExperiment.set(new ShardPlanExperiment.Config(true, ShardPlanner.Budget.RULE, 0.0));
        try {
            for (int seed = 0; seed < 40; seed++) {
                final int m = new int[] { 1, 2, 4, 8 }[seed % 4];
                ProbeCurveExperiment.set(ProbeCurveExperiment.DEFAULT_CURVE.withIntercept(m).withCoefficient(0.0));
                final Random random = new Random(1300 + seed);
                final int k = new int[] { 1, 10, 100 }[seed % 3];
                final Fixture f = fixture(
                    random,
                    VectorSimilarityFunction.EUCLIDEAN,
                    1 + random.nextInt(400),
                    4,
                    filtered ? 0.01 + random.nextDouble() * 0.5 : 1.0
                );
                final float[] query = query(random, 4);
                final int[] baseline = CentroidPlanner.plan(
                    f.clusters(),
                    query,
                    PlanParams.of(f.clusters().numClusters()),
                    k,
                    filtered ? f.acceptDocs() : null,
                    () -> {
                        throw new AssertionError("dense");
                    }
                );
                final FilterDigest digest = filtered ? f.digest() : null;
                final ShardPlanner.Plan plan = ShardPlanner.plan(
                    List.of(new ShardPlanner.Segment(f.clusters(), digest)),
                    query,
                    k,
                    ShardPlanner.Budget.RULE
                );
                assertArrayEquals(
                    baseline,
                    CentroidPlanner.planWithin(f.clusters(), query, plan.segmentParams()[0], digest),
                    "seed " + seed + " m " + m
                );
            }
        } finally {
            KneeExperiment.reset();
            ShardPlanExperiment.reset();
            ProbeCurveExperiment.reset();
        }
    }

    // ---------------------------------------------------------------- segment re-rank == shard count

    /**
     * The equivalence the counts rest on: a segment re-ranking its own centroids within (min_s, max_s) probes exactly
     * the clusters the shard counted for it, in the same order. Exercised with duplicated centroids (ties within and
     * across segments), filters, both metrics, both budgets, k on either side of what the shard holds, and one-segment
     * shards.
     */
    @ParameterizedTest
    @ValueSource(booleans = { false, true })
    void testPlanWithin_thenProbesExactlyTheClustersTheShardCounted(final boolean filtered) throws IOException {
        for (int seed = 0; seed < 60; seed++) {
            final Random random = new Random(seed);
            final VectorSimilarityFunction similarity = seed % 2 == 0
                ? VectorSimilarityFunction.EUCLIDEAN
                : VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
            final int k = new int[] { 1, 10, 100, 1000 }[seed % 4];
            final int segmentCount = seed % 5 == 0 ? 1 : 2 + random.nextInt(5);
            final float[][] shared = randomCentroids(random, 6, 3);
            final List<ShardPlanner.Segment> segments = new ArrayList<>();
            for (int s = 0; s < segmentCount; s++) {
                final Fixture f = fixture(
                    random,
                    similarity,
                    random.nextInt(120),
                    3,
                    filtered ? 0.01 + random.nextDouble() * 0.2 : 1.0,
                    shared
                );
                // Filtered shards mix in segments with nothing filtered out (no digest), as deletions in some do.
                segments.add(new ShardPlanner.Segment(f.clusters(), filtered && (s == 0 || random.nextBoolean()) ? f.digest() : null));
            }
            final float[] query = random.nextBoolean() ? shared[0].clone() : query(random, 3);
            for (final ShardPlanner.Budget budget : ShardPlanner.Budget.values()) {
                final ShardPlanner.Ranked planned = ShardPlanner.planRanked(segments, query, k, budget);
                for (int s = 0; s < segmentCount; s++) {
                    final PlanParams params = planned.plan().segmentParams()[s];
                    assertTrue(params.minProbes() <= params.maxProbes());
                    final int[] counted = Arrays.copyOf(planned.rankings()[s].ords(), params.maxProbes());
                    final int[] probed = CentroidPlanner.planWithin(segments.get(s).clusters(), query, params, segments.get(s).digest());
                    assertArrayEquals(counted, probed, "seed " + seed + " budget " + budget + " segment " + s);
                }
            }
        }
    }

    /** A one-segment shard under H probes exactly what today's per-segment plan probes, filtered or not. */
    @ParameterizedTest
    @ValueSource(booleans = { false, true })
    void testPlan_whenOneSegment_thenSameProbesAsToday(final boolean filtered) throws IOException {
        for (int seed = 0; seed < 40; seed++) {
            final Random random = new Random(500 + seed);
            final int k = new int[] { 1, 10, 100 }[seed % 3];
            final Fixture f = fixture(
                random,
                VectorSimilarityFunction.EUCLIDEAN,
                1 + random.nextInt(300),
                4,
                filtered ? 0.01 + random.nextDouble() * 0.5 : 1.0
            );
            final float[] query = query(random, 4);
            final int[] today = CentroidPlanner.plan(
                f.clusters(),
                query,
                PlanParams.of(f.clusters().numClusters()),
                k,
                filtered ? f.acceptDocs() : null,
                () -> {
                    throw new AssertionError("dense");
                }
            );
            final FilterDigest digest = filtered ? f.digest() : null;
            final ShardPlanner.Plan plan = ShardPlanner.plan(
                List.of(new ShardPlanner.Segment(f.clusters(), digest)),
                query,
                k,
                ShardPlanner.Budget.CAPPED
            );
            assertArrayEquals(today, CentroidPlanner.planWithin(f.clusters(), query, plan.segmentParams()[0], digest), "seed " + seed);
        }
    }

    /** planWithin with no knee and no hold: maxProbes is the count, even when it holds fewer than k. */
    @Test
    void testPlanWithin_whenZeroOrSome_thenExactlyThatMany() throws IOException {
        final Clusters clusters = line(VectorSimilarityFunction.EUCLIDEAN, 0, 1, 20);
        assertEquals(0, CentroidPlanner.planWithin(clusters, ORIGIN, new PlanParams(0, 0), null).length);
        assertArrayEquals(new int[] { 0, 1, 2 }, CentroidPlanner.planWithin(clusters, ORIGIN, new PlanParams(0, 3), null));
        assertEquals(20, CentroidPlanner.planWithin(clusters, ORIGIN, new PlanParams(1, 50), null).length);
    }

    /** The per-segment planner, handed a zero ceiling, plans nothing rather than failing to build its heap. */
    @Test
    void testPlan_whenTheCeilingIsZero_thenNoProbes() throws IOException {
        final Clusters clusters = line(VectorSimilarityFunction.EUCLIDEAN, 0, 1, 20);
        assertEquals(0, CentroidPlanner.plan(clusters, ORIGIN, new PlanParams(0, 0)).length);
    }

    // ---------------------------------------------------------------- fixtures

    private record Fixture(Clusters clusters, FilterDigest digest, AcceptDocs acceptDocs) {
    }

    private Fixture fixture(Random random, VectorSimilarityFunction similarity, int numClusters, int dimension, double selectivity)
        throws IOException {
        return fixture(random, similarity, numClusters, dimension, selectivity, null);
    }

    /**
     * A segment of {@code numClusters} random centroids (some copied from {@code shared}, to force ties), 1 to 40
     * vectors assigned to each (some left empty), and a random filter at about {@code selectivity}.
     */
    private Fixture fixture(
        Random random,
        VectorSimilarityFunction similarity,
        int numClusters,
        int dimension,
        double selectivity,
        float[][] shared
    ) throws IOException {
        final float[][] centroids = randomCentroids(random, numClusters, dimension);
        if (shared != null) {
            for (int c = 0; c < numClusters; c++) {
                if (random.nextInt(4) == 0) {
                    centroids[c] = shared[random.nextInt(shared.length)].clone();
                }
            }
        }
        final int[] sizes = new int[numClusters];
        final List<Integer> assignment = new ArrayList<>();
        for (int c = 0; c < numClusters; c++) {
            sizes[c] = random.nextInt(10) == 0 ? 0 : 1 + random.nextInt(40);
            for (int i = 0; i < sizes[c]; i++) {
                assignment.add(c);
            }
        }
        final int numVectors = assignment.size();
        final FixedBitSet accepted = new FixedBitSet(Math.max(1, numVectors));
        for (int doc = 0; doc < numVectors; doc++) {
            if (random.nextDouble() < selectivity) {
                accepted.set(doc);
            }
        }
        final Clusters clusters = clusters(similarity, centroids, sizes);
        when(clusters.numVectors()).thenReturn(numVectors);
        when(clusters.hasDenseVectorValues()).thenReturn(true);
        final int[] ordToCentroid = assignment.stream().mapToInt(Integer::intValue).toArray();
        when(clusters.ordToCentroid()).thenReturn(new LongValues() {
            @Override
            public long get(long ord) {
                return ordToCentroid[(int) ord];
            }
        });
        final AcceptDocs acceptDocs = AcceptDocs.fromLiveDocs(accepted, Math.max(1, numVectors));
        final FilterDigest digest = numClusters == 0
            ? new FilterDigest(new FixedBitSet(1), new int[1], 0, 0)
            : FilterDigest.of(clusters, acceptDocs, () -> {
                throw new AssertionError("dense");
            });
        return new Fixture(clusters, digest, acceptDocs);
    }

    private static int eligible(List<ShardPlanner.Segment> segments) {
        int total = 0;
        for (final ShardPlanner.Segment segment : segments) {
            total += segment.digest().eligibleClusters();
        }
        return total;
    }

    private static float[][] randomCentroids(Random random, int n, int dimension) {
        final float[][] centroids = new float[n][dimension];
        for (final float[] centroid : centroids) {
            for (int d = 0; d < dimension; d++) {
                centroid[d] = random.nextFloat() * 2 - 1;
            }
        }
        return centroids;
    }

    private static float[] query(Random random, int dimension) {
        return randomCentroids(random, 1, dimension)[0];
    }

    /** {@code count} centroids along the x axis at {@code start, start + step, ...}, each holding 10 vectors. */
    private Clusters line(VectorSimilarityFunction similarity, float start, float step, int count) throws IOException {
        final float[][] centroids = new float[count][];
        final int[] sizes = new int[count];
        for (int i = 0; i < count; i++) {
            centroids[i] = new float[] { start + i * step, 0f };
            sizes[i] = 10;
        }
        final Clusters clusters = clusters(similarity, centroids, sizes);
        when(clusters.numVectors()).thenReturn(10 * count);
        return clusters;
    }

    private static Clusters emptyClusters() {
        final Clusters clusters = mock(Clusters.class);
        when(clusters.numClusters()).thenReturn(0);
        when(clusters.numVectors()).thenReturn(0);
        return clusters;
    }

    private static CentroidPlanner.Ranked ranked(float[] keys) {
        final int[] ords = new int[keys.length];
        for (int i = 0; i < ords.length; i++) {
            ords[i] = i;
        }
        return new CentroidPlanner.Ranked(ords, keys);
    }

    private static List<ShardPlanner.Segment> unfilteredSegments(int n) {
        final List<ShardPlanner.Segment> segments = new ArrayList<>();
        for (int s = 0; s < n; s++) {
            final Clusters clusters = mock(Clusters.class);
            when(clusters.clusterSize(org.mockito.ArgumentMatchers.anyInt())).thenReturn(10);
            segments.add(new ShardPlanner.Segment(clusters, null));
        }
        return segments;
    }

    private static ShardPlanner.Segment filteredSegment(int[] counts) {
        final FixedBitSet eligible = new FixedBitSet(counts.length);
        eligible.set(0, counts.length);
        int accepted = 0;
        for (final int c : counts) {
            accepted += c;
        }
        return new ShardPlanner.Segment(mock(Clusters.class), new FilterDigest(eligible, counts, accepted, counts.length));
    }

    private Clusters clusters(VectorSimilarityFunction similarity, float[][] centroids, int[] clusterSizes) throws IOException {
        final Clusters clusters = mock(Clusters.class);
        when(clusters.numClusters()).thenReturn(centroids.length);
        when(clusters.clusterMeta()).thenReturn(fieldMeta(similarity, centroids.length, clusterSizes));
        if (centroids.length > 0) {
            final CentroidVectorValues base = centroidValues(centroids);
            when(clusters.centroids()).thenAnswer(invocation -> base.copy());
        }
        for (int ordinal = 0; ordinal < clusterSizes.length; ordinal++) {
            when(clusters.clusterSize(ordinal)).thenReturn(clusterSizes[ordinal]);
        }
        return clusters;
    }

    /** The centroid region as the format lays it out: the vector, then its squared norm. */
    private CentroidVectorValues centroidValues(float[][] centroids) throws IOException {
        final Directory directory = new ByteBuffersDirectory();
        directories.add(directory);
        try (IndexOutput out = directory.createOutput("clac", IOContext.DEFAULT)) {
            for (final float[] centroid : centroids) {
                for (final float value : centroid) {
                    out.writeInt(Float.floatToIntBits(value));
                }
                out.writeInt(Float.floatToIntBits(VectorUtil.dotProduct(centroid, centroid)));
            }
        }
        final IndexInput input = directory.openInput("clac", IOContext.DEFAULT);
        return new CentroidVectorValues(input, centroids.length, centroids[0].length);
    }

    private static ClusterANNFieldMeta fieldMeta(VectorSimilarityFunction similarity, int centroidCount, int[] clusterSizes) {
        int vectorCount = 0;
        for (final int size : clusterSizes) {
            vectorCount += size;
        }
        return new ClusterANNFieldMeta(
            32,
            2,
            vectorCount,
            centroidCount,
            similarity,
            1,
            ClusterANNFormatConstants.ROTATION_NONE,
            0,
            new byte[0],
            0L,
            0L,
            0L,
            -1L,
            0L,
            0L,
            new long[centroidCount],
            new int[centroidCount],
            clusterSizes,
            -1L,
            -1L,
            null
        );
    }
}
