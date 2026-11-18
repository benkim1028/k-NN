/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.component.suite;

import org.apache.lucene.index.VectorSimilarityFunction;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * One measurement's parameters, read from a suite file: which corpus, how to index it, how to search it, which codec.
 * {@code suite} is the file it was read from; several suites can load together and their scenario names may repeat.
 */
public record Scenario(Path suite, String name, CorpusSpec corpus, IndexSpec index, SearchSpec search, CodecSpec codec) {
    public record CorpusSpec(String type, int size, int queries, int dimension, int clusters,

        double spread, int depth, long seed, VectorSimilarityFunction similarity, String file) {
    }

    /**
     * {@code childrenPerParent} above 0 indexes the corpus the way OpenSearch indexes a nested field: each vector in
     * its own child document, written as a block of that many children followed by a parent that has no vector.
     */
    public record IndexSpec(int docsPerSegment, List<String> expectFiles, Integer forceMerge, int childrenPerParent) {
    }

    public record SearchSpec(int k, int queries, int repeat, int warmup, int threads, double scoreTolerance, double scoreBias) {
    }

    public record CodecSpec(String name, Map<String, Object> params) {
    }

    /** {@code <suite stem>:<name>}, e.g. {@code cohere-1k:1bit-flushed}; the test's display name. */
    @Override
    public String toString() {
        return suiteName() + ":" + name;
    }

    /** The suite file's name without {@code -suite.yml}/{@code .yml}, e.g. {@code cohere-1k}. */
    public String suiteName() {
        String stem = suite.getFileName().toString();
        if (stem.endsWith(".yml")) {
            stem = stem.substring(0, stem.length() - ".yml".length());
        }
        if (stem.endsWith("-suite")) {
            stem = stem.substring(0, stem.length() - "-suite".length());
        }
        return stem;
    }
}
