/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clusterann.component.corpus;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Runs {@link Hdf5Sampler} through the build, since the build gives tests a classpath and a JVM. Not a test of
 * anything: it is skipped unless told what to sample, so it only does work when run by name. {@code clusterAnnTest}
 * passes every {@code -Dknn.*} property through to the forked test JVM:
 *
 * <pre>
 * # the four standard files (SIFT L2 and GloVe cosine, 1k and 10k rows each), downloading the sources on a miss
 * ./gradlew clusterAnnTest --tests '*Hdf5SamplerRun' -Dknn.sample.standard=true
 *
 * # one sample of your own
 * ./gradlew clusterAnnTest --tests '*Hdf5SamplerRun' -Dknn.sample.source=src/test/resources/dataset/documents-1k.hdf5 \
 *     -Dknn.sample.size=200 -Dknn.sample.queries=50 -Dknn.sample.depth=10 -Dknn.sample.similarity=maximum_inner_product
 * </pre>
 *
 * <p>k-NN: skipped through {@link EnabledIf} rather than formats' {@code assumeTrue}, which needs opentest4j on the
 * test compile classpath, and k-NN's test dependencies do not put it there.
 */
class Hdf5SamplerRun {

    /** Whether the run was told what to sample; the condition {@link #sample()} is enabled by. */
    static boolean configured() {
        final Map<String, String> properties = Hdf5Sampler.systemProperties();
        final String source = properties.get(Hdf5Sampler.PREFIX + "source");
        final boolean hasSource = source != null && !source.isBlank();
        return hasSource || Boolean.parseBoolean(properties.getOrDefault(Hdf5Sampler.PREFIX + "standard", "false"));
    }

    @Test
    @EnabledIf("configured")
    void sample() {
        final Map<String, String> properties = Hdf5Sampler.systemProperties();
        final String source = properties.get(Hdf5Sampler.PREFIX + "source");
        final boolean hasSource = source != null && !source.isBlank();

        if (hasSource) {
            final Path out = Hdf5Sampler.sample(Hdf5Sampler.Options.parse(properties));
            assertTrue(Files.exists(out), "sampler returned without writing " + out);
            return;
        }

        final List<Path> written = Hdf5Sampler.sampleStandard();
        for (Hdf5Sampler.Options options : Hdf5Sampler.STANDARD) {
            assertTrue(Files.exists(options.out()), "standard sample missing after the run: " + options.out());
        }
        assertTrue(written.size() <= Hdf5Sampler.STANDARD.size());
    }
}
