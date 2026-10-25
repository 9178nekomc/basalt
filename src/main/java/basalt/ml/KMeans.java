package basalt.ml;

import basalt.core.Matrix;
import basalt.core.NativeKernels;
import basalt.core.ModelFormatException;
import basalt.core.ModelKind;
import basalt.core.ModelReader;
import basalt.core.ModelWriter;
import basalt.core.Persistable;
import basalt.core.VectorMath;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * Lloyd iterations with genuine K-Means++ distance-weighted seeding and optional restarts.
 *
 * <p>The assignment step is the dominant cost ({@code O(n k p)}); it is distributed over row ranges
 * while centroid updates stay sequential and allocation free. Empty clusters are repaired by
 * donating the farthest sample from the largest donor cluster, which preserves the partition.</p>
 */
public final class KMeans implements Estimator, Persistable {
    private final int clusters;
    private final int maxIterations;
    private final double tolerance;
    private final long seed;
    private final int restarts;
    private final int threads;
    private double[] centroids;
    private int featureCount;
    private double inertia;
    private int iterations;
    private int completedRestarts;

    private KMeans(Builder builder) {
        clusters = builder.clusters;
        maxIterations = builder.maxIterations;
        tolerance = builder.tolerance;
        seed = builder.seed;
        restarts = builder.restarts;
        threads = builder.threads;
    }

    public static Builder builder() {
        return new Builder();
    }

    public KMeans fit(Matrix features) {
        return fit(features, null);
    }

    @Override
    public KMeans fit(Matrix features, double[] ignoredTargets) {
        MlChecks.features(features);
        if (clusters > features.rows()) {
            throw new IllegalArgumentException("Cluster count cannot exceed sample count");
        }
        featureCount = features.columns();
        if (NativeKernels.available() && features.rows() >= 1_024 && featureCount >= 8) {
            double[] nativeCenters = new double[clusters * featureCount];
            double[] nativeInertia = new double[1];
            int[] nativeIterations = new int[1];
            if (NativeKernels.kmeansFitChecked(features.data(), features.rows(), clusters,
                    featureCount, restarts, maxIterations, tolerance, seed, threads,
                    nativeCenters, nativeInertia, nativeIterations)) {
                centroids = nativeCenters;
                inertia = nativeInertia[0];
                iterations = nativeIterations[0];
                completedRestarts = restarts;
                return this;
            }
        }
        ForkJoinPool pool = new ForkJoinPool(threads);
        try {
            double bestInertia = Double.POSITIVE_INFINITY;
            double[] bestCentroids = null;
            int bestIterations = 0;
            for (int restart = 0; restart < restarts; restart++) {
                Random random = new Random(seed + restart * 7_919L);
                Run run = runOnce(features, random, pool);
                if (run.inertia < bestInertia) {
                    bestInertia = run.inertia;
                    bestCentroids = run.centroids;
                    bestIterations = run.iterations;
                }
            }
            centroids = bestCentroids;
            inertia = bestInertia;
            iterations = bestIterations;
            completedRestarts = restarts;
        } finally {
            pool.shutdown();
        }
        return this;
    }

    private Run runOnce(Matrix features, Random random, ForkJoinPool pool) {
        double[] centers = initializePlusPlus(features.data(), features.rows(), random);
        double[] next = new double[centers.length];
        int[] assignments = new int[features.rows()];
        double[] distances = new double[features.rows()];
        int[] counts = new int[clusters];
        double[] data = features.data();
        int rows = features.rows();
        int completed = 0;

        for (int iteration = 1; iteration <= maxIterations; iteration++) {
            completed = iteration;
            final double[] current = centers;
            boolean nativeDone = NativeKernels.available() && featureCount >= 128
                    && NativeKernels.kmeansAssignChecked(data, current, rows, clusters,
                    featureCount, threads, assignments, distances);
            if (!nativeDone) {
                pool.submit(() -> IntStream.range(0, rows).parallel().forEach(row -> {
                    int offset = row * featureCount;
                    int best = 0;
                    double bestDistance = Double.POSITIVE_INFINITY;
                    for (int cluster = 0; cluster < clusters; cluster++) {
                        double distance = VectorMath.squaredEuclidean(data, offset,
                                current, cluster * featureCount, featureCount);
                        if (distance < bestDistance) {
                            bestDistance = distance;
                            best = cluster;
                        }
                    }
                    assignments[row] = best;
                    distances[row] = bestDistance;
                })).join();
            }
            Arrays.fill(next, 0.0);
            Arrays.fill(counts, 0);
            for (int row = 0; row < rows; row++) {
                int cluster = assignments[row];
                counts[cluster]++;
                int source = row * featureCount;
                int target = cluster * featureCount;
                for (int feature = 0; feature < featureCount; feature++) {
                    next[target + feature] += data[source + feature];
                }
            }
            boolean[] reserved = new boolean[rows];
            for (int cluster = 0; cluster < clusters; cluster++) {
                if (counts[cluster] != 0) {
                    continue;
                }
                int farthest = selectEmptyClusterDonor(distances, assignments, counts, reserved);
                reserved[farthest] = true;
                int donor = assignments[farthest];
                int targetOffset = cluster * featureCount;
                int sourceOffset = farthest * featureCount;
                if (counts[donor] > 1) {
                    counts[donor]--;
                    for (int feature = 0; feature < featureCount; feature++) {
                        next[donor * featureCount + feature] -= data[sourceOffset + feature];
                    }
                    assignments[farthest] = cluster;
                }
                System.arraycopy(features.data(), sourceOffset, next, targetOffset, featureCount);
                counts[cluster] = 1;
                distances[farthest] = 0.0;
            }
            double shift = 0.0;
            for (int cluster = 0; cluster < clusters; cluster++) {
                for (int feature = 0; feature < featureCount; feature++) {
                    int index = cluster * featureCount + feature;
                    next[index] /= counts[cluster];
                    double delta = next[index] - centers[index];
                    shift += delta * delta;
                }
            }
            double[] temporary = centers;
            centers = next;
            next = temporary;
            if (shift <= tolerance * tolerance) {
                break;
            }
        }
        double total = 0.0;
        for (int row = 0; row < rows; row++) {
            int cluster = nearest(data, row * featureCount, centers);
            total += VectorMath.squaredEuclidean(data, row * featureCount,
                    centers, cluster * featureCount, featureCount);
        }
        return new Run(centers, total, completed);
    }

    private double[] initializePlusPlus(double[] values, int rows, Random random) {
        double[] centers = new double[clusters * featureCount];
        double[] distances = new double[rows];
        Arrays.fill(distances, Double.POSITIVE_INFINITY);
        int first = random.nextInt(rows);
        System.arraycopy(values, first * featureCount, centers, 0, featureCount);
        for (int cluster = 1; cluster < clusters; cluster++) {
            double sum = 0.0;
            for (int row = 0; row < rows; row++) {
                double distance = VectorMath.squaredEuclidean(values, row * featureCount,
                        centers, (cluster - 1) * featureCount, featureCount);
                distances[row] = Math.min(distances[row], distance);
                sum += distances[row];
            }
            int selected;
            if (sum == 0.0) {
                selected = cluster % rows;
            } else {
                double threshold = random.nextDouble() * sum;
                selected = rows - 1;
                double cumulative = 0.0;
                for (int row = 0; row < rows; row++) {
                    cumulative += distances[row];
                    if (cumulative >= threshold) {
                        selected = row;
                        break;
                    }
                }
            }
            System.arraycopy(values, selected * featureCount,
                    centers, cluster * featureCount, featureCount);
        }
        return centers;
    }

    private int nearest(double[] values, int offset, double[] centers) {
        int best = 0;
        double bestDistance = Double.POSITIVE_INFINITY;
        for (int cluster = 0; cluster < clusters; cluster++) {
            double distance = VectorMath.squaredEuclidean(values, offset,
                    centers, cluster * featureCount, featureCount);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = cluster;
            }
        }
        return best;
    }

    private static int selectEmptyClusterDonor(
            double[] distances, int[] assignments, int[] counts, boolean[] reserved) {
        int best = -1;
        for (int row = 0; row < distances.length; row++) {
            if (!reserved[row] && counts[assignments[row]] > 1
                    && (best < 0 || distances[row] > distances[best])) {
                best = row;
            }
        }
        if (best >= 0) {
            return best;
        }
        for (int row = 0; row < distances.length; row++) {
            if (!reserved[row] && (best < 0 || distances[row] > distances[best])) {
                best = row;
            }
        }
        if (best < 0) {
            throw new IllegalStateException("Unable to select an empty-cluster donor");
        }
        return best;
    }

    @Override
    public double[] predict(Matrix features) {
        MlChecks.predictionData(features, featureCount, centroids != null);
        double[] result = new double[features.rows()];
        for (int row = 0; row < features.rows(); row++) {
            result[row] = nearest(features.data(), row * featureCount, centroids);
        }
        return result;
    }

    /** Squared distance to the assigned centroid for each row. */
    public double[] transform(Matrix features) {
        MlChecks.predictionData(features, featureCount, centroids != null);
        double[] result = new double[features.rows()];
        for (int row = 0; row < features.rows(); row++) {
            int cluster = nearest(features.data(), row * featureCount, centroids);
            result[row] = VectorMath.squaredEuclidean(features.data(), row * featureCount,
                    centroids, cluster * featureCount, featureCount);
        }
        return result;
    }

    public Matrix centroids() {
        if (centroids == null) {
            throw new IllegalStateException("KMeans must be fitted first");
        }
        return new Matrix(clusters, featureCount, centroids);
    }

    public double inertia() {
        return inertia;
    }

    public int iterations() {
        return iterations;
    }

    public int restarts() {
        return completedRestarts;
    }

    @Override
    public ModelKind kind() {
        return ModelKind.KMEANS;
    }

    @Override
    public void write(ModelWriter writer) {
        if (centroids == null) {
            throw new IllegalStateException("KMeans must be fitted before serialization");
        }
        writer.putInt(clusters);
        writer.putInt(featureCount);
        writer.putDouble(inertia);
        writer.putInt(iterations);
        writer.putDoubles(centroids);
    }

    public static KMeans read(ModelReader reader) {
        int clusters = reader.getInt();
        int features = reader.getInt();
        double inertia = reader.getDouble();
        int iterations = reader.getInt();
        if (clusters <= 0 || features <= 0) {
            throw new ModelFormatException("Invalid KMeans header");
        }
        double[] centroids = reader.getDoubles(Math.multiplyExact(clusters, features));
        KMeans model = new KMeans(new Builder().clusters(clusters));
        model.featureCount = features;
        model.centroids = centroids;
        model.inertia = inertia;
        model.iterations = iterations;
        return model;
    }

    private record Run(double[] centroids, double inertia, int iterations) {
    }

    public static final class Builder {
        private int clusters = 8;
        private int maxIterations = 300;
        private double tolerance = 1e-6;
        private long seed = 42;
        private int restarts = 1;
        private int threads = Runtime.getRuntime().availableProcessors();

        public Builder clusters(int clusters) {
            this.clusters = clusters;
            return this;
        }

        public Builder maxIterations(int maxIterations) {
            this.maxIterations = maxIterations;
            return this;
        }

        public Builder tolerance(double tolerance) {
            this.tolerance = tolerance;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder restarts(int restarts) {
            this.restarts = restarts;
            return this;
        }

        public Builder threads(int threads) {
            this.threads = threads;
            return this;
        }

        public KMeans build() {
            if (clusters <= 0 || maxIterations <= 0 || restarts <= 0 || threads <= 0) {
                throw new IllegalArgumentException(
                        "clusters, maxIterations, restarts and threads must be positive");
            }
            MlChecks.nonNegativeFinite(tolerance, "tolerance");
            return new KMeans(this);
        }
    }
}
