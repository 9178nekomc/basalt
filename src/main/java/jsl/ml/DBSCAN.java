package jsl.ml;

import jsl.core.Matrix;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.Objects;

/**
 * Density-based spatial clustering with the classic DBSCAN definition.
 *
 * <p>Points are core points when their {@code eps}-neighborhood contains at least
 * {@code minPoints} samples. Clusters grow by breadth-first expansion from core points; border
 * points join the first cluster that reaches them, and unreachable points are labelled -1.
 * The naive region query is {@code O(n^2)} and intended for small/medium dense datasets.</p>
 */
public final class DBSCAN implements Estimator {
    private final double eps;
    private final int minPoints;
    private final DistanceMetric metric;
    private Matrix data;
    private int[] labels;
    private boolean[] core;
    private int clusterCount;

    private DBSCAN(Builder builder) {
        eps = builder.eps;
        minPoints = builder.minPoints;
        metric = builder.metric;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public DBSCAN fit(Matrix features, double[] ignoredTargets) {
        MlChecks.features(features);
        data = features.copy();
        int rows = features.rows();
        labels = new int[rows];
        core = new boolean[rows];
        Arrays.fill(labels, -1);
        boolean[] visited = new boolean[rows];
        clusterCount = 0;
        int[] neighbors = new int[rows];
        for (int point = 0; point < rows; point++) {
            if (visited[point]) {
                continue;
            }
            visited[point] = true;
            int count = collectNeighbors(point, neighbors);
            if (count < minPoints) {
                labels[point] = -1;
                continue;
            }
            core[point] = true;
            int cluster = clusterCount++;
            labels[point] = cluster;
            Deque<Integer> queue = new ArrayDeque<>();
            queue.add(point);
            while (!queue.isEmpty()) {
                int current = queue.poll();
                int currentCount = current == point ? count : collectNeighbors(current, neighbors);
                if (currentCount >= minPoints) {
                    core[current] = true;
                    for (int index = 0; index < currentCount; index++) {
                        int neighbor = neighbors[index];
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            labels[neighbor] = cluster;
                            queue.add(neighbor);
                        } else if (labels[neighbor] < 0) {
                            labels[neighbor] = cluster;
                        }
                    }
                }
            }
        }
        return this;
    }

    private int collectNeighbors(int point, int[] buffer) {
        int count = 0;
        int columns = data.columns();
        double[] values = data.data();
        int offset = point * columns;
        for (int other = 0; other < data.rows(); other++) {
            if (metric.distance(values, offset, values, other * columns, columns) <= eps) {
                buffer[count++] = other;
            }
        }
        return count;
    }

    @Override
    public double[] predict(Matrix features) {
        if (data == null) {
            throw new IllegalStateException("DBSCAN must be fitted first");
        }
        if (features.columns() != data.columns()) {
            throw new IllegalArgumentException("Feature count mismatch");
        }
        double[] result = new double[features.rows()];
        int columns = data.columns();
        double[] values = data.data();
        for (int row = 0; row < features.rows(); row++) {
            int offset = row * columns;
            int best = -1;
            double bestDistance = Double.POSITIVE_INFINITY;
            for (int point = 0; point < data.rows(); point++) {
                if (!core[point]) {
                    continue;
                }
                double distance = metric.distance(features.data(), offset,
                        values, point * columns, columns);
                if (distance <= eps && distance < bestDistance) {
                    bestDistance = distance;
                    best = labels[point];
                }
            }
            result[row] = best;
        }
        return result;
    }

    public int[] labels() {
        return labels == null ? null : labels.clone();
    }

    public int clusterCount() {
        return clusterCount;
    }

    public int noiseCount() {
        int count = 0;
        for (int label : labels) {
            if (label < 0) {
                count++;
            }
        }
        return count;
    }

    public static final class Builder {
        private double eps = 0.5;
        private int minPoints = 5;
        private DistanceMetric metric = DistanceMetric.EUCLIDEAN;

        public Builder eps(double eps) {
            this.eps = eps;
            return this;
        }

        public Builder minPoints(int minPoints) {
            this.minPoints = minPoints;
            return this;
        }

        public Builder metric(DistanceMetric metric) {
            this.metric = Objects.requireNonNull(metric, "metric");
            return this;
        }

        public DBSCAN build() {
            MlChecks.positiveFinite(eps, "eps");
            if (minPoints <= 0) {
                throw new IllegalArgumentException("minPoints must be positive");
            }
            return new DBSCAN(this);
        }
    }
}
