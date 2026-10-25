package basalt.ml;

import basalt.core.Matrix;
import java.util.Objects;
import java.util.PriorityQueue;

/** Uniform-weight KNN regression with SIMD distances. */
public final class KNeighborsRegressor implements Estimator {
    private final int k;
    private final DistanceMetric metric;
    private Matrix trainingFeatures;
    private double[] trainingTargets;

    private KNeighborsRegressor(Builder builder) {
        k = builder.k;
        metric = builder.metric;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public KNeighborsRegressor fit(Matrix features, double[] targets) {
        MlChecks.trainingData(features, targets);
        if (k > features.rows()) {
            throw new IllegalArgumentException("k cannot exceed the number of training samples");
        }
        trainingFeatures = features.copy();
        trainingTargets = targets.clone();
        return this;
    }

    @Override
    public double[] predict(Matrix features) {
        MlChecks.predictionData(features, trainingFeatures == null ? 0 : trainingFeatures.columns(),
                trainingFeatures != null);
        int columns = trainingFeatures.columns();
        double[] train = trainingFeatures.data();
        double[] result = new double[features.rows()];
        for (int row = 0; row < features.rows(); row++) {
            PriorityQueue<Neighbor> nearest = new PriorityQueue<>(k,
                    (left, right) -> Double.compare(right.distance, left.distance));
            int offset = row * columns;
            for (int sample = 0; sample < trainingFeatures.rows(); sample++) {
                double distance = metric.distance(features.data(), offset,
                        train, sample * columns, columns);
                nearest.offer(new Neighbor(distance, trainingTargets[sample]));
                if (nearest.size() > k) {
                    nearest.poll();
                }
            }
            double sum = 0.0;
            for (Neighbor neighbor : nearest) {
                sum += neighbor.target;
            }
            result[row] = sum / nearest.size();
        }
        return result;
    }

    private record Neighbor(double distance, double target) {
    }

    public static final class Builder {
        private int k = 5;
        private DistanceMetric metric = DistanceMetric.EUCLIDEAN;

        public Builder k(int k) {
            this.k = k;
            return this;
        }

        public Builder metric(DistanceMetric metric) {
            this.metric = Objects.requireNonNull(metric, "metric");
            return this;
        }

        public KNeighborsRegressor build() {
            if (k <= 0) {
                throw new IllegalArgumentException("k must be positive");
            }
            return new KNeighborsRegressor(this);
        }
    }
}
