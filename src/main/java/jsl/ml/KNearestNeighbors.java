package jsl.ml;

import jsl.core.Matrix;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;

/** Deterministic, uniform-weight KNN classification with SIMD distance calculations. */
public final class KNearestNeighbors implements Estimator {
    private final int k;
    private final DistanceMetric metric;
    private Matrix trainingFeatures;
    private double[] trainingTargets;
    private boolean fitted;

    private KNearestNeighbors(Builder builder) {
        this.k = builder.k;
        this.metric = builder.metric;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public KNearestNeighbors fit(Matrix features, double[] targets) {
        MlChecks.trainingData(features, targets);
        if (k > features.rows()) {
            throw new IllegalArgumentException("k cannot exceed the number of training samples");
        }
        trainingFeatures = features.copy();
        trainingTargets = targets.clone();
        fitted = true;
        return this;
    }

    @Override
    public double[] predict(Matrix features) {
        MlChecks.predictionData(features, trainingFeatures == null ? 0 : trainingFeatures.columns(), fitted);
        int featureCount = trainingFeatures.columns();
        double[] result = new double[features.rows()];
        for (int row = 0; row < features.rows(); row++) {
            result[row] = predictOne(features.data(), row * featureCount);
        }
        return result;
    }

    private double predictOne(double[] sample, int sampleOffset) {
        PriorityQueue<Neighbor> nearest = new PriorityQueue<>(k,
                (left, right) -> Double.compare(right.distance, left.distance));
        double[] train = trainingFeatures.data();
        int featureCount = trainingFeatures.columns();
        for (int row = 0; row < trainingFeatures.rows(); row++) {
            double distance = metric.distance(sample, sampleOffset, train, row * featureCount, featureCount);
            nearest.offer(new Neighbor(distance, trainingTargets[row]));
            if (nearest.size() > k) {
                nearest.poll();
            }
        }
        Map<Double, Integer> votes = new HashMap<>();
        for (Neighbor neighbor : nearest) {
            votes.merge(neighbor.label, 1, Integer::sum);
        }
        double bestLabel = Double.NaN;
        int bestVotes = -1;
        for (Map.Entry<Double, Integer> vote : votes.entrySet()) {
            int count = vote.getValue();
            double label = vote.getKey();
            if (count > bestVotes || (count == bestVotes && label < bestLabel)) {
                bestLabel = label;
                bestVotes = count;
            }
        }
        return bestLabel;
    }

    private record Neighbor(double distance, double label) {
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

        public KNearestNeighbors build() {
            if (k <= 0) {
                throw new IllegalArgumentException("k must be positive");
            }
            return new KNearestNeighbors(this);
        }
    }
}
