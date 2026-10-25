package jsl.ml;

import jsl.core.Matrix;

/** L1-regularized least squares fitted by cyclic coordinate descent. */
public final class Lasso implements Estimator {
    private final double alpha;
    private final int maxIterations;
    private final double tolerance;
    private double[] weights;
    private double intercept;
    private int featureCount;

    private Lasso(Builder builder) {
        alpha = builder.alpha;
        maxIterations = builder.maxIterations;
        tolerance = builder.tolerance;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Lasso fit(Matrix features, double[] targets) {
        MlChecks.trainingData(features, targets);
        featureCount = features.columns();
        int rows = features.rows();
        weights = new double[featureCount];
        intercept = 0.0;
        for (double target : targets) {
            intercept += target;
        }
        intercept /= rows;
        double[] residual = new double[rows];
        for (int row = 0; row < rows; row++) {
            residual[row] = targets[row] - intercept;
        }
        double[] norms = new double[featureCount];
        for (int feature = 0; feature < featureCount; feature++) {
            double sum = 0.0;
            for (int row = 0; row < rows; row++) {
                double value = features.get(row, feature);
                sum += value * value;
            }
            norms[feature] = sum;
        }
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            double maximumChange = 0.0;
            for (int feature = 0; feature < featureCount; feature++) {
                if (norms[feature] == 0.0) {
                    continue;
                }
                double rho = 0.0;
                for (int row = 0; row < rows; row++) {
                    double value = features.get(row, feature);
                    rho += value * (residual[row] + weights[feature] * value);
                }
                double updated = softThreshold(rho, alpha) / norms[feature];
                double delta = updated - weights[feature];
                if (delta != 0.0) {
                    for (int row = 0; row < rows; row++) {
                        residual[row] -= delta * features.get(row, feature);
                    }
                    weights[feature] = updated;
                    maximumChange = Math.max(maximumChange, Math.abs(delta));
                }
            }
            double mean = 0.0;
            for (double value : residual) {
                mean += value;
            }
            mean /= rows;
            if (Math.abs(mean) > 0.0) {
                intercept += mean;
                for (int row = 0; row < rows; row++) {
                    residual[row] -= mean;
                }
            }
            if (maximumChange < tolerance) {
                break;
            }
        }
        return this;
    }

    private static double softThreshold(double value, double threshold) {
        if (value > threshold) {
            return value - threshold;
        }
        if (value < -threshold) {
            return value + threshold;
        }
        return 0.0;
    }

    @Override
    public double[] predict(Matrix features) {
        MlChecks.predictionData(features, featureCount, weights != null);
        double[] result = new double[features.rows()];
        for (int row = 0; row < features.rows(); row++) {
            double sum = intercept;
            for (int feature = 0; feature < featureCount; feature++) {
                sum += weights[feature] * features.get(row, feature);
            }
            result[row] = sum;
        }
        return result;
    }

    public double[] weights() {
        return weights.clone();
    }

    public double intercept() {
        return intercept;
    }

    public static final class Builder {
        private double alpha = 0.1;
        private int maxIterations = 1_000;
        private double tolerance = 1e-8;

        public Builder alpha(double alpha) {
            this.alpha = alpha;
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

        public Lasso build() {
            MlChecks.nonNegativeFinite(alpha, "alpha");
            if (maxIterations <= 0) {
                throw new IllegalArgumentException("maxIterations must be positive");
            }
            MlChecks.positiveFinite(tolerance, "tolerance");
            return new Lasso(this);
        }
    }
}
