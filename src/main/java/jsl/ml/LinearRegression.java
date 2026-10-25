package jsl.ml;

import jsl.core.Matrix;
import java.util.Arrays;

/** Ordinary linear regression trained by full-batch gradient descent. */
public final class LinearRegression implements Estimator {
    private final double learningRate;
    private final int epochs;
    private final double l2;
    private final boolean fitIntercept;
    private double[] weights;
    private double intercept;
    private int featureCount;
    private double[] lossHistory = new double[0];
    private boolean fitted;

    private LinearRegression(Builder builder) {
        this.learningRate = builder.learningRate;
        this.epochs = builder.epochs;
        this.l2 = builder.l2;
        this.fitIntercept = builder.fitIntercept;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public LinearRegression fit(Matrix features, double[] targets) {
        MlChecks.trainingData(features, targets);
        int samples = features.rows();
        featureCount = features.columns();
        weights = new double[featureCount];
        intercept = 0.0;
        lossHistory = new double[epochs];
        double[] x = features.data();
        double[] gradient = new double[featureCount];

        for (int epoch = 0; epoch < epochs; epoch++) {
            Arrays.fill(gradient, 0.0);
            double interceptGradient = 0.0;
            double loss = 0.0;
            for (int row = 0; row < samples; row++) {
                double prediction = intercept;
                int offset = row * featureCount;
                for (int feature = 0; feature < featureCount; feature++) {
                    prediction += weights[feature] * x[offset + feature];
                }
                double error = prediction - targets[row];
                loss += error * error;
                for (int feature = 0; feature < featureCount; feature++) {
                    gradient[feature] += error * x[offset + feature];
                }
                interceptGradient += error;
            }
            double scale = 2.0 / samples;
            for (int feature = 0; feature < featureCount; feature++) {
                gradient[feature] = scale * gradient[feature] + 2.0 * l2 * weights[feature];
                weights[feature] -= learningRate * gradient[feature];
            }
            if (fitIntercept) {
                intercept -= learningRate * scale * interceptGradient;
            }
            lossHistory[epoch] = loss / samples;
        }
        fitted = true;
        return this;
    }

    @Override
    public double[] predict(Matrix features) {
        MlChecks.predictionData(features, featureCount, fitted);
        double[] predictions = new double[features.rows()];
        for (int row = 0; row < features.rows(); row++) {
            predictions[row] = predictOne(features.data(), row * featureCount);
        }
        return predictions;
    }

    public double predictOne(double[] features, int offset) {
        if (!fitted) {
            throw new IllegalStateException("Estimator must be fitted before prediction");
        }
        if (offset < 0 || offset + featureCount > features.length) {
            throw new IndexOutOfBoundsException("Insufficient feature values");
        }
        double result = intercept;
        for (int feature = 0; feature < featureCount; feature++) {
            result += weights[feature] * features[offset + feature];
        }
        return result;
    }

    public double[] weights() {
        return weights == null ? new double[0] : weights.clone();
    }

    public double intercept() {
        return intercept;
    }

    public double[] lossHistory() {
        return lossHistory.clone();
    }

    public static final class Builder {
        private double learningRate = 0.05;
        private int epochs = 2_000;
        private double l2;
        private boolean fitIntercept = true;

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder epochs(int epochs) {
            this.epochs = epochs;
            return this;
        }

        public Builder l2(double l2) {
            this.l2 = l2;
            return this;
        }

        public Builder fitIntercept(boolean fitIntercept) {
            this.fitIntercept = fitIntercept;
            return this;
        }

        public LinearRegression build() {
            MlChecks.positiveFinite(learningRate, "learningRate");
            MlChecks.nonNegativeFinite(l2, "l2");
            if (epochs <= 0) {
                throw new IllegalArgumentException("epochs must be positive");
            }
            return new LinearRegression(this);
        }
    }
}
