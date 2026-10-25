package jsl.ml;

import jsl.core.Matrix;
import java.util.Arrays;

/** Binary logistic regression trained with numerically stable cross-entropy. */
public final class LogisticRegression implements Estimator {
    private final double learningRate;
    private final int epochs;
    private final double l2;
    private double[] weights;
    private double intercept;
    private int featureCount;
    private double[] lossHistory = new double[0];
    private boolean fitted;

    private LogisticRegression(Builder builder) {
        this.learningRate = builder.learningRate;
        this.epochs = builder.epochs;
        this.l2 = builder.l2;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public LogisticRegression fit(Matrix features, double[] targets) {
        MlChecks.trainingData(features, targets);
        for (double target : targets) {
            if (target != 0.0 && target != 1.0) {
                throw new IllegalArgumentException("Binary logistic targets must be 0 or 1");
            }
        }
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
                int offset = row * featureCount;
                double logit = intercept;
                for (int feature = 0; feature < featureCount; feature++) {
                    logit += weights[feature] * x[offset + feature];
                }
                double probability = sigmoid(logit);
                double error = probability - targets[row];
                loss += Math.max(logit, 0.0) - logit * targets[row]
                        + Math.log1p(Math.exp(-Math.abs(logit)));
                for (int feature = 0; feature < featureCount; feature++) {
                    gradient[feature] += error * x[offset + feature];
                }
                interceptGradient += error;
            }
            for (int feature = 0; feature < featureCount; feature++) {
                gradient[feature] = gradient[feature] / samples + 2.0 * l2 * weights[feature];
                weights[feature] -= learningRate * gradient[feature];
            }
            intercept -= learningRate * interceptGradient / samples;
            lossHistory[epoch] = loss / samples;
        }
        fitted = true;
        return this;
    }

    private static double sigmoid(double value) {
        if (value >= 0.0) {
            return 1.0 / (1.0 + Math.exp(-value));
        }
        double exponential = Math.exp(value);
        return exponential / (1.0 + exponential);
    }

    public double[] predictProbability(Matrix features) {
        MlChecks.predictionData(features, featureCount, fitted);
        double[] probabilities = new double[features.rows()];
        for (int row = 0; row < features.rows(); row++) {
            int offset = row * featureCount;
            double logit = intercept;
            for (int feature = 0; feature < featureCount; feature++) {
                logit += weights[feature] * features.data()[offset + feature];
            }
            probabilities[row] = sigmoid(logit);
        }
        return probabilities;
    }

    @Override
    public double[] predict(Matrix features) {
        double[] probabilities = predictProbability(features);
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = probabilities[i] >= 0.5 ? 1.0 : 0.0;
        }
        return probabilities;
    }

    public double[] lossHistory() {
        return lossHistory.clone();
    }

    public static final class Builder {
        private double learningRate = 0.1;
        private int epochs = 2_000;
        private double l2;

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

        public LogisticRegression build() {
            MlChecks.positiveFinite(learningRate, "learningRate");
            MlChecks.nonNegativeFinite(l2, "l2");
            if (epochs <= 0) {
                throw new IllegalArgumentException("epochs must be positive");
            }
            return new LogisticRegression(this);
        }
    }
}
