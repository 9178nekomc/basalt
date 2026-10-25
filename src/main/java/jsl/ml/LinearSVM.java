package jsl.ml;

import jsl.core.Matrix;
import java.util.Random;
import java.util.TreeMap;

/**
 * Binary linear support vector machine trained with the PEGASOS sub-gradient method.
 *
 * <p>Objective: {@code lambda/2 * ||w||^2 + mean(max(0, 1 - y(w·x + b)))} with {@code y in {-1,1}}.
 * The step size is {@code 1/(lambda * t)}; the bias is unregularized. For linearly separable data
 * the hinge loss reaches zero and the model recovers a maximum-margin direction.</p>
 */
public final class LinearSVM implements Estimator {
    private final double lambda;
    private final int epochs;
    private final long seed;
    private double[] weights;
    private double bias;
    private double[] classes;
    private int featureCount;

    private LinearSVM(Builder builder) {
        lambda = builder.lambda;
        epochs = builder.epochs;
        seed = builder.seed;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public LinearSVM fit(Matrix features, double[] targets) {
        MlChecks.trainingData(features, targets);
        featureCount = features.columns();
        TreeMap<Double, Integer> classIndex = new TreeMap<>();
        for (double target : targets) {
            classIndex.putIfAbsent(target, 0);
        }
        if (classIndex.size() != 2) {
            throw new IllegalArgumentException("LinearSVM requires exactly two classes");
        }
        classes = new double[]{classIndex.firstKey(), classIndex.lastKey()};
        int rows = features.rows();
        double[] signs = new double[rows];
        for (int row = 0; row < rows; row++) {
            signs[row] = targets[row] == classes[1] ? 1.0 : -1.0;
        }
        weights = new double[featureCount];
        bias = 0.0;
        int[] order = new int[rows];
        for (int row = 0; row < rows; row++) {
            order[row] = row;
        }
        Random random = new Random(seed);
        long step = 1;
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = rows - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int temporary = order[i];
                order[i] = order[j];
                order[j] = temporary;
            }
            for (int index : order) {
                double margin = signs[index] * (score(features.data(), index * featureCount) );
                double eta = 1.0 / (lambda * step);
                double shrinkage = 1.0 - eta * lambda;
                if (margin < 1.0) {
                    for (int feature = 0; feature < featureCount; feature++) {
                        weights[feature] = shrinkage * weights[feature]
                                + eta * signs[index] * features.get(index, feature);
                    }
                    bias += eta * signs[index];
                } else {
                    for (int feature = 0; feature < featureCount; feature++) {
                        weights[feature] *= shrinkage;
                    }
                }
                step++;
            }
        }
        return this;
    }

    private double score(double[] values, int offset) {
        double sum = bias;
        for (int feature = 0; feature < featureCount; feature++) {
            sum += weights[feature] * values[offset + feature];
        }
        return sum;
    }

    @Override
    public double[] predict(Matrix features) {
        MlChecks.predictionData(features, featureCount, weights != null);
        double[] result = new double[features.rows()];
        for (int row = 0; row < features.rows(); row++) {
            result[row] = score(features.data(), row * featureCount) >= 0.0
                    ? classes[1] : classes[0];
        }
        return result;
    }

    public double[] weights() {
        return weights.clone();
    }

    public double bias() {
        return bias;
    }

    public static final class Builder {
        private double lambda = 0.01;
        private int epochs = 20;
        private long seed = 42;

        public Builder lambda(double lambda) {
            this.lambda = lambda;
            return this;
        }

        public Builder epochs(int epochs) {
            this.epochs = epochs;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public LinearSVM build() {
            MlChecks.positiveFinite(lambda, "lambda");
            if (epochs <= 0) {
                throw new IllegalArgumentException("epochs must be positive");
            }
            return new LinearSVM(this);
        }
    }
}
