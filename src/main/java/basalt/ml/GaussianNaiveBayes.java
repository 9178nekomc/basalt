package basalt.ml;

import basalt.core.Matrix;
import basalt.core.ModelFormatException;
import basalt.core.ModelKind;
import basalt.core.ModelReader;
import basalt.core.ModelWriter;
import basalt.core.NativeKernels;
import basalt.core.Persistable;
import java.util.TreeMap;

/**
 * Gaussian likelihood with per-class mean, variance and prior estimation.
 *
 * <p>Variance smoothing follows the scale-invariant scikit-learn convention: a single epsilon
 * equal to {@code var_smoothing * max(variance)} is added to every variance. Adding an absolute
 * epsilon instead would behave differently for features measured in meters versus millimetres.</p>
 */
public final class GaussianNaiveBayes implements Estimator, Persistable {
    private final double varianceSmoothing;
    private double[] classes;
    private double[] means;
    private double[] variances;
    private double[] logPriors;
    private int featureCount;

    private GaussianNaiveBayes(Builder builder) {
        varianceSmoothing = builder.varianceSmoothing;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public GaussianNaiveBayes fit(Matrix features, double[] targets) {
        MlChecks.trainingData(features, targets);
        featureCount = features.columns();
        TreeMap<Double, Integer> classIndices = new TreeMap<>();
        for (double target : targets) {
            classIndices.putIfAbsent(target, 0);
        }
        int classCount = classIndices.size();
        classes = new double[classCount];
        int position = 0;
        for (double label : classIndices.keySet()) {
            classes[position++] = label;
        }
        for (int index = 0; index < classCount; index++) {
            classIndices.put(classes[index], index);
        }
        int rows = features.rows();
        int[] rowClasses = new int[rows];
        for (int row = 0; row < rows; row++) {
            rowClasses[row] = classIndices.get(targets[row]);
        }
        if (NativeKernels.available() && rows >= 512) {
            double[] nativeMeans = new double[classCount * featureCount];
            double[] nativeVariances = new double[nativeMeans.length];
            double[] nativePriors = new double[classCount];
            int[] nativeCounts = new int[classCount];
            if (NativeKernels.gaussianNbFitChecked(features.data(), rows, featureCount,
                    rowClasses, classCount, nativeMeans, nativeVariances, nativePriors,
                    nativeCounts, varianceSmoothing)) {
                means = nativeMeans;
                variances = nativeVariances;
                logPriors = nativePriors;
                return this;
            }
        }
        means = new double[classCount * featureCount];
        variances = new double[means.length];
        logPriors = new double[classCount];
        int[] counts = new int[classCount];
        double[] data = features.data();
        for (int row = 0; row < rows; row++) {
            int classIndex = rowClasses[row];
            counts[classIndex]++;
            int offset = row * featureCount;
            int base = classIndex * featureCount;
            for (int feature = 0; feature < featureCount; feature++) {
                means[base + feature] += data[offset + feature];
            }
        }
        for (int classIndex = 0; classIndex < classCount; classIndex++) {
            logPriors[classIndex] = Math.log((double) counts[classIndex] / features.rows());
            int base = classIndex * featureCount;
            for (int feature = 0; feature < featureCount; feature++) {
                means[base + feature] /= counts[classIndex];
            }
        }
        for (int row = 0; row < rows; row++) {
            int classIndex = rowClasses[row];
            int offset = row * featureCount;
            int base = classIndex * featureCount;
            for (int feature = 0; feature < featureCount; feature++) {
                double delta = data[offset + feature] - means[base + feature];
                variances[base + feature] += delta * delta;
            }
        }
        double maximumVariance = 0.0;
        for (int classIndex = 0; classIndex < classCount; classIndex++) {
            int base = classIndex * featureCount;
            for (int feature = 0; feature < featureCount; feature++) {
                variances[base + feature] /= counts[classIndex];
                maximumVariance = Math.max(maximumVariance, variances[base + feature]);
            }
        }
        double epsilon = varianceSmoothing * maximumVariance;
        if (!(epsilon > 0.0)) {
            epsilon = varianceSmoothing;
        }
        for (int i = 0; i < variances.length; i++) {
            variances[i] += epsilon;
        }
        return this;
    }

    public Matrix predictLogProbability(Matrix features) {
        MlChecks.predictionData(features, featureCount, classes != null);
        Matrix result = Matrix.zeros(features.rows(), classes.length);
        double[] data = features.data();
        for (int row = 0; row < features.rows(); row++) {
            int offset = row * featureCount;
            for (int classIndex = 0; classIndex < classes.length; classIndex++) {
                double logProbability = logPriors[classIndex];
                int base = classIndex * featureCount;
                for (int feature = 0; feature < featureCount; feature++) {
                    double variance = variances[base + feature];
                    double delta = data[offset + feature] - means[base + feature];
                    logProbability += -0.5 * (Math.log(2.0 * Math.PI * variance)
                            + delta * delta / variance);
                }
                result.set(row, classIndex, logProbability);
            }
        }
        return result;
    }

    public Matrix predictProbability(Matrix features) {
        Matrix logProbability = predictLogProbability(features);
        for (int row = 0; row < features.rows(); row++) {
            double maximum = Double.NEGATIVE_INFINITY;
            for (int column = 0; column < classes.length; column++) {
                maximum = Math.max(maximum, logProbability.get(row, column));
            }
            double sum = 0.0;
            if (!Double.isFinite(maximum)) {
                double priorSum = 0.0;
                for (int column = 0; column < classes.length; column++) {
                    double value = Math.exp(logPriors[column]);
                    logProbability.set(row, column, value);
                    priorSum += value;
                }
                for (int column = 0; column < classes.length; column++) {
                    logProbability.set(row, column, priorSum > 0.0
                            ? logProbability.get(row, column) / priorSum
                            : 1.0 / classes.length);
                }
                continue;
            }
            for (int column = 0; column < classes.length; column++) {
                double value = Math.exp(logProbability.get(row, column) - maximum);
                logProbability.set(row, column, value);
                sum += value;
            }
            for (int column = 0; column < classes.length; column++) {
                logProbability.set(row, column, logProbability.get(row, column) / sum);
            }
        }
        return logProbability;
    }

    @Override
    public double[] predict(Matrix features) {
        Matrix probabilities = predictProbability(features);
        double[] predictions = new double[features.rows()];
        for (int row = 0; row < features.rows(); row++) {
            int best = 0;
            for (int column = 1; column < classes.length; column++) {
                if (probabilities.get(row, column) > probabilities.get(row, best)) {
                    best = column;
                }
            }
            predictions[row] = classes[best];
        }
        return predictions;
    }

    public double[] classes() {
        return classes == null ? null : classes.clone();
    }

    public Matrix means() {
        if (classes == null) {
            throw new IllegalStateException("GaussianNaiveBayes must be fitted first");
        }
        return new Matrix(classes.length, featureCount, means);
    }

    public Matrix variances() {
        if (classes == null) {
            throw new IllegalStateException("GaussianNaiveBayes must be fitted first");
        }
        return new Matrix(classes.length, featureCount, variances);
    }

    public double[] logPriors() {
        return logPriors == null ? null : logPriors.clone();
    }

    @Override
    public ModelKind kind() {
        return ModelKind.GAUSSIAN_NAIVE_BAYES;
    }

    @Override
    public void write(ModelWriter writer) {
        if (classes == null) {
            throw new IllegalStateException("GaussianNaiveBayes must be fitted before serialization");
        }
        writer.putInt(featureCount);
        writer.putInt(classes.length);
        writer.putDoubles(classes);
        writer.putDoubles(means);
        writer.putDoubles(variances);
        writer.putDoubles(logPriors);
    }

    public static GaussianNaiveBayes read(ModelReader reader) {
        int features = reader.getInt();
        int classCount = reader.getInt();
        if (features <= 0 || classCount <= 0) {
            throw new ModelFormatException("Invalid GaussianNaiveBayes header");
        }
        int size = Math.multiplyExact(classCount, features);
        GaussianNaiveBayes model = new GaussianNaiveBayes(new Builder());
        model.featureCount = features;
        model.classes = reader.getDoubles(classCount);
        model.means = reader.getDoubles(size);
        model.variances = reader.getDoubles(size);
        model.logPriors = reader.getDoubles(classCount);
        return model;
    }

    public static final class Builder {
        private double varianceSmoothing = 1e-9;

        public Builder varianceSmoothing(double varianceSmoothing) {
            this.varianceSmoothing = varianceSmoothing;
            return this;
        }

        public GaussianNaiveBayes build() {
            MlChecks.positiveFinite(varianceSmoothing, "varianceSmoothing");
            return new GaussianNaiveBayes(this);
        }
    }
}
