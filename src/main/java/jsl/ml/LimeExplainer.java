package jsl.ml;

import jsl.core.Matrix;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

/**
 * Local Interpretable Model-agnostic Explanations (LIME): explains individual predictions of
 * any black-box {@link Estimator} or {@link Pipeline} by sampling Gaussian perturbations around
 * the target instance, weighting them by an exponential Euclidean kernel
 * {@code w(z) = exp(-||x - z||^2 / kernelWidth^2)}, and fitting a local weighted Ridge linear
 * surrogate model.
 */
public final class LimeExplainer {
    public record LocalExplanation(double intercept, double[] featureWeights,
                                   double localPrediction, double modelPrediction,
                                   double r2Fidelity, int[] topFeatures) {
    }

    private final Estimator model;
    private final double[] featureStds;
    private final int numSamples;
    private final double kernelWidth;
    private final long seed;

    public LimeExplainer(Estimator model, Matrix trainingData) {
        this(model, trainingData, 500, 0.0, 42L);
    }

    public LimeExplainer(Estimator model, Matrix trainingData, int numSamples,
                         double kernelWidth, long seed) {
        this.model = Objects.requireNonNull(model, "model");
        MlChecks.features(trainingData);
        if (numSamples < 20) {
            throw new IllegalArgumentException("numSamples must be >= 20");
        }
        int rows = trainingData.rows();
        int cols = trainingData.columns();
        this.featureStds = new double[cols];
        double[] means = new double[cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                means[c] += trainingData.get(r, c);
            }
        }
        for (int c = 0; c < cols; c++) means[c] /= rows;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double d = trainingData.get(r, c) - means[c];
                featureStds[c] += d * d;
            }
        }
        for (int c = 0; c < cols; c++) {
            double s = Math.sqrt(featureStds[c] / Math.max(1, rows));
            featureStds[c] = s > 1e-12 ? s : 1.0;
        }
        this.numSamples = numSamples;
        this.kernelWidth = kernelWidth > 0.0 ? kernelWidth : 0.75 * Math.sqrt(cols);
        this.seed = seed;
    }

    public LocalExplanation explainInstance(double[] instance) {
        Objects.requireNonNull(instance, "instance");
        int cols = featureStds.length;
        if (instance.length != cols) {
            throw new IllegalArgumentException(
                    "Expected instance of length " + cols + " but got " + instance.length);
        }
        Random rng = new Random(seed);
        double[] perturbed = new double[numSamples * cols];
        double[] sampleWeights = new double[numSamples];

        // First row is the exact instance itself
        System.arraycopy(instance, 0, perturbed, 0, cols);
        sampleWeights[0] = 1.0;

        double kw2 = kernelWidth * kernelWidth;
        for (int s = 1; s < numSamples; s++) {
            double sqDist = 0.0;
            int base = s * cols;
            for (int c = 0; c < cols; c++) {
                double z = rng.nextGaussian();
                perturbed[base + c] = instance[c] + z * featureStds[c] * 0.5;
                sqDist += (z * 0.5) * (z * 0.5);
            }
            sampleWeights[s] = Math.exp(-sqDist / kw2);
        }

        Matrix perturbedMatrix = new Matrix(numSamples, cols, perturbed);
        double[] y = ShapExplainer.evaluateModel(model, perturbedMatrix);
        double modelPred = y[0];

        // Solve weighted Ridge regression: (X_aug^T W X_aug + lambda I) beta = X_aug^T W y
        int aug = cols + 1;
        double[] xtwx = new double[aug * aug];
        double[] xtwy = new double[aug];
        double[] rowAug = new double[aug];
        for (int s = 0; s < numSamples; s++) {
            double w = sampleWeights[s];
            rowAug[0] = 1.0;
            System.arraycopy(perturbed, s * cols, rowAug, 1, cols);
            for (int i = 0; i < aug; i++) {
                xtwy[i] += w * rowAug[i] * y[s];
                for (int j = i; j < aug; j++) {
                    xtwx[i * aug + j] += w * rowAug[i] * rowAug[j];
                }
            }
        }
        for (int i = 0; i < aug; i++) {
            for (int j = 0; j < i; j++) {
                xtwx[i * aug + j] = xtwx[j * aug + i];
            }
            if (i > 0) {
                xtwx[i * aug + i] += 1e-4;
            } else {
                xtwx[0] += 1e-9;
            }
        }
        double[] beta = KernelRidge.choleskySolve(xtwx, xtwy, aug);
        double intercept = beta[0];
        double[] weights = Arrays.copyOfRange(beta, 1, aug);

        double localPred = intercept;
        for (int c = 0; c < cols; c++) {
            localPred += weights[c] * instance[c];
        }

        // Compute weighted R2 fidelity
        double wSum = 0.0;
        double yWeightedMean = 0.0;
        for (int s = 0; s < numSamples; s++) {
            wSum += sampleWeights[s];
            yWeightedMean += sampleWeights[s] * y[s];
        }
        yWeightedMean /= wSum;
        double ssRes = 0.0;
        double ssTot = 0.0;
        for (int s = 0; s < numSamples; s++) {
            double predS = intercept;
            for (int c = 0; c < cols; c++) {
                predS += weights[c] * perturbed[s * cols + c];
            }
            double err = y[s] - predS;
            double tot = y[s] - yWeightedMean;
            ssRes += sampleWeights[s] * err * err;
            ssTot += sampleWeights[s] * tot * tot;
        }
        double r2 = ssTot <= 1e-15 ? 1.0 : Math.max(0.0, 1.0 - ssRes / ssTot);

        Integer[] order = new Integer[cols];
        for (int c = 0; c < cols; c++) order[c] = c;
        Arrays.sort(order, (a, b) -> Double.compare(
                Math.abs(weights[b] * featureStds[b]),
                Math.abs(weights[a] * featureStds[a])));
        int[] topFeatures = Arrays.stream(order).mapToInt(Integer::intValue).toArray();

        return new LocalExplanation(intercept, weights, localPred, modelPred, r2, topFeatures);
    }
}
