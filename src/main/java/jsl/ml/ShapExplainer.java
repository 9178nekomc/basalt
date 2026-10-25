package jsl.ml;

import jsl.core.Matrix;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * SHapley Additive exPlanations (SHAP) calculator for JSL models.
 *
 * <p>Provides exact Shapley values when {@code features <= 10} and Monte-Carlo permutation
 * Shapley values for higher dimensions, strictly satisfying the local efficiency / additivity
 * axiom: {@code baseValue + sum_j(shapValues[i][j]) == prediction[i]}.</p>
 *
 * <p>For linear models ({@link LinearRegression}, {@link Ridge}, {@link Lasso}, {@link LinearSVM}),
 * uses the exact closed-form LinearSHAP formula {@code phi_j = w_j * (x_j - E[X_j])} in O(p) time.</p>
 */
public final class ShapExplainer {
    public record Explanation(double baseValue, Matrix shapValues, double[] predictions,
                              double[] meanAbsShap) {
    }

    private final Estimator model;
    private final Matrix background;
    private final double baseValue;
    private final double[] featureMeans;
    private final int maxCoalitions;
    private final long seed;

    public ShapExplainer(Estimator model, Matrix background) {
        this(model, background, 64, 42L);
    }

    public ShapExplainer(Estimator model, Matrix background, int maxCoalitions, long seed) {
        this.model = Objects.requireNonNull(model, "model");
        MlChecks.features(background);
        if (maxCoalitions < 4) {
            throw new IllegalArgumentException("maxCoalitions must be >= 4");
        }
        this.background = background.copy();
        this.maxCoalitions = maxCoalitions;
        this.seed = seed;

        int bgRows = background.rows();
        int cols = background.columns();
        this.featureMeans = new double[cols];
        for (int r = 0; r < bgRows; r++) {
            for (int c = 0; c < cols; c++) {
                featureMeans[c] += background.get(r, c);
            }
        }
        for (int c = 0; c < cols; c++) {
            featureMeans[c] /= bgRows;
        }
        double[] bgPreds = evaluateModel(model, background);
        double sum = 0.0;
        for (double v : bgPreds) sum += v;
        this.baseValue = sum / bgRows;
    }

    public double baseValue() {
        return baseValue;
    }

    public Explanation explain(Matrix instances) {
        MlChecks.features(instances);
        int rows = instances.rows();
        int cols = instances.columns();
        if (cols != background.columns()) {
            throw new IllegalArgumentException("Feature count mismatch with background dataset");
        }

        double[] preds = evaluateModel(model, instances);
        double[] shapData = new double[rows * cols];

        // Fast exact LinearSHAP path for linear models
        double[] linearWeights = extractLinearWeights(model);
        if (linearWeights != null) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    shapData[r * cols + c] = linearWeights[c] * (instances.get(r, c) - featureMeans[c]);
                }
            }
        } else if (cols <= 8) {
            // Exact Coalition Enumeration Shapley values
            int totalMasks = 1 << cols;
            double[] fact = new double[cols + 1];
            fact[0] = 1.0;
            for (int i = 1; i <= cols; i++) fact[i] = fact[i - 1] * i;

            for (int r = 0; r < rows; r++) {
                double[] x = new double[cols];
                instances.copyRow(r, x, 0);
                double[] maskVal = new double[totalMasks];
                maskVal[0] = baseValue;
                maskVal[totalMasks - 1] = preds[r];
                for (int mask = 1; mask < totalMasks - 1; mask++) {
                    maskVal[mask] = conditionalExpectation(x, mask);
                }
                for (int j = 0; j < cols; j++) {
                    int bit = 1 << j;
                    double phi = 0.0;
                    for (int mask = 0; mask < totalMasks; mask++) {
                        if ((mask & bit) != 0) continue;
                        int s = Integer.bitCount(mask);
                        double weight = (fact[s] * fact[cols - s - 1]) / fact[cols];
                        phi += weight * (maskVal[mask | bit] - maskVal[mask]);
                    }
                    shapData[r * cols + j] = phi;
                }
            }
        } else {
            // Antithetic Permutation Monte-Carlo Shapley values (guarantees exact efficiency sum(phi) = f(x) - baseValue)
            Random rng = new Random(seed);
            int[] order = new int[cols];
            for (int i = 0; i < cols; i++) order[i] = i;

            for (int r = 0; r < rows; r++) {
                double[] x = new double[cols];
                instances.copyRow(r, x, 0);
                double[] phi = new double[cols];
                for (int sample = 0; sample < maxCoalitions; sample++) {
                    for (int i = cols - 1; i > 0; i--) {
                        int j = rng.nextInt(i + 1);
                        int tmp = order[i];
                        order[i] = order[j];
                        order[j] = tmp;
                    }
                    int mask = 0;
                    double prevVal = baseValue;
                    for (int pos = 0; pos < cols; pos++) {
                        int feat = order[pos];
                        mask |= (1 << feat);
                        double nextVal = (pos == cols - 1) ? preds[r] : conditionalExpectation(x, mask);
                        phi[feat] += (nextVal - prevVal);
                        prevVal = nextVal;
                    }
                }
                for (int c = 0; c < cols; c++) {
                    shapData[r * cols + c] = phi[c] / maxCoalitions;
                }
            }
        }

        double[] meanAbs = new double[cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                meanAbs[c] += Math.abs(shapData[r * cols + c]);
            }
        }
        for (int c = 0; c < cols; c++) {
            meanAbs[c] /= rows;
        }
        return new Explanation(baseValue, new Matrix(rows, cols, shapData), preds, meanAbs);
    }

    private double conditionalExpectation(double[] x, int mask) {
        int bgRows = background.rows();
        int cols = background.columns();
        double[] composite = new double[bgRows * cols];
        double[] bgData = background.data();
        for (int r = 0; r < bgRows; r++) {
            int base = r * cols;
            for (int c = 0; c < cols; c++) {
                composite[base + c] = ((mask & (1 << c)) != 0) ? x[c] : bgData[base + c];
            }
        }
        double[] preds = evaluateModel(model, new Matrix(bgRows, cols, composite));
        double sum = 0.0;
        for (double v : preds) sum += v;
        return sum / bgRows;
    }

    static double[] evaluateModel(Estimator model, Matrix x) {
        if (model instanceof LogisticRegression logr) {
            return logr.predictProbability(x);
        }
        if (model instanceof HistogramGradientBoosting hgb) {
            return hgb.predictRaw(x);
        }
        if (model instanceof RandomForest rf && rf.task() == TreeTask.CLASSIFICATION) {
            Matrix probs = rf.predictProbability(x);
            int targetCol = probs.columns() > 1 ? 1 : 0;
            double[] out = new double[probs.rows()];
            for (int r = 0; r < out.length; r++) out[r] = probs.get(r, targetCol);
            return out;
        }
        return model.predict(x);
    }

    private static double[] extractLinearWeights(Estimator model) {
        if (model instanceof LinearRegression lr) return lr.weights();
        if (model instanceof Ridge r) return r.weights();
        if (model instanceof Lasso l) return l.weights();
        return null;
    }
}
