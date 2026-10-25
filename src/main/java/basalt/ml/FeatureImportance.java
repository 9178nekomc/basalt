package basalt.ml;

import basalt.core.Matrix;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

/**
 * Unified feature importance utility supporting:
 * <ul>
 *   <li>Direct impurity-based tree feature importances ({@link DecisionTree}, {@link RandomForest},
 *       {@link HistogramGradientBoosting}) and normalized linear weight magnitudes;</li>
 *   <li>Model-agnostic Permutation Feature Importance ({@code sklearn.inspection.permutation_importance})
 *       applicable to any {@link Estimator} or {@link Pipeline}.</li>
 * </ul>
 */
public final class FeatureImportance {
    public record PermutationResult(double[] importancesMean, double[] importancesStd,
                                    double[][] rawImportances) {
        public int[] rankedFeaturesDescending() {
            Integer[] order = new Integer[importancesMean.length];
            for (int i = 0; i < order.length; i++) order[i] = i;
            Arrays.sort(order, (a, b) -> Double.compare(importancesMean[b], importancesMean[a]));
            return Arrays.stream(order).mapToInt(Integer::intValue).toArray();
        }
    }

    private FeatureImportance() {
    }

    /** Returns normalized impurity-based or weight-magnitude feature importances summing to 1.0. */
    public static double[] of(Estimator model) {
        Objects.requireNonNull(model, "model");
        if (model instanceof Pipeline p && p.estimator() != null) {
            return of(p.estimator());
        }
        double[] raw;
        if (model instanceof DecisionTree dt) {
            raw = dt.featureImportances();
        } else if (model instanceof RandomForest rf) {
            raw = rf.featureImportances();
        } else if (model instanceof HistogramGradientBoosting hgb) {
            raw = hgb.featureImportances();
        } else if (model instanceof Ridge r) {
            raw = absCopy(r.weights());
        } else if (model instanceof Lasso l) {
            raw = absCopy(l.weights());
        } else if (model instanceof LinearRegression lr) {
            raw = absCopy(lr.weights());
        } else if (model instanceof LogisticRegression logr) {
            raw = absCopy(logr.weights());
        } else if (model instanceof LinearSVM svm) {
            raw = absCopy(svm.weights());
        } else {
            throw new IllegalArgumentException(
                    model.getClass().getSimpleName() + " does not expose intrinsic feature importances; "
                            + "use FeatureImportance.permutationImportance(...) instead.");
        }
        double sum = 0.0;
        for (double v : raw) sum += v;
        if (sum > 0.0) {
            for (int i = 0; i < raw.length; i++) raw[i] /= sum;
        }
        return raw;
    }

    /**
     * Computes model-agnostic permutation feature importance over {@code nRepeats} random shuffles
     * per feature column.
     */
    public static PermutationResult permutationImportance(Estimator model, Matrix features,
                                                          double[] targets, Estimators.Metric metric,
                                                          int nRepeats, long seed) {
        Objects.requireNonNull(model, "model");
        MlChecks.trainingData(features, targets);
        Objects.requireNonNull(metric, "metric");
        if (nRepeats <= 0) throw new IllegalArgumentException("nRepeats must be positive");

        double baselineScore = GridSearchCV.evaluateScore(metric, targets, model.predict(features));
        int rows = features.rows();
        int cols = features.columns();
        double[][] raw = new double[cols][nRepeats];
        double[] mean = new double[cols];
        double[] std = new double[cols];
        Random rng = new Random(seed);

        int[] perm = new int[rows];
        for (int i = 0; i < rows; i++) perm[i] = i;

        for (int c = 0; c < cols; c++) {
            double sum = 0.0;
            for (int rep = 0; rep < nRepeats; rep++) {
                for (int i = rows - 1; i > 0; i--) {
                    int j = rng.nextInt(i + 1);
                    int tmp = perm[i];
                    perm[i] = perm[j];
                    perm[j] = tmp;
                }
                Matrix corrupted = features.copy();
                for (int r = 0; r < rows; r++) {
                    corrupted.set(r, c, features.get(perm[r], c));
                }
                double permutedScore = GridSearchCV.evaluateScore(metric, targets, model.predict(corrupted));
                double drop = baselineScore - permutedScore;
                raw[c][rep] = drop;
                sum += drop;
            }
            mean[c] = sum / nRepeats;
            double var = 0.0;
            for (int rep = 0; rep < nRepeats; rep++) {
                double d = raw[c][rep] - mean[c];
                var += d * d;
            }
            std[c] = Math.sqrt(var / nRepeats);
        }
        return new PermutationResult(mean, std, raw);
    }

    private static double[] absCopy(double[] weights) {
        double[] out = new double[weights.length];
        for (int i = 0; i < weights.length; i++) out[i] = Math.abs(weights[i]);
        return out;
    }
}
