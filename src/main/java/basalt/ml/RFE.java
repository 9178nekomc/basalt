package basalt.ml;

import basalt.core.Matrix;
import basalt.core.ModelKind;
import basalt.core.ModelWriter;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Recursive Feature Elimination (RFE) that iteratively fits an importance-capable or linear
 * estimator and prunes the least important features until {@code nFeaturesToSelect} remain,
 * matching {@code sklearn.feature_selection.RFE}.
 */
public final class RFE implements Transformer, Estimator {
    private final Supplier<Estimator> estimatorSupplier;
    private final int nFeaturesToSelect;
    private final int step;

    private int inputFeatures;
    private boolean[] support;
    private int[] ranking;
    private int[] selectedIndices;
    private Estimator finalEstimator;

    private RFE(Builder builder) {
        this.estimatorSupplier = builder.estimatorSupplier;
        this.nFeaturesToSelect = builder.nFeaturesToSelect;
        this.step = builder.step;
    }

    public static Builder builder(Supplier<Estimator> estimatorSupplier) {
        return new Builder(estimatorSupplier);
    }

    @Override
    public RFE fit(Matrix features, double[] targets) {
        MlChecks.trainingData(features, targets);
        inputFeatures = features.columns();
        if (nFeaturesToSelect <= 0 || nFeaturesToSelect > inputFeatures) {
            throw new IllegalArgumentException(
                    "nFeaturesToSelect must be in [1, " + inputFeatures + "]");
        }
        support = new boolean[inputFeatures];
        Arrays.fill(support, true);
        ranking = new int[inputFeatures];
        Arrays.fill(ranking, 1);

        int remainingCount = inputFeatures;
        while (remainingCount > nFeaturesToSelect) {
            int[] active = activeIndices(support, remainingCount);
            Matrix subX = subMatrix(features, active);
            Estimator est = estimatorSupplier.get();
            est.fit(subX, targets);
            double[] imp = extractImportances(est, active.length);

            Integer[] order = new Integer[active.length];
            for (int i = 0; i < active.length; i++) order[i] = i;
            Arrays.sort(order, (a, b) -> Double.compare(imp[a], imp[b]));

            int eliminate = Math.min(step, remainingCount - nFeaturesToSelect);
            for (int i = 0; i < eliminate; i++) {
                int featureIdx = active[order[i]];
                support[featureIdx] = false;
            }
            for (int f = 0; f < inputFeatures; f++) {
                if (!support[f]) {
                    ranking[f]++;
                }
            }
            remainingCount -= eliminate;
        }

        selectedIndices = activeIndices(support, nFeaturesToSelect);
        finalEstimator = estimatorSupplier.get();
        finalEstimator.fit(subMatrix(features, selectedIndices), targets);
        return this;
    }

    private static int[] activeIndices(boolean[] support, int count) {
        int[] active = new int[count];
        int pos = 0;
        for (int i = 0; i < support.length; i++) {
            if (support[i]) active[pos++] = i;
        }
        return active;
    }

    private static Matrix subMatrix(Matrix features, int[] cols) {
        int rows = features.rows();
        int k = cols.length;
        double[] dst = new double[rows * k];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < k; c++) {
                dst[r * k + c] = features.get(r, cols[c]);
            }
        }
        return new Matrix(rows, k, dst);
    }

    static double[] extractImportances(Estimator est, int expectedFeatures) {
        if (est instanceof DecisionTree tree) return tree.featureImportances();
        if (est instanceof RandomForest rf) return rf.featureImportances();
        if (est instanceof HistogramGradientBoosting hgb) return hgb.featureImportances();
        double[] weights = null;
        if (est instanceof Ridge ridge) weights = ridge.weights();
        else if (est instanceof Lasso lasso) weights = lasso.weights();
        else if (est instanceof LinearRegression lr) weights = lr.weights();
        else if (est instanceof LogisticRegression logr) weights = logr.weights();
        else if (est instanceof LinearSVM svm) weights = svm.weights();
        if (weights != null && weights.length == expectedFeatures) {
            double[] imp = new double[expectedFeatures];
            for (int i = 0; i < expectedFeatures; i++) {
                imp[i] = Math.abs(weights[i]);
            }
            return imp;
        }
        throw new IllegalArgumentException(
                "RFE requires an estimator with featureImportances() or linear weights(), got "
                        + est.getClass().getSimpleName());
    }

    @Override
    public Matrix transform(Matrix features) {
        MlChecks.predictionData(features, inputFeatures, selectedIndices != null);
        return subMatrix(features, selectedIndices);
    }

    @Override
    public double[] predict(Matrix features) {
        Matrix reduced = transform(features);
        return finalEstimator.predict(reduced);
    }

    public boolean[] support() {
        if (support == null) throw new IllegalStateException("RFE is not fitted");
        return support.clone();
    }

    public int[] ranking() {
        if (ranking == null) throw new IllegalStateException("RFE is not fitted");
        return ranking.clone();
    }

    public int[] selectedIndices() {
        if (selectedIndices == null) throw new IllegalStateException("RFE is not fitted");
        return selectedIndices.clone();
    }

    public Estimator estimator() {
        return finalEstimator;
    }

    @Override
    public ModelKind kind() {
        return ModelKind.SELECT_K_BEST;
    }

    @Override
    public void write(ModelWriter writer) {
        if (selectedIndices == null) {
            throw new IllegalStateException("RFE must be fitted before serialization");
        }
        writer.putInt(nFeaturesToSelect);
        writer.putInt(0);
        writer.putInt(inputFeatures);
        double[] pseudoScores = new double[inputFeatures];
        for (int i = 0; i < inputFeatures; i++) pseudoScores[i] = 1.0 / ranking[i];
        writer.putDoubles(pseudoScores);
        writer.putInts(selectedIndices);
    }

    public static final class Builder {
        private final Supplier<Estimator> estimatorSupplier;
        private int nFeaturesToSelect = 2;
        private int step = 1;

        private Builder(Supplier<Estimator> estimatorSupplier) {
            this.estimatorSupplier = Objects.requireNonNull(estimatorSupplier, "estimatorSupplier");
        }

        public Builder nFeaturesToSelect(int nFeaturesToSelect) {
            if (nFeaturesToSelect <= 0) throw new IllegalArgumentException("nFeaturesToSelect must be positive");
            this.nFeaturesToSelect = nFeaturesToSelect;
            return this;
        }

        public Builder step(int step) {
            if (step <= 0) throw new IllegalArgumentException("step must be positive");
            this.step = step;
            return this;
        }

        public RFE build() {
            return new RFE(this);
        }
    }
}
