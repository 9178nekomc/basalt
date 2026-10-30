package basalt.ml;

import basalt.core.Matrix;
import basalt.core.ModelFormatException;
import basalt.core.ModelKind;
import basalt.core.ModelReader;
import basalt.core.ModelWriter;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * Selects the top {@code k} features according to univariate statistical tests
 * ({@code F_CLASSIF}, {@code F_REGRESSION}, {@code CHI2}).
 */
public final class SelectKBest implements Transformer, Estimator {
    public enum ScoreFunc {
        F_CLASSIF,
        F_REGRESSION,
        CHI2
    }

    private final int k;
    private final ScoreFunc scoreFunc;
    private int inputFeatures;
    private double[] scores;
    private int[] selectedIndices;

    private SelectKBest(Builder builder) {
        this.k = builder.k;
        this.scoreFunc = builder.scoreFunc;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public SelectKBest fit(Matrix features, double[] targets) {
        MlChecks.trainingData(features, targets);
        inputFeatures = features.columns();
        if (k <= 0 || k > inputFeatures) {
            throw new IllegalArgumentException("k must be in [1, " + inputFeatures + "]");
        }
        scores = switch (scoreFunc) {
            case F_CLASSIF -> computeAnovaF(features, targets);
            case F_REGRESSION -> computeRegressionF(features, targets);
            case CHI2 -> computeChi2(features, targets);
        };
        Integer[] order = new Integer[inputFeatures];
        for (int i = 0; i < inputFeatures; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> {
            int cmp = Double.compare(scores[b], scores[a]);
            return cmp != 0 ? cmp : Integer.compare(a, b);
        });
        selectedIndices = new int[k];
        for (int i = 0; i < k; i++) {
            selectedIndices[i] = order[i];
        }
        Arrays.sort(selectedIndices);
        return this;
    }

    private static double[] computeAnovaF(Matrix features, double[] targets) {
        int rows = features.rows();
        int cols = features.columns();
        TreeMap<Double, Integer> classMap = new TreeMap<>();
        for (double t : targets) classMap.putIfAbsent(t, 0);
        int kClasses = classMap.size();
        int idx = 0;
        for (Double key : classMap.keySet()) classMap.put(key, idx++);

        int[] classCounts = new int[kClasses];
        int[] encoded = new int[rows];
        for (int r = 0; r < rows; r++) {
            int c = classMap.get(targets[r]);
            encoded[r] = c;
            classCounts[c]++;
        }

        double[] fScores = new double[cols];
        for (int f = 0; f < cols; f++) {
            double grandMean = 0.0;
            double[] classMeans = new double[kClasses];
            for (int r = 0; r < rows; r++) {
                double v = features.get(r, f);
                grandMean += v;
                classMeans[encoded[r]] += v;
            }
            grandMean /= rows;
            for (int c = 0; c < kClasses; c++) {
                if (classCounts[c] > 0) classMeans[c] /= classCounts[c];
            }
            double ssb = 0.0;
            for (int c = 0; c < kClasses; c++) {
                double d = classMeans[c] - grandMean;
                ssb += classCounts[c] * d * d;
            }
            double ssw = 0.0;
            for (int r = 0; r < rows; r++) {
                double d = features.get(r, f) - classMeans[encoded[r]];
                ssw += d * d;
            }
            double dfBetween = Math.max(1, kClasses - 1);
            double dfWithin = Math.max(1, rows - kClasses);
            double msb = ssb / dfBetween;
            double msw = ssw / dfWithin;
            fScores[f] = msw > 1e-15 ? msb / msw : 0.0;
        }
        return fScores;
    }

    private static double[] computeRegressionF(Matrix features, double[] targets) {
        int rows = features.rows();
        int cols = features.columns();
        double yMean = 0.0;
        for (double t : targets) yMean += t;
        yMean /= rows;
        double yVar = 0.0;
        for (double t : targets) {
            double d = t - yMean;
            yVar += d * d;
        }

        double[] fScores = new double[cols];
        for (int f = 0; f < cols; f++) {
            double xMean = 0.0;
            for (int r = 0; r < rows; r++) xMean += features.get(r, f);
            xMean /= rows;
            double cov = 0.0;
            double xVar = 0.0;
            for (int r = 0; r < rows; r++) {
                double dx = features.get(r, f) - xMean;
                double dy = targets[r] - yMean;
                cov += dx * dy;
                xVar += dx * dx;
            }
            double denom = Math.sqrt(xVar * yVar);
            double corr = denom > 1e-15 ? cov / denom : 0.0;
            double r2 = Math.min(1.0 - 1e-15, corr * corr);
            fScores[f] = (r2 / (1.0 - r2)) * Math.max(1, rows - 2);
        }
        return fScores;
    }

    private static double[] computeChi2(Matrix features, double[] targets) {
        int rows = features.rows();
        int cols = features.columns();
        TreeMap<Double, Integer> classMap = new TreeMap<>();
        for (double t : targets) classMap.putIfAbsent(t, 0);
        int kClasses = classMap.size();
        int idx = 0;
        for (Double key : classMap.keySet()) classMap.put(key, idx++);

        int[] classCounts = new int[kClasses];
        int[] encoded = new int[rows];
        for (int r = 0; r < rows; r++) {
            int c = classMap.get(targets[r]);
            encoded[r] = c;
            classCounts[c]++;
        }

        double[] chi2 = new double[cols];
        for (int f = 0; f < cols; f++) {
            double featureSum = 0.0;
            double[] observed = new double[kClasses];
            for (int r = 0; r < rows; r++) {
                double v = features.get(r, f);
                if (v < 0.0) {
                    throw new IllegalArgumentException("CHI2 requires non-negative features");
                }
                featureSum += v;
                observed[encoded[r]] += v;
            }
            double score = 0.0;
            for (int c = 0; c < kClasses; c++) {
                double expected = featureSum * ((double) classCounts[c] / rows);
                if (expected > 1e-15) {
                    double d = observed[c] - expected;
                    score += (d * d) / expected;
                }
            }
            chi2[f] = score;
        }
        return chi2;
    }

    @Override
    public Matrix transform(Matrix features) {
        MlChecks.predictionData(features, inputFeatures, selectedIndices != null);
        int rows = features.rows();
        double[] dst = new double[rows * k];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < k; c++) {
                dst[r * k + c] = features.get(r, selectedIndices[c]);
            }
        }
        return new Matrix(rows, k, dst);
    }

    @Override
    public double[] predict(Matrix features) {
        return transform(features).toArray();
    }

    public double[] scores() {
        if (scores == null) throw new IllegalStateException("SelectKBest is not fitted");
        return scores.clone();
    }

    public int[] selectedIndices() {
        if (selectedIndices == null) throw new IllegalStateException("SelectKBest is not fitted");
        return selectedIndices.clone();
    }

    public boolean[] getSupport() {
        if (selectedIndices == null) throw new IllegalStateException("SelectKBest is not fitted");
        boolean[] mask = new boolean[inputFeatures];
        for (int idx : selectedIndices) mask[idx] = true;
        return mask;
    }

    @Override
    public ModelKind kind() {
        return ModelKind.SELECT_K_BEST;
    }

    @Override
    public void write(ModelWriter writer) {
        if (selectedIndices == null) {
            throw new IllegalStateException("SelectKBest must be fitted before serialization");
        }
        writer.putInt(k);
        writer.putInt(scoreFunc.ordinal());
        writer.putInt(inputFeatures);
        writer.putDoubles(scores);
        writer.putInts(selectedIndices);
    }

    public static SelectKBest read(ModelReader reader) {
        int k = reader.getInt();
        int sfOrd = reader.getInt();
        int inFeatures = reader.getInt();
        if (k <= 0 || inFeatures < k || sfOrd < 0 || sfOrd >= ScoreFunc.values().length) {
            throw new ModelFormatException("Invalid SelectKBest header");
        }
        SelectKBest selector = new Builder().k(k).scoreFunc(ScoreFunc.values()[sfOrd]).build();
        selector.inputFeatures = inFeatures;
        selector.scores = reader.getDoubles(inFeatures);
        selector.selectedIndices = reader.getInts(k);
        return selector;
    }

    public static final class Builder {
        private int k = 5;
        private ScoreFunc scoreFunc = ScoreFunc.F_CLASSIF;

        public Builder k(int k) {
            if (k <= 0) throw new IllegalArgumentException("k must be positive");
            this.k = k;
            return this;
        }

        public Builder scoreFunc(ScoreFunc scoreFunc) {
            this.scoreFunc = Objects.requireNonNull(scoreFunc, "scoreFunc");
            return this;
        }

        public SelectKBest build() {
            return new SelectKBest(this);
        }
    }
}
