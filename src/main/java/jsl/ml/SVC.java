package jsl.ml;

import jsl.core.Matrix;
import jsl.core.ModelFormatException;
import jsl.core.ModelKind;
import jsl.core.ModelReader;
import jsl.core.ModelWriter;
import jsl.core.Persistable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.TreeSet;

/**
 * Non-linear Support Vector Classifier trained with Sequential Minimal Optimization (SMO),
 * supporting {@code LINEAR}, {@code RBF}, {@code POLY}, and {@code SIGMOID} kernels as well as
 * binary and multi-class (One-vs-Rest) classification, matching {@code sklearn.svm.SVC}.
 */
public final class SVC implements Estimator, Persistable {
    private final Kernel kernel;
    private final double cParam;
    private final double tolerance;
    private final int maxIterations;
    private final long seed;

    private int featureCount;
    private double[] classes;
    private Matrix supportX;
    private double[][] alphas;
    private double[] intercepts;

    private SVC(Builder builder) {
        this.kernel = builder.kernel;
        this.cParam = builder.c;
        this.tolerance = builder.tolerance;
        this.maxIterations = builder.maxIterations;
        this.seed = builder.seed;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public SVC fit(Matrix features, double[] targets) {
        MlChecks.trainingData(features, targets);
        featureCount = features.columns();
        int rows = features.rows();
        TreeSet<Double> unique = new TreeSet<>();
        for (double t : targets) unique.add(t);
        if (unique.size() < 2) {
            throw new IllegalArgumentException("SVC requires at least 2 classes");
        }
        classes = new double[unique.size()];
        int idx = 0;
        for (double v : unique) classes[idx++] = v;

        supportX = features.copy();
        Matrix gram = kernel.gram(supportX, supportX);
        double[] kMatrix = gram.data();

        int models = classes.length == 2 ? 1 : classes.length;
        alphas = new double[models][rows];
        intercepts = new double[models];

        for (int m = 0; m < models; m++) {
            double posLabel = classes.length == 2 ? classes[1] : classes[m];
            double[] y = new double[rows];
            for (int i = 0; i < rows; i++) {
                y[i] = Double.compare(targets[i], posLabel) == 0 ? 1.0 : -1.0;
            }
            BinarySmoResult res = fitBinarySmo(kMatrix, y, rows, seed + m * 101L);
            for (int i = 0; i < rows; i++) {
                alphas[m][i] = res.alpha[i] * y[i];
            }
            intercepts[m] = res.bias;
        }
        return this;
    }

    private BinarySmoResult fitBinarySmo(double[] kMatrix, double[] y, int n, long rngSeed) {
        double[] alpha = new double[n];
        double[] fCache = new double[n];
        double bias = 0.0;
        Random random = new Random(rngSeed);

        int passes = 0;
        int iter = 0;
        while (passes < 5 && iter < maxIterations) {
            iter++;
            int numChanged = 0;
            for (int i = 0; i < n; i++) {
                double ei = fCache[i] + bias - y[i];
                if ((y[i] * ei < -tolerance && alpha[i] < cParam)
                        || (y[i] * ei > tolerance && alpha[i] > 0.0)) {
                    int j = i;
                    double maxDelta = -1.0;
                    for (int cand = 0; cand < n; cand++) {
                        if (cand == i) continue;
                        double ec = fCache[cand] + bias - y[cand];
                        double delta = Math.abs(ei - ec);
                        if (delta > maxDelta) {
                            maxDelta = delta;
                            j = cand;
                        }
                    }
                    if (j == i || random.nextDouble() < 0.15) {
                        j = random.nextInt(n - 1);
                        if (j >= i) j++;
                    }
                    double ej = fCache[j] + bias - y[j];
                    double oldAi = alpha[i];
                    double oldAj = alpha[j];
                    double low;
                    double high;
                    if (y[i] != y[j]) {
                        low = Math.max(0.0, oldAj - oldAi);
                        high = Math.min(cParam, cParam + oldAj - oldAi);
                    } else {
                        low = Math.max(0.0, oldAi + oldAj - cParam);
                        high = Math.min(cParam, oldAi + oldAj);
                    }
                    if (high - low <= 1e-12) continue;

                    double eta = 2.0 * kMatrix[i * n + j] - kMatrix[i * n + i] - kMatrix[j * n + j];
                    if (eta >= -1e-12) continue;

                    double newAj = oldAj - y[j] * (ei - ej) / eta;
                    if (newAj > high) newAj = high;
                    else if (newAj < low) newAj = low;
                    if (Math.abs(newAj - oldAj) < 1e-7) continue;

                    double newAi = oldAi + y[i] * y[j] * (oldAj - newAj);
                    alpha[i] = newAi;
                    alpha[j] = newAj;

                    double deltaAi = (newAi - oldAi) * y[i];
                    double deltaAj = (newAj - oldAj) * y[j];
                    for (int k = 0; k < n; k++) {
                        fCache[k] += deltaAi * kMatrix[i * n + k] + deltaAj * kMatrix[j * n + k];
                    }

                    double b1 = bias - ei - deltaAi * kMatrix[i * n + i] - deltaAj * kMatrix[i * n + j];
                    double b2 = bias - ej - deltaAi * kMatrix[i * n + j] - deltaAj * kMatrix[j * n + j];
                    if (0.0 < newAi && newAi < cParam) {
                        bias = b1;
                    } else if (0.0 < newAj && newAj < cParam) {
                        bias = b2;
                    } else {
                        bias = 0.5 * (b1 + b2);
                    }
                    numChanged++;
                }
            }
            if (numChanged == 0) {
                passes++;
            } else {
                passes = 0;
            }
        }
        return new BinarySmoResult(alpha, bias);
    }

    public double[] decisionFunction(Matrix features) {
        MlChecks.predictionData(features, featureCount, supportX != null);
        int rows = features.rows();
        int trainRows = supportX.rows();
        double[] xData = features.data();
        double[] svData = supportX.data();
        double[] scores = new double[rows];
        for (int r = 0; r < rows; r++) {
            double sum = intercepts[0];
            int rOff = r * featureCount;
            for (int i = 0; i < trainRows; i++) {
                double coeff = alphas[0][i];
                if (coeff != 0.0) {
                    sum += coeff * kernel.evaluate(xData, rOff, svData, i * featureCount, featureCount);
                }
            }
            scores[r] = sum;
        }
        return scores;
    }

    @Override
    public double[] predict(Matrix features) {
        MlChecks.predictionData(features, featureCount, supportX != null);
        int rows = features.rows();
        int trainRows = supportX.rows();
        double[] xData = features.data();
        double[] svData = supportX.data();
        double[] out = new double[rows];

        if (classes.length == 2) {
            for (int r = 0; r < rows; r++) {
                double sum = intercepts[0];
                int rOff = r * featureCount;
                for (int i = 0; i < trainRows; i++) {
                    double coeff = alphas[0][i];
                    if (coeff != 0.0) {
                        sum += coeff * kernel.evaluate(xData, rOff, svData, i * featureCount, featureCount);
                    }
                }
                out[r] = sum >= 0.0 ? classes[1] : classes[0];
            }
            return out;
        }

        for (int r = 0; r < rows; r++) {
            int rOff = r * featureCount;
            int bestClass = 0;
            double bestScore = Double.NEGATIVE_INFINITY;
            for (int m = 0; m < classes.length; m++) {
                double sum = intercepts[m];
                for (int i = 0; i < trainRows; i++) {
                    double coeff = alphas[m][i];
                    if (coeff != 0.0) {
                        sum += coeff * kernel.evaluate(xData, rOff, svData, i * featureCount, featureCount);
                    }
                }
                if (sum > bestScore) {
                    bestScore = sum;
                    bestClass = m;
                }
            }
            out[r] = classes[bestClass];
        }
        return out;
    }

    public Kernel kernel() {
        return kernel;
    }

    public double[] classes() {
        return classes == null ? null : classes.clone();
    }

    @Override
    public ModelKind kind() {
        return ModelKind.SVC;
    }

    @Override
    public void write(ModelWriter writer) {
        if (supportX == null) {
            throw new IllegalStateException("SVC must be fitted before serialization");
        }
        kernel.write(writer);
        writer.putDouble(cParam);
        writer.putDouble(tolerance);
        writer.putInt(maxIterations);
        writer.putLong(seed);
        writer.putInt(featureCount);
        writer.putInt(classes.length);
        writer.putDoubles(classes);
        writer.putInt(supportX.rows());
        writer.putDoubles(supportX.data());
        writer.putInt(alphas.length);
        for (double[] row : alphas) {
            writer.putDoubles(row);
        }
        writer.putDoubles(intercepts);
    }

    public static SVC read(ModelReader reader) {
        Kernel kernel = Kernel.read(reader);
        double c = reader.getDouble();
        double tol = reader.getDouble();
        int maxIter = reader.getInt();
        long seed = reader.getLong();
        int features = reader.getInt();
        int classCount = reader.getInt();
        if (features <= 0 || classCount < 2) {
            throw new ModelFormatException("Invalid SVC header");
        }
        double[] classes = reader.getDoubles(classCount);
        int svRows = reader.getInt();
        double[] svData = reader.getDoubles(Math.multiplyExact(svRows, features));
        int models = reader.getInt();
        double[][] alphas = new double[models][];
        for (int m = 0; m < models; m++) {
            alphas[m] = reader.getDoubles(svRows);
        }
        double[] intercepts = reader.getDoubles(models);
        SVC svc = new Builder().kernel(kernel).c(c).tolerance(tol).maxIterations(maxIter).seed(seed).build();
        svc.featureCount = features;
        svc.classes = classes;
        svc.supportX = new Matrix(svRows, features, svData);
        svc.alphas = alphas;
        svc.intercepts = intercepts;
        return svc;
    }

    private record BinarySmoResult(double[] alpha, double bias) {
    }

    public static final class Builder {
        private Kernel kernel = Kernel.rbf(1.0);
        private double c = 1.0;
        private double tolerance = 1e-4;
        private int maxIterations = 600;
        private long seed = 42L;

        public Builder kernel(Kernel kernel) {
            this.kernel = Objects.requireNonNull(kernel, "kernel");
            return this;
        }

        public Builder rbf(double gamma) {
            this.kernel = Kernel.rbf(gamma);
            return this;
        }

        public Builder poly(int degree, double gamma, double coef0) {
            this.kernel = Kernel.poly(degree, gamma, coef0);
            return this;
        }

        public Builder linear() {
            this.kernel = Kernel.linear();
            return this;
        }

        public Builder c(double c) {
            MlChecks.positiveFinite(c, "c");
            this.c = c;
            return this;
        }

        public Builder tolerance(double tolerance) {
            MlChecks.positiveFinite(tolerance, "tolerance");
            this.tolerance = tolerance;
            return this;
        }

        public Builder maxIterations(int maxIterations) {
            if (maxIterations <= 0) throw new IllegalArgumentException("maxIterations must be positive");
            this.maxIterations = maxIterations;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public SVC build() {
            return new SVC(this);
        }
    }
}
