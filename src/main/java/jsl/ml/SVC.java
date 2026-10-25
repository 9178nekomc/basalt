package jsl.ml;

import jsl.core.Matrix;
import jsl.core.ModelFormatException;
import jsl.core.ModelKind;
import jsl.core.ModelReader;
import jsl.core.ModelWriter;
import jsl.core.Persistable;
import java.util.Objects;
import java.util.Random;
import java.util.TreeSet;

/**
 * Non-linear Support Vector Classifier trained with Sequential Minimal Optimization (SMO),
 * supporting {@code LINEAR}, {@code RBF}, {@code POLY}, and {@code SIGMOID} kernels as well as
 * binary and multi-class (One-vs-Rest) classification, matching {@code sklearn.svm.SVC}.
 *
 * <p>With {@link Builder#probability(boolean)} enabled, decision values are calibrated into
 * probabilities with Platt scaling (the damped Newton method used by libsvm), one
 * {@code (A, B)} pair per binary sub-problem.
 */
public final class SVC implements Estimator, Persistable {
    private final Kernel kernel;
    private final double cParam;
    private final double tolerance;
    private final int maxIterations;
    private final long seed;
    private final boolean probability;

    private int featureCount;
    private double[] classes;
    private Matrix supportX;
    private double[][] alphas;
    private double[] intercepts;
    private double[] plattA;
    private double[] plattB;

    private SVC(Builder builder) {
        this.kernel = builder.kernel;
        this.cParam = builder.c;
        this.tolerance = builder.tolerance;
        this.maxIterations = builder.maxIterations;
        this.seed = builder.seed;
        this.probability = builder.probability;
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

        if (probability) {
            calibrate(features, targets, models);
        }
        return this;
    }

    /** Fits one Platt sigmoid per binary sub-problem on the training decision values. */
    private void calibrate(Matrix features, double[] targets, int models) {
        plattA = new double[models];
        plattB = new double[models];
        int rows = features.rows();
        for (int m = 0; m < models; m++) {
            double posLabel = classes.length == 2 ? classes[1] : classes[m];
            double[] decision = decision(features, m);
            double[] t = new double[rows];
            int positives = 0;
            for (int i = 0; i < rows; i++) {
                boolean positive = Double.compare(targets[i], posLabel) == 0;
                t[i] = positive ? 1.0 : 0.0;
                if (positive) {
                    positives++;
                }
            }
            int negatives = rows - positives;
            // Platt's target smoothing keeps the fit finite for separable data.
            double hiTarget = (positives + 1.0) / (positives + 2.0);
            double loTarget = 1.0 / (negatives + 2.0);
            for (int i = 0; i < rows; i++) {
                t[i] = t[i] > 0.5 ? hiTarget : loTarget;
            }
            double[] ab = sigmoidTrain(decision, t, rows);
            plattA[m] = ab[0];
            plattB[m] = ab[1];
        }
    }

    /**
     * Damped Newton minimisation of the negative log-likelihood of
     * {@code P(y=1|f) = 1 / (1 + exp(A*f + B))}, following libsvm's {@code sigmoid_train}.
     */
    private static double[] sigmoidTrain(double[] f, double[] t, int n) {
        double prior1 = 0.0;
        double prior0 = 0.0;
        for (int i = 0; i < n; i++) {
            if (t[i] > 0.5) {
                prior1 += 1.0;
            } else {
                prior0 += 1.0;
            }
        }
        final int maxIter = 100;
        final double minStep = 1e-10;
        final double sigma = 1e-12;
        final double eps = 1e-5;

        double a = 0.0;
        double b = Math.log((prior0 + 1.0) / (prior1 + 1.0));
        double fval = 0.0;
        for (int i = 0; i < n; i++) {
            double fApB = f[i] * a + b;
            fval += logLoss(t[i], fApB);
        }

        for (int iter = 0; iter < maxIter; iter++) {
            double h11 = sigma;
            double h22 = sigma;
            double h21 = 0.0;
            double g1 = 0.0;
            double g2 = 0.0;
            for (int i = 0; i < n; i++) {
                double fApB = f[i] * a + b;
                double p;
                if (fApB >= 0) {
                    p = Math.exp(-fApB) / (1.0 + Math.exp(-fApB));
                } else {
                    p = 1.0 / (1.0 + Math.exp(fApB));
                }
                double d2 = p * (1.0 - p);
                h11 += f[i] * f[i] * d2;
                h22 += d2;
                h21 += f[i] * d2;
                double d1 = t[i] - p;
                g1 += f[i] * d1;
                g2 += d1;
            }
            if (Math.abs(g1) < eps && Math.abs(g2) < eps) {
                break;
            }
            double det = h11 * h22 - h21 * h21;
            if (!(Math.abs(det) > 1e-15)) {
                break;
            }
            double dA = -(h22 * g1 - h21 * g2) / det;
            double dB = -(-h21 * g1 + h11 * g2) / det;
            double gd = g1 * dA + g2 * dB;

            double stepSize = 1.0;
            boolean improved = false;
            while (stepSize >= minStep) {
                double newA = a + stepSize * dA;
                double newB = b + stepSize * dB;
                double newF = 0.0;
                for (int i = 0; i < n; i++) {
                    newF += logLoss(t[i], f[i] * newA + newB);
                }
                if (newF < fval + 0.0001 * stepSize * gd) {
                    a = newA;
                    b = newB;
                    fval = newF;
                    improved = true;
                    break;
                }
                stepSize /= 2.0;
            }
            if (!improved) {
                break;
            }
        }
        return new double[] {a, b};
    }

    /** Stable {@code t*log(p) + (1-t)*log(1-p)} for {@code p = sigmoid(-fApB)}. */
    private static double logLoss(double t, double fApB) {
        if (fApB >= 0) {
            return t * fApB + Math.log1p(Math.exp(-fApB));
        }
        return (t - 1.0) * fApB + Math.log1p(Math.exp(fApB));
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

    /** Decision value of one binary sub-problem for every row. */
    private double[] decision(Matrix features, int model) {
        int rows = features.rows();
        int trainRows = supportX.rows();
        double[] xData = features.data();
        double[] svData = supportX.data();
        double[] scores = new double[rows];
        for (int r = 0; r < rows; r++) {
            double sum = intercepts[model];
            int rOff = r * featureCount;
            for (int i = 0; i < trainRows; i++) {
                double coeff = alphas[model][i];
                if (coeff != 0.0) {
                    sum += coeff * kernel.evaluate(xData, rOff, svData, i * featureCount, featureCount);
                }
            }
            scores[r] = sum;
        }
        return scores;
    }

    /** Decision function of the binary problem (or of the first One-vs-Rest model). */
    public double[] decisionFunction(Matrix features) {
        MlChecks.predictionData(features, featureCount, supportX != null);
        return decision(features, 0);
    }

    /**
     * Class probabilities, calibrated with Platt scaling.
     *
     * @throws IllegalStateException if the model was built without {@code probability(true)}
     */
    public Matrix predictProbability(Matrix features) {
        MlChecks.predictionData(features, featureCount, supportX != null);
        if (!probability || plattA == null) {
            throw new IllegalStateException(
                    "predictProbability requires SVC.builder().probability(true)");
        }
        int rows = features.rows();
        int classCount = classes.length;
        int models = alphas.length;
        double[] out = new double[rows * classCount];

        if (classCount == 2) {
            double[] f = decision(features, 0);
            for (int r = 0; r < rows; r++) {
                double p = sigmoid(f[r], plattA[0], plattB[0]);
                out[r * 2] = 1.0 - p;
                out[r * 2 + 1] = p;
            }
            return new Matrix(rows, 2, out);
        }

        double[] raw = new double[models];
        double[][] decisions = new double[models][];
        for (int m = 0; m < models; m++) {
            decisions[m] = decision(features, m);
        }
        for (int r = 0; r < rows; r++) {
            double sum = 0.0;
            for (int m = 0; m < models; m++) {
                raw[m] = sigmoid(decisions[m][r], plattA[m], plattB[m]);
                sum += raw[m];
            }
            double denom = sum > 0.0 ? sum : 1.0;
            for (int m = 0; m < classCount; m++) {
                out[r * classCount + m] = raw[m] / denom;
            }
        }
        return new Matrix(rows, classCount, out);
    }

    /**
     * {@code P(y = +1 | f) = 1 / (1 + exp(A*f + B))}.
     *
     * <p>Note the sign: Platt's parametrisation puts the positive class at
     * {@code A*f + B -> -inf}, so this is {@code sigmoid(-(A*f + B))}, not
     * {@code sigmoid(A*f + B)}.
     */
    private static double sigmoid(double f, double a, double b) {
        double z = a * f + b;
        if (z >= 0.0) {
            double e = Math.exp(-z);
            return e / (1.0 + e);
        }
        return 1.0 / (1.0 + Math.exp(z));
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

    public boolean probabilityEnabled() {
        return probability;
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
        writer.putByte(probability ? 1 : 0);
        if (probability) {
            writer.putDoubles(plattA);
            writer.putDoubles(plattB);
        }
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
        boolean probability = reader.getByte() != 0;
        double[] plattA = null;
        double[] plattB = null;
        if (probability) {
            plattA = reader.getDoubles(models);
            plattB = reader.getDoubles(models);
        }
        SVC svc = new Builder().kernel(kernel).c(c).tolerance(tol).maxIterations(maxIter)
                .seed(seed).probability(probability).build();
        svc.featureCount = features;
        svc.classes = classes;
        svc.supportX = new Matrix(svRows, features, svData);
        svc.alphas = alphas;
        svc.intercepts = intercepts;
        svc.plattA = plattA;
        svc.plattB = plattB;
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
        private boolean probability = false;

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

        public Builder probability(boolean probability) {
            this.probability = probability;
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
