package basalt.ml;

import basalt.core.Matrix;
import basalt.core.ModelFormatException;
import basalt.core.ModelKind;
import basalt.core.ModelReader;
import basalt.core.ModelWriter;
import basalt.core.Persistable;
import java.util.Objects;

/**
 * Non-linear dimensionality reduction through centered Kernel Principal Component Analysis
 * (Kernel PCA), matching {@code sklearn.decomposition.KernelPCA}.
 */
public final class KernelPCA implements Estimator, Transformer, Persistable {
    private final int componentCount;
    private final Kernel kernel;
    private final int maxIterations;
    private final double tolerance;

    private Matrix fitX;
    private double[] kFitRows;
    private double kFitAll;
    private double[] eigenvalues;
    private double[] alphas; // [componentCount * n_samples], scaled by 1/sqrt(lambda)
    private int featureCount;

    private KernelPCA(Builder builder) {
        this.componentCount = builder.components;
        this.kernel = builder.kernel;
        this.maxIterations = builder.maxIterations;
        this.tolerance = builder.tolerance;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public KernelPCA fit(Matrix features) {
        return fit(features, null);
    }

    @Override
    public KernelPCA fit(Matrix features, double[] ignoredTargets) {
        MlChecks.features(features);
        int n = features.rows();
        featureCount = features.columns();
        if (componentCount > n) {
            throw new IllegalArgumentException("components cannot exceed sample count");
        }
        fitX = features.copy();
        Matrix gram = kernel.gram(fitX, fitX);
        double[] kMat = gram.data();

        kFitRows = new double[n];
        kFitAll = 0.0;
        for (int i = 0; i < n; i++) {
            double rowSum = 0.0;
            for (int j = 0; j < n; j++) {
                rowSum += kMat[i * n + j];
            }
            kFitRows[i] = rowSum / n;
            kFitAll += rowSum;
        }
        kFitAll /= ((double) n * n);

        double[] centered = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double v = kMat[i * n + j] - kFitRows[i] - kFitRows[j] + kFitAll;
                centered[i * n + j] = v;
                centered[j * n + i] = v;
            }
        }

        double[] eigVecs = new double[n * n];
        double[] eigVals = jacobiSymmetric(centered, eigVecs, n, maxIterations, tolerance);
        int[] order = descendingOrder(eigVals);

        eigenvalues = new double[componentCount];
        alphas = new double[componentCount * n];
        for (int c = 0; c < componentCount; c++) {
            int idx = order[c];
            double lambda = Math.max(0.0, eigVals[idx]);
            eigenvalues[c] = lambda;
            double scale = lambda > 1e-12 ? 1.0 / Math.sqrt(lambda) : 0.0;
            int largestRow = 0;
            for (int r = 1; r < n; r++) {
                if (Math.abs(eigVecs[r * n + idx]) > Math.abs(eigVecs[largestRow * n + idx])) {
                    largestRow = r;
                }
            }
            double sign = eigVecs[largestRow * n + idx] < 0.0 ? -1.0 : 1.0;
            for (int r = 0; r < n; r++) {
                alphas[c * n + r] = sign * eigVecs[r * n + idx] * scale;
            }
        }
        return this;
    }

    static double[] jacobiSymmetric(double[] matrix, double[] eigenvectors, int dim,
                                    int maxSweeps, double tol) {
        for (int i = 0; i < dim; i++) {
            eigenvectors[i * dim + i] = 1.0;
        }
        double frob = 0.0;
        for (double v : matrix) frob += v * v;
        double threshold = tol * Math.max(1.0, Math.sqrt(frob));
        for (int sweep = 0; sweep < maxSweeps; sweep++) {
            double offDiag = 0.0;
            for (int p = 0; p < dim - 1; p++) {
                for (int q = p + 1; q < dim; q++) {
                    double apq = matrix[p * dim + q];
                    offDiag += apq * apq;
                    if (apq == 0.0) continue;
                    double app = matrix[p * dim + p];
                    double aqq = matrix[q * dim + q];
                    double angle = 0.5 * Math.atan2(2.0 * apq, aqq - app);
                    double c = Math.cos(angle);
                    double s = Math.sin(angle);
                    matrix[p * dim + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                    matrix[q * dim + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                    matrix[p * dim + q] = 0.0;
                    matrix[q * dim + p] = 0.0;
                    for (int k = 0; k < dim; k++) {
                        if (k == p || k == q) continue;
                        double akp = matrix[k * dim + p];
                        double akq = matrix[k * dim + q];
                        matrix[k * dim + p] = matrix[p * dim + k] = c * akp - s * akq;
                        matrix[k * dim + q] = matrix[q * dim + k] = s * akp + c * akq;
                    }
                    for (int r = 0; r < dim; r++) {
                        int base = r * dim;
                        double vp = eigenvectors[base + p];
                        double vq = eigenvectors[base + q];
                        eigenvectors[base + p] = c * vp - s * vq;
                        eigenvectors[base + q] = s * vp + c * vq;
                    }
                }
            }
            if (Math.sqrt(offDiag) <= threshold) break;
        }
        double[] evals = new double[dim];
        for (int i = 0; i < dim; i++) evals[i] = matrix[i * dim + i];
        return evals;
    }

    private static int[] descendingOrder(double[] values) {
        int[] order = new int[values.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        for (int i = 1; i < order.length; i++) {
            int sel = order[i];
            int pos = i;
            while (pos > 0 && values[order[pos - 1]] < values[sel]) {
                order[pos] = order[pos - 1];
                pos--;
            }
            order[pos] = sel;
        }
        return order;
    }

    @Override
    public Matrix transform(Matrix features) {
        MlChecks.predictionData(features, featureCount, alphas != null);
        int rows = features.rows();
        int n = fitX.rows();
        double[] xData = features.data();
        double[] trainData = fitX.data();
        double[] out = new double[rows * componentCount];
        double[] kRow = new double[n];

        for (int r = 0; r < rows; r++) {
            int rOff = r * featureCount;
            double rowMean = 0.0;
            for (int i = 0; i < n; i++) {
                double kv = kernel.evaluate(xData, rOff, trainData, i * featureCount, featureCount);
                kRow[i] = kv;
                rowMean += kv;
            }
            rowMean /= n;
            for (int i = 0; i < n; i++) {
                kRow[i] = kRow[i] - rowMean - kFitRows[i] + kFitAll;
            }
            for (int c = 0; c < componentCount; c++) {
                double proj = 0.0;
                int aBase = c * n;
                for (int i = 0; i < n; i++) {
                    proj += kRow[i] * alphas[aBase + i];
                }
                out[r * componentCount + c] = proj;
            }
        }
        return new Matrix(rows, componentCount, out);
    }

    @Override
    public double[] predict(Matrix features) {
        return transform(features).toArray();
    }

    public double[] eigenvalues() {
        if (eigenvalues == null) throw new IllegalStateException("KernelPCA is not fitted");
        return eigenvalues.clone();
    }

    @Override
    public ModelKind kind() {
        return ModelKind.KERNEL_PCA;
    }

    @Override
    public void write(ModelWriter writer) {
        if (alphas == null) {
            throw new IllegalStateException("KernelPCA must be fitted before serialization");
        }
        kernel.write(writer);
        writer.putInt(componentCount);
        writer.putInt(featureCount);
        writer.putInt(fitX.rows());
        writer.putDouble(kFitAll);
        writer.putDoubles(fitX.data());
        writer.putDoubles(kFitRows);
        writer.putDoubles(eigenvalues);
        writer.putDoubles(alphas);
    }

    public static KernelPCA read(ModelReader reader) {
        Kernel kernel = Kernel.read(reader);
        int components = reader.getInt();
        int features = reader.getInt();
        int rows = reader.getInt();
        double kAll = reader.getDouble();
        if (components <= 0 || features <= 0 || rows <= 0) {
            throw new ModelFormatException("Invalid KernelPCA header");
        }
        double[] xData = reader.getDoubles(Math.multiplyExact(rows, features));
        double[] kRows = reader.getDoubles(rows);
        double[] evals = reader.getDoubles(components);
        double[] alphas = reader.getDoubles(Math.multiplyExact(components, rows));
        KernelPCA model = new Builder().components(components).kernel(kernel).build();
        model.featureCount = features;
        model.fitX = new Matrix(rows, features, xData);
        model.kFitAll = kAll;
        model.kFitRows = kRows;
        model.eigenvalues = evals;
        model.alphas = alphas;
        return model;
    }

    public static final class Builder {
        private int components = 2;
        private Kernel kernel = Kernel.rbf(1.0);
        private int maxIterations = 60;
        private double tolerance = 1e-12;

        public Builder components(int components) {
            if (components <= 0) throw new IllegalArgumentException("components must be positive");
            this.components = components;
            return this;
        }

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

        public KernelPCA build() {
            return new KernelPCA(this);
        }
    }
}
