package basalt.ml;

import basalt.core.Matrix;
import basalt.core.ModelFormatException;
import basalt.core.ModelKind;
import basalt.core.ModelReader;
import basalt.core.ModelWriter;
import basalt.core.Persistable;
import java.util.Objects;

/**
 * Kernel Ridge Regression (KRR) combining Ridge regularization with the kernel trick
 * {@code (K + alpha * I) dual_coef = y}, solved by Cholesky decomposition.
 */
public final class KernelRidge implements Estimator, Persistable {
    private final Kernel kernel;
    private final double alpha;
    private Matrix fitX;
    private double[] dualCoef;
    private int featureCount;

    private KernelRidge(Builder builder) {
        this.kernel = builder.kernel;
        this.alpha = builder.alpha;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public KernelRidge fit(Matrix features, double[] targets) {
        MlChecks.trainingData(features, targets);
        featureCount = features.columns();
        int n = features.rows();
        fitX = features.copy();
        Matrix gram = kernel.gram(fitX, fitX);
        double[] kMat = gram.data();
        for (int i = 0; i < n; i++) {
            kMat[i * n + i] += alpha;
        }
        dualCoef = choleskySolve(kMat, targets, n);
        return this;
    }

    static double[] choleskySolve(double[] matrix, double[] rhs, int size) {
        double[] lower = matrix.clone();
        for (int i = 0; i < size; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = lower[i * size + j];
                for (int k = 0; k < j; k++) {
                    sum -= lower[i * size + k] * lower[j * size + k];
                }
                if (i == j) {
                    lower[i * size + j] = Math.sqrt(Math.max(sum, 1e-12));
                } else {
                    lower[i * size + j] = sum / lower[j * size + j];
                }
            }
        }
        double[] y = new double[size];
        for (int i = 0; i < size; i++) {
            double sum = rhs[i];
            for (int k = 0; k < i; k++) {
                sum -= lower[i * size + k] * y[k];
            }
            y[i] = sum / lower[i * size + i];
        }
        double[] x = new double[size];
        for (int i = size - 1; i >= 0; i--) {
            double sum = y[i];
            for (int k = i + 1; k < size; k++) {
                sum -= lower[k * size + i] * x[k];
            }
            x[i] = sum / lower[i * size + i];
        }
        return x;
    }

    @Override
    public double[] predict(Matrix features) {
        MlChecks.predictionData(features, featureCount, dualCoef != null);
        int rows = features.rows();
        int trainRows = fitX.rows();
        double[] xData = features.data();
        double[] trainData = fitX.data();
        double[] preds = new double[rows];
        for (int r = 0; r < rows; r++) {
            double sum = 0.0;
            int rOff = r * featureCount;
            for (int i = 0; i < trainRows; i++) {
                sum += dualCoef[i] * kernel.evaluate(xData, rOff, trainData, i * featureCount, featureCount);
            }
            preds[r] = sum;
        }
        return preds;
    }

    public double[] dualCoef() {
        if (dualCoef == null) throw new IllegalStateException("KernelRidge is not fitted");
        return dualCoef.clone();
    }

    public Kernel kernel() {
        return kernel;
    }

    @Override
    public ModelKind kind() {
        return ModelKind.KERNEL_RIDGE;
    }

    @Override
    public void write(ModelWriter writer) {
        if (dualCoef == null) {
            throw new IllegalStateException("KernelRidge must be fitted before serialization");
        }
        kernel.write(writer);
        writer.putDouble(alpha);
        writer.putInt(featureCount);
        writer.putInt(fitX.rows());
        writer.putDoubles(fitX.data());
        writer.putDoubles(dualCoef);
    }

    public static KernelRidge read(ModelReader reader) {
        Kernel kernel = Kernel.read(reader);
        double alpha = reader.getDouble();
        int features = reader.getInt();
        int rows = reader.getInt();
        if (features <= 0 || rows <= 0) {
            throw new ModelFormatException("Invalid KernelRidge header");
        }
        double[] xData = reader.getDoubles(Math.multiplyExact(rows, features));
        double[] dual = reader.getDoubles(rows);
        KernelRidge model = new Builder().kernel(kernel).alpha(alpha).build();
        model.featureCount = features;
        model.fitX = new Matrix(rows, features, xData);
        model.dualCoef = dual;
        return model;
    }

    public static final class Builder {
        private Kernel kernel = Kernel.rbf(1.0);
        private double alpha = 1.0;

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

        public Builder alpha(double alpha) {
            MlChecks.positiveFinite(alpha, "alpha");
            this.alpha = alpha;
            return this;
        }

        public KernelRidge build() {
            return new KernelRidge(this);
        }
    }
}
