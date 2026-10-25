package jsl.ml;

import jsl.core.Matrix;
import jsl.core.ModelFormatException;
import jsl.core.ModelKind;
import jsl.core.ModelReader;
import jsl.core.ModelWriter;
import jsl.core.Persistable;
import jsl.core.NativeKernels;
import jsl.core.VectorMath;

/** L2-regularized least squares with an unpenalized intercept, solved by Cholesky. */
public final class Ridge implements Estimator, Persistable {
    private final double alpha;
    private final boolean fitIntercept;
    private double[] weights;
    private double intercept;
    private int featureCount;

    private Ridge(Builder builder) {
        alpha = builder.alpha;
        fitIntercept = builder.fitIntercept;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Ridge fit(Matrix features, double[] targets) {
        MlChecks.trainingData(features, targets);
        featureCount = features.columns();
        int rows = features.rows();
        if (NativeKernels.available() && rows >= 512 && featureCount >= 4) {
            double[] nativeWeights = new double[featureCount];
            double[] nativeIntercept = new double[1];
            int nativeThreads = Math.max(1, Runtime.getRuntime().availableProcessors());
            if (NativeKernels.ridgeFitChecked(features.data(), rows, featureCount, targets,
                    alpha, fitIntercept, nativeThreads, nativeWeights, nativeIntercept)) {
                weights = nativeWeights;
                intercept = nativeIntercept[0];
                return this;
            }
        }
        double[] means = new double[featureCount];
        double targetMean = 0.0;
        for (int row = 0; row < rows; row++) {
            for (int feature = 0; feature < featureCount; feature++) {
                means[feature] += features.get(row, feature);
            }
            targetMean += targets[row];
        }
        for (int feature = 0; feature < featureCount; feature++) {
            means[feature] /= rows;
        }
        targetMean /= rows;
        double[] normal = new double[featureCount * featureCount];
        double[] rhs = new double[featureCount];
        double[] centered = new double[featureCount];
        for (int row = 0; row < rows; row++) {
            for (int feature = 0; feature < featureCount; feature++) {
                centered[feature] = features.get(row, feature)
                        - (fitIntercept ? means[feature] : 0.0);
            }
            double target = targets[row] - (fitIntercept ? targetMean : 0.0);
            for (int i = 0; i < featureCount; i++) {
                rhs[i] += centered[i] * target;
                int base = i * featureCount;
                for (int j = i; j < featureCount; j++) {
                    normal[base + j] += centered[i] * centered[j];
                }
            }
        }
        for (int i = 0; i < featureCount; i++) {
            for (int j = i; j < featureCount; j++) {
                normal[j * featureCount + i] = normal[i * featureCount + j];
            }
            normal[i * featureCount + i] += alpha;
        }
        weights = solveSymmetricPositiveDefinite(normal, rhs, featureCount);
        intercept = 0.0;
        if (fitIntercept) {
            intercept = targetMean;
            for (int feature = 0; feature < featureCount; feature++) {
                intercept -= means[feature] * weights[feature];
            }
        }
        return this;
    }

    private static double[] solveSymmetricPositiveDefinite(double[] matrix, double[] rhs, int size) {
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
        double[] intermediate = new double[size];
        for (int i = 0; i < size; i++) {
            double sum = rhs[i];
            for (int k = 0; k < i; k++) {
                sum -= lower[i * size + k] * intermediate[k];
            }
            intermediate[i] = sum / lower[i * size + i];
        }
        double[] solution = new double[size];
        for (int i = size - 1; i >= 0; i--) {
            double sum = intermediate[i];
            for (int k = i + 1; k < size; k++) {
                sum -= lower[k * size + i] * solution[k];
            }
            solution[i] = sum / lower[i * size + i];
        }
        return solution;
    }

    @Override
    public double[] predict(Matrix features) {
        MlChecks.predictionData(features, featureCount, weights != null);
        double[] result = new double[features.rows()];
        for (int row = 0; row < features.rows(); row++) {
            result[row] = intercept + VectorMath.dot(weights, 0,
                    features.data(), row * featureCount, featureCount);
        }
        return result;
    }

    public double[] weights() {
        return weights.clone();
    }

    public double intercept() {
        return intercept;
    }


    @Override
    public ModelKind kind() {
        return ModelKind.RIDGE;
    }

    @Override
    public void write(ModelWriter writer) {
        if (weights == null) {
            throw new IllegalStateException("Ridge must be fitted before serialization");
        }
        writer.putDouble(alpha);
        writer.putByte(fitIntercept ? 1 : 0);
        writer.putInt(featureCount);
        writer.putDouble(intercept);
        writer.putDoubles(weights);
    }

    public static Ridge read(ModelReader reader) {
        double alpha = reader.getDouble();
        boolean fitInt = reader.getByte() != 0;
        int features = reader.getInt();
        if (features <= 0) {
            throw new ModelFormatException("Invalid Ridge feature count");
        }
        double intercept = reader.getDouble();
        double[] weights = reader.getDoubles(features);
        Ridge model = new Builder().alpha(alpha).fitIntercept(fitInt).build();
        model.featureCount = features;
        model.intercept = intercept;
        model.weights = weights;
        return model;
    }

    public static final class Builder {
        private double alpha = 1.0;
        private boolean fitIntercept = true;

        public Builder alpha(double alpha) {
            this.alpha = alpha;
            return this;
        }

        public Builder fitIntercept(boolean fitIntercept) {
            this.fitIntercept = fitIntercept;
            return this;
        }

        public Ridge build() {
            MlChecks.positiveFinite(alpha, "alpha");
            return new Ridge(this);
        }
    }
}
