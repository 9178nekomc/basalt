package basalt.nn;

import basalt.core.BC;
import basalt.core.Tensor;
import java.util.Arrays;
import java.util.Objects;

/** Loss functions over batch-channel predictions. */
public enum Loss {
    CATEGORICAL_CROSSENTROPY {
        @Override
        public LossEvaluation evaluate(Tensor<BC> prediction, Tensor<BC> target) {
            validateSameShape(prediction, target);
            int batch = prediction.dimension(0);
            int classes = prediction.dimension(1);
            double[] p = prediction.data();
            double[] t = target.data();
            double[] gradient = new double[p.length];
            double[] logitGradient = new double[p.length];
            double loss = 0.0;
            for (int row = 0; row < batch; row++) {
                double targetSum = 0.0;
                for (int column = 0; column < classes; column++) {
                    int index = row * classes + column;
                    requireFinite(p[index], "prediction");
                    requireFinite(t[index], "target");
                    if (p[index] < 0.0 || p[index] > 1.0) {
                        throw new IllegalArgumentException("Categorical predictions must be in [0, 1]");
                    }
                    if (t[index] < 0.0) {
                        throw new IllegalArgumentException("Categorical targets must be non-negative");
                    }
                    targetSum += t[index];
                    double probability = Math.max(p[index], EPSILON);
                    loss -= t[index] * Math.log(probability);
                    gradient[index] = -t[index] / probability / batch;
                    logitGradient[index] = (p[index] - t[index]) / batch;
                }
                if (Math.abs(targetSum - 1.0) > 1e-6) {
                    throw new IllegalArgumentException("Each categorical target row must sum to 1");
                }
            }
            return new LossEvaluation(loss / batch, Tensor.ofBC(batch, classes, gradient),
                    Tensor.ofBC(batch, classes, logitGradient));
        }
    },
    BINARY_CROSSENTROPY {
        @Override
        public LossEvaluation evaluate(Tensor<BC> prediction, Tensor<BC> target) {
            validateSameShape(prediction, target);
            double[] p = prediction.data();
            double[] t = target.data();
            double[] gradient = new double[p.length];
            double loss = 0.0;
            for (int i = 0; i < p.length; i++) {
                requireFinite(p[i], "prediction");
                requireFinite(t[i], "target");
                if (p[i] < 0.0 || p[i] > 1.0 || t[i] < 0.0 || t[i] > 1.0) {
                    throw new IllegalArgumentException("Binary predictions and targets must be in [0, 1]");
                }
                double probability = Math.min(1.0 - EPSILON, Math.max(EPSILON, p[i]));
                loss -= t[i] * Math.log(probability)
                        + (1.0 - t[i]) * Math.log1p(-probability);
                gradient[i] = (probability - t[i])
                        / (probability * (1.0 - probability)) / p.length;
            }
            return new LossEvaluation(loss / p.length,
                    Tensor.ofBC(prediction.dimension(0), prediction.dimension(1), gradient));
        }
    },
    MEAN_SQUARED_ERROR {
        @Override
        public LossEvaluation evaluate(Tensor<BC> prediction, Tensor<BC> target) {
            validateSameShape(prediction, target);
            double[] p = prediction.data();
            double[] t = target.data();
            double[] gradient = new double[p.length];
            double loss = 0.0;
            for (int i = 0; i < p.length; i++) {
                requireFinite(p[i], "prediction");
                requireFinite(t[i], "target");
                double error = p[i] - t[i];
                loss += error * error;
                gradient[i] = 2.0 * error / p.length;
            }
            return new LossEvaluation(loss / p.length,
                    Tensor.ofBC(prediction.dimension(0), prediction.dimension(1), gradient));
        }
    };

    private static final double EPSILON = 1e-12;

    public abstract LossEvaluation evaluate(Tensor<BC> prediction, Tensor<BC> target);

    private static void validateSameShape(Tensor<BC> prediction, Tensor<BC> target) {
        Objects.requireNonNull(prediction, "prediction");
        Objects.requireNonNull(target, "target");
        if (!Arrays.equals(prediction.shape(), target.shape())) {
            throw new IllegalArgumentException("Prediction and target shapes must match");
        }
    }

    private static void requireFinite(double value, String name) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(name + " values must be finite");
        }
    }
}
