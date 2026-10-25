package jsl.nn;

import jsl.core.BC;
import jsl.core.Tensor;
import java.util.Objects;

/**
 * Scalar loss and output gradient. Categorical cross-entropy additionally carries the
 * algebraically simplified Softmax-logit gradient used by CNN training for stability.
 */
public record LossEvaluation(double loss, Tensor<BC> gradient, Tensor<BC> softmaxLogitGradient) {
    public LossEvaluation(double loss, Tensor<BC> gradient) {
        this(loss, gradient, null);
    }

    public LossEvaluation {
        Objects.requireNonNull(gradient, "gradient");
    }

    public boolean hasSoftmaxLogitGradient() {
        return softmaxLogitGradient != null;
    }
}
