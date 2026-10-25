package jsl.nn;

import java.util.Objects;

/** Immutable optimizer hyperparameters shared by every trainable layer. */
public record OptimizerConfig(
        Optimizer optimizer,
        double learningRate,
        double beta1,
        double beta2,
        double epsilon) {

    public OptimizerConfig {
        Objects.requireNonNull(optimizer, "optimizer");
        if (!Double.isFinite(learningRate) || learningRate <= 0.0) {
            throw new IllegalArgumentException("learningRate must be positive and finite");
        }
        if (!Double.isFinite(beta1) || beta1 < 0.0 || beta1 >= 1.0
                || !Double.isFinite(beta2) || beta2 < 0.0 || beta2 >= 1.0) {
            throw new IllegalArgumentException("Adam beta values must be in [0, 1)");
        }
        if (!Double.isFinite(epsilon) || epsilon <= 0.0) {
            throw new IllegalArgumentException("epsilon must be positive and finite");
        }
    }
}
