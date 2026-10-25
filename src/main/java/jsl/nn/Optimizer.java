package jsl.nn;

/** Supported first-order optimization algorithms. */
public enum Optimizer {
    SGD,
    ADAM,
    ADAMW;

    public OptimizerConfig withLearningRate(double learningRate) {
        return new OptimizerConfig(this, learningRate, 0.9, 0.999, 1e-8);
    }
}
