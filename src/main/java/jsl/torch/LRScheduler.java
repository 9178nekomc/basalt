package jsl.torch;

/**
 * Contract for learning rate schedulers that adjust {@link Optimizer#setLearningRate(float)}
 * across training epochs or steps.
 */
public interface LRScheduler {
    float step();

    float currentLearningRate();

    int currentStep();
}
