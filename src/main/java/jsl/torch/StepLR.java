package jsl.torch;

import java.util.Objects;

/**
 * Decays the learning rate of the optimizer by {@code gamma} every {@code stepSize} epochs/steps,
 * matching {@code torch.optim.lr_scheduler.StepLR}.
 */
public final class StepLR implements LRScheduler {
    private final Optimizer optimizer;
    private final float baseLr;
    private final int stepSize;
    private final float gamma;
    private int currentStep;

    public StepLR(Optimizer optimizer, int stepSize, float gamma) {
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer");
        if (stepSize <= 0) throw new IllegalArgumentException("stepSize must be positive");
        if (!(gamma > 0.0f) || !Float.isFinite(gamma)) {
            throw new IllegalArgumentException("gamma must be positive and finite");
        }
        this.baseLr = optimizer.learningRate();
        this.stepSize = stepSize;
        this.gamma = gamma;
    }

    @Override
    public float step() {
        currentStep++;
        int decays = currentStep / stepSize;
        float nextLr = (float) (baseLr * Math.pow(gamma, decays));
        optimizer.setLearningRate(nextLr);
        return nextLr;
    }

    @Override
    public float currentLearningRate() {
        return optimizer.learningRate();
    }

    @Override
    public int currentStep() {
        return currentStep;
    }
}
