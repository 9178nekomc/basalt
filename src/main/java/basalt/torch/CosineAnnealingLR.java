package basalt.torch;

import java.util.Objects;

/**
 * Cosine Annealing learning rate scheduler:
 * {@code lr_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * t / T_max))},
 * matching {@code torch.optim.lr_scheduler.CosineAnnealingLR}.
 */
public final class CosineAnnealingLR implements LRScheduler {
    private final Optimizer optimizer;
    private final float baseLr;
    private final int tMax;
    private final float etaMin;
    private int currentStep;

    public CosineAnnealingLR(Optimizer optimizer, int tMax) {
        this(optimizer, tMax, 0.0f);
    }

    public CosineAnnealingLR(Optimizer optimizer, int tMax, float etaMin) {
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer");
        if (tMax <= 0) throw new IllegalArgumentException("tMax must be positive");
        if (etaMin < 0.0f || !Float.isFinite(etaMin)) {
            throw new IllegalArgumentException("etaMin must be finite and non-negative");
        }
        this.baseLr = optimizer.learningRate();
        this.tMax = tMax;
        this.etaMin = etaMin;
    }

    @Override
    public float step() {
        currentStep++;
        int clamped = Math.min(currentStep, tMax);
        double cosine = Math.cos(Math.PI * clamped / tMax);
        float nextLr = (float) (etaMin + 0.5 * (baseLr - etaMin) * (1.0 + cosine));
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
