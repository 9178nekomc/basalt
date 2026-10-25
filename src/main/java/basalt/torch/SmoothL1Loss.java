package basalt.torch;

/** Smooth L1 (Huber) loss with configurable beta transition threshold. */
public final class SmoothL1Loss implements Loss {
    private final float beta;

    public SmoothL1Loss() {
        this(1.0f);
    }

    public SmoothL1Loss(float beta) {
        if (beta < 0.0f) throw new IllegalArgumentException("beta must be non-negative");
        this.beta = beta;
    }

    public float beta() {
        return beta;
    }

    @Override
    public Tensor loss(Tensor prediction, Tensor target) {
        return prediction.smoothL1Loss(target, beta);
    }
}
