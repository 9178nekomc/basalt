package jsl.torch;

import java.util.List;

/** Adam with decoupled weight decay and bias correction. */
public final class AdamW extends Optimizer {
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;
    private long step;

    public AdamW(List<Tensor> parameters, float learningRate) {
        this(parameters, learningRate, 0.9f, 0.999f, 1e-8f, 0.01f);
    }

    public AdamW(List<Tensor> parameters, float learningRate, float beta1, float beta2,
                 float epsilon, float weightDecay) {
        super(parameters, learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
    }

    @Override
    public void step() {
        step++;
        for (Tensor parameter : parameters) {
            parameter.adam(learningRate, beta1, beta2, epsilon, weightDecay, step);
        }
    }

    public long stepCount() {
        return step;
    }
}
