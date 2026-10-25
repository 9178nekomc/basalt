package jsl.torch;

import java.util.List;

/** Stochastic gradient descent with decoupled weight decay. */
public final class SGD extends Optimizer {
    private final float weightDecay;

    public SGD(List<Tensor> parameters, float learningRate) {
        this(parameters, learningRate, 0.0f);
    }

    public SGD(List<Tensor> parameters, float learningRate, float weightDecay) {
        super(parameters, learningRate);
        this.weightDecay = weightDecay;
    }

    @Override
    public void step() {
        for (Tensor parameter : parameters) {
            parameter.sgd(learningRate, weightDecay);
        }
    }
}
