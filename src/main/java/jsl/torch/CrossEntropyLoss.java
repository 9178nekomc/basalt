package jsl.torch;

/** Fused softmax + negative log likelihood over [N, C] logits. */
public final class CrossEntropyLoss implements Loss {
    @Override
    public Tensor loss(Tensor prediction, Tensor target) {
        return prediction.crossEntropy(target);
    }
}
