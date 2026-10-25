package basalt.torch;

/** Numerically stable Binary Cross-Entropy with Logits loss. */
public final class BCEWithLogitsLoss implements Loss {
    @Override
    public Tensor loss(Tensor prediction, Tensor target) {
        return prediction.bceWithLogitsLoss(target);
    }
}
