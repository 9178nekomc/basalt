package jsl.torch;

/** Mean Absolute Error (L1) loss. */
public final class L1Loss implements Loss {
    @Override
    public Tensor loss(Tensor prediction, Tensor target) {
        return prediction.l1Loss(target);
    }
}
