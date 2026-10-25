package basalt.torch;

/** Mean squared error. */
public final class MSELoss implements Loss {
    @Override
    public Tensor loss(Tensor prediction, Tensor target) {
        return prediction.mse(target);
    }
}
