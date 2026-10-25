package jsl.torch;

/** A differentiable scalar loss. */
public interface Loss {
    Tensor loss(Tensor prediction, Tensor target);
}
