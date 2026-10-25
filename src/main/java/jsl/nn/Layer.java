package jsl.nn;

import jsl.core.Layout;
import jsl.core.Tensor;
import java.util.List;

/** A differentiable transformation with compile-time input and output layouts. */
public interface Layer<I extends Layout, O extends Layout> {
    Tensor<O> forward(Tensor<I> input);

    Tensor<I> backward(Tensor<O> gradient);

    default void update(OptimizerConfig optimizer) {
        // Parameter-free layers need no update.
    }

    /**
     * Live parameter views for gradient checking, gradient clipping and persistence.
     * Parameter-free layers return an empty list.
     */
    default List<ParameterRef> parameters() {
        return ParameterRef.none();
    }
}
