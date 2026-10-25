package jsl.torch;

import jsl.core.Layout;
import java.util.List;

/** Element-wise activation with layout-preserving type. */
public final class Activation<L extends Layout> extends Module<L, L> {
    public enum Kind {
        RELU, GELU, SILU, SIGMOID, TANH
    }

    private final Kind kind;

    public Activation(Kind kind) {
        this.kind = kind;
    }

    @Override
    public Tensor forward(Tensor input) {
        return switch (kind) {
            case RELU -> input.relu();
            case GELU -> input.gelu();
            case SILU -> input.silu();
            case SIGMOID -> input.sigmoid();
            case TANH -> input.tanh();
        };
    }

    @Override
    public List<Tensor> parameters() {
        return List.of();
    }
}
