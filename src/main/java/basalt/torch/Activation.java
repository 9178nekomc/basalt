package basalt.torch;

import basalt.core.Layout;
import java.util.List;

/** Element-wise activation with layout-preserving type. */
public final class Activation<L extends Layout> extends Module<L, L> {
    public enum Kind {
        RELU, GELU, SILU, SIGMOID, TANH, LEAKY_RELU, ELU
    }

    private final Kind kind;
    private final float alpha;

    public Activation(Kind kind) {
        this(kind, kind == Kind.LEAKY_RELU ? 0.01f : 1.0f);
    }

    public Activation(Kind kind, float alpha) {
        this.kind = kind;
        this.alpha = alpha;
    }

    public Kind kind() {
        return kind;
    }

    public float alpha() {
        return alpha;
    }

    @Override
    public Tensor forward(Tensor input) {
        return switch (kind) {
            case RELU -> input.relu();
            case GELU -> input.gelu();
            case SILU -> input.silu();
            case SIGMOID -> input.sigmoid();
            case TANH -> input.tanh();
            case LEAKY_RELU -> input.leakyRelu(alpha);
            case ELU -> input.elu(alpha);
        };
    }

    @Override
    public List<Tensor> parameters() {
        return List.of();
    }
}
