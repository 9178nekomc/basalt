package basalt.torch;

import basalt.core.Layout;
import java.util.List;

/** Inverted dropout; evaluation is the identity. */
public final class Dropout<L extends Layout> extends Module<L, L> {
    private final float probability;
    private long seed;

    public Dropout(float probability) {
        this(probability, 0x9E3779B97F4A7C15L);
    }

    public Dropout(float probability, long seed) {
        if (!(probability >= 0.0f && probability < 1.0f)) {
            throw new IllegalArgumentException("dropout probability must be in [0, 1)");
        }
        this.probability = probability;
        this.seed = seed;
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.dropout(probability, isTraining(), seed++);
    }

    @Override
    public List<Tensor> parameters() {
        return List.of();
    }
}
