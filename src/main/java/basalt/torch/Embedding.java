package basalt.torch;

import basalt.core.BC;
import basalt.core.BSF;
import basalt.nativebridge.NativeContext;
import java.util.List;
import java.util.Map;

/** Lookup table: [batch, sequence] integer-valued floats -> [batch, sequence, dim]. */
public final class Embedding extends Module<BC, BSF> {
    private final Tensor weight;

    public Embedding(NativeContext context, int vocabularySize, int dimension) {
        this(context, vocabularySize, dimension, 0L);
    }

    public Embedding(NativeContext context, int vocabularySize, int dimension, long seed) {
        float scale = (float) (1.0 / Math.sqrt(dimension));
        weight = Tensor.random(context, new int[]{vocabularySize, dimension},
                seed + 41L * vocabularySize + dimension, scale, true);
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.embedding(weight);
    }

    @Override
    public List<Tensor> parameters() {
        return List.of(weight);
    }

    @Override
    public List<Map.Entry<String, Tensor>> namedParameters() {
        return List.of(Map.entry("weight", weight));
    }

    public Tensor weight() {
        return weight;
    }
}
