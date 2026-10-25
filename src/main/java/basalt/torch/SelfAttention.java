package basalt.torch;

import basalt.core.BSF;
import basalt.nativebridge.NativeContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Single-head self-attention over [batch, time, channels].
 *
 * <p>Built entirely from the public tensor API: Q/K/V projections, batched matmul, scaled
 * softmax, optional causal mask, and output projection. Multi-head attention can be composed by
 * reshaping the channel dimension before this module.</p>
 */
public final class SelfAttention extends Module<BSF, BSF> {
    private final Linear query;
    private final Linear key;
    private final Linear value;
    private final Linear output;
    private final Tensor scale;
    private final boolean causal;

    public SelfAttention(NativeContext context, int dimension, boolean causal, long seed) {
        query = new Linear(context, dimension, dimension, true, seed + 101);
        key = new Linear(context, dimension, dimension, true, seed + 211);
        value = new Linear(context, dimension, dimension, true, seed + 307);
        output = new Linear(context, dimension, dimension, true, seed + 401);
        scale = Tensor.scalar(context, (float) (1.0 / Math.sqrt(dimension)));
        this.causal = causal;
    }

    @Override
    public Tensor forward(Tensor input) {
        int[] shape = input.shape();
        if (shape.length != 3) {
            throw new IllegalArgumentException("SelfAttention expects [batch, time, channels]");
        }
        int time = shape[1];
        Tensor q = query.forward(input);
        Tensor k = key.forward(input);
        Tensor v = value.forward(input);
        Tensor scores = q.batchedMatMul(k.permute(0, 2, 1)).mul(scale);
        if (causal) {
            float[] mask = new float[time * time];
            for (int row = 0; row < time; row++) {
                for (int column = 0; column < time; column++) {
                    if (column > row) {
                        mask[row * time + column] = -1.0e9f;
                    }
                }
            }
            scores = scores.add(Tensor.of(input.context(), mask, time, time));
        }
        Tensor weights = scores.softmax(2);
        Tensor context = weights.batchedMatMul(v);
        return output.forward(context);
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> parameters = new ArrayList<>();
        parameters.addAll(query.parameters());
        parameters.addAll(key.parameters());
        parameters.addAll(value.parameters());
        parameters.addAll(output.parameters());
        return parameters;
    }

    @Override
    public List<Map.Entry<String, Tensor>> namedParameters() {
        List<Map.Entry<String, Tensor>> entries = new ArrayList<>();
        addPrefixed(entries, "query.", query);
        addPrefixed(entries, "key.", key);
        addPrefixed(entries, "value.", value);
        addPrefixed(entries, "output.", output);
        return entries;
    }

    private static void addPrefixed(List<Map.Entry<String, Tensor>> entries, String prefix,
                                    Module<?, ?> module) {
        for (Map.Entry<String, Tensor> entry : module.namedParameters()) {
            entries.add(Map.entry(prefix + entry.getKey(), entry.getValue()));
        }
    }

    @Override
    public void close() {
        query.close();
        key.close();
        value.close();
        output.close();
        scale.close();
    }
}
