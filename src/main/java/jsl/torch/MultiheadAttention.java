package jsl.torch;

import jsl.core.BSF;
import jsl.nativebridge.NativeContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Multi-head self-attention over [batch, time, channels], composed from reshape, permute,
 * batched matmul and softmax. This is the transformer primitive modern PyTorch users expect,
 * built entirely on the same public tensor API.
 */
public final class MultiheadAttention extends Module<BSF, BSF> {
    private final Linear query;
    private final Linear key;
    private final Linear value;
    private final Linear output;
    private final Dropout<BSF> dropout;
    private final Tensor scale;
    private final int heads;
    private final boolean causal;

    public MultiheadAttention(NativeContext context, int dimension, int heads, boolean causal,
                              float dropoutProbability, long seed) {
        if (dimension % heads != 0) {
            throw new IllegalArgumentException("dimension must be divisible by heads");
        }
        this.heads = heads;
        this.causal = causal;
        query = new Linear(context, dimension, dimension, true, seed + 101);
        key = new Linear(context, dimension, dimension, true, seed + 211);
        value = new Linear(context, dimension, dimension, true, seed + 307);
        output = new Linear(context, dimension, dimension, true, seed + 401);
        dropout = new Dropout<>(dropoutProbability, seed + 503);
        scale = Tensor.scalar(context, (float) (1.0 / Math.sqrt(dimension / heads)));
    }

    @Override
    public Tensor forward(Tensor input) {
        int[] shape = input.shape();
        if (shape.length != 3) {
            throw new IllegalArgumentException("attention expects [batch, time, channels]");
        }
        int batch = shape[0];
        int time = shape[1];
        int channels = shape[2];
        if (channels % heads != 0) {
            throw new IllegalArgumentException("channels must be divisible by heads");
        }
        int headDim = channels / heads;
        Tensor q = query.forward(input)
                .reshape(batch, time, heads, headDim)
                .permute(0, 2, 1, 3)
                .reshape(batch * heads, time, headDim);
        Tensor k = key.forward(input)
                .reshape(batch, time, heads, headDim)
                .permute(0, 2, 3, 1)
                .reshape(batch * heads, headDim, time);
        Tensor v = value.forward(input)
                .reshape(batch, time, heads, headDim)
                .permute(0, 2, 1, 3)
                .reshape(batch * heads, time, headDim);
        Tensor scores = q.batchedMatMul(k).mul(scale);
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
        Tensor weights = dropout.forward(scores.softmax(2));
        Tensor context = weights.batchedMatMul(v)
                .reshape(batch, heads, time, headDim)
                .permute(0, 2, 1, 3)
                .reshape(batch, time, channels);
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
    public void train() {
        super.train();
        dropout.train();
    }

    @Override
    public void eval() {
        super.eval();
        dropout.eval();
    }

    @Override
    public void close() {
        query.close();
        key.close();
        value.close();
        output.close();
        dropout.close();
        scale.close();
    }
}
