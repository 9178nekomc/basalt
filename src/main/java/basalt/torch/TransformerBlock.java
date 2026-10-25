package basalt.torch;

import basalt.core.BSF;
import basalt.nativebridge.NativeContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Pre-norm transformer block: attention residual + GELU MLP residual. */
public final class TransformerBlock extends Module<BSF, BSF> {
    private final LayerNorm<BSF> norm1;
    private final MultiheadAttention attention;
    private final LayerNorm<BSF> norm2;
    private final Linear feedForward1;
    private final Activation<BSF> activation;
    private final Linear feedForward2;
    private final Dropout<BSF> dropout;

    public TransformerBlock(NativeContext context, int dimension, int hiddenSize, boolean causal,
                            float dropoutProbability, long seed) {
        this(context, dimension, hiddenSize, 1, causal, dropoutProbability, seed);
    }

    public TransformerBlock(NativeContext context, int dimension, int hiddenSize, int heads,
                            boolean causal, float dropoutProbability, long seed) {
        norm1 = new LayerNorm<>(context, 1e-5f, dimension);
        attention = new MultiheadAttention(context, dimension, heads, causal,
                dropoutProbability, seed + 11);
        norm2 = new LayerNorm<>(context, 1e-5f, dimension);
        feedForward1 = new Linear(context, dimension, hiddenSize, true, seed + 23);
        activation = new Activation<>(Activation.Kind.GELU);
        feedForward2 = new Linear(context, hiddenSize, dimension, true, seed + 37);
        dropout = new Dropout<>(dropoutProbability, seed + 41);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor attended = attention.forward(norm1.forward(input));
        Tensor residual = input.add(dropout.forward(attended));
        Tensor hidden = feedForward2.forward(activation.forward(feedForward1.forward(
                norm2.forward(residual))));
        return residual.add(dropout.forward(hidden));
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> parameters = new ArrayList<>();
        parameters.addAll(norm1.parameters());
        parameters.addAll(attention.parameters());
        parameters.addAll(norm2.parameters());
        parameters.addAll(feedForward1.parameters());
        parameters.addAll(feedForward2.parameters());
        return parameters;
    }

    @Override
    public List<Map.Entry<String, Tensor>> namedParameters() {
        List<Map.Entry<String, Tensor>> entries = new ArrayList<>();
        addPrefixed(entries, "norm1.", norm1);
        addPrefixed(entries, "attention.", attention);
        addPrefixed(entries, "norm2.", norm2);
        addPrefixed(entries, "ffn1.", feedForward1);
        addPrefixed(entries, "ffn2.", feedForward2);
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
        norm1.train();
        attention.train();
        norm2.train();
        feedForward1.train();
        feedForward2.train();
        dropout.train();
    }

    @Override
    public void eval() {
        super.eval();
        norm1.eval();
        attention.eval();
        norm2.eval();
        feedForward1.eval();
        feedForward2.eval();
        dropout.eval();
    }

    @Override
    public void close() {
        norm1.close();
        attention.close();
        norm2.close();
        feedForward1.close();
        feedForward2.close();
        dropout.close();
    }
}
