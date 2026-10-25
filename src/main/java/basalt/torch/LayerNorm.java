package basalt.torch;

import basalt.core.Layout;
import basalt.nativebridge.NativeContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Layer normalization over the last dimension. */
public final class LayerNorm<L extends Layout> extends Module<L, L> {
    private final Tensor gamma;
    private final Tensor beta;
    private final float epsilon;

    public LayerNorm(NativeContext context, int... normalizedShape) {
        this(context, 1e-5f, normalizedShape);
    }

    public LayerNorm(NativeContext context, float epsilon, int... normalizedShape) {
        if (normalizedShape.length == 0) {
            throw new IllegalArgumentException("normalized shape must not be empty");
        }
        this.epsilon = epsilon;
        gamma = Tensor.parameter(context, normalizedShape);
        beta = Tensor.parameter(context, normalizedShape);
        gamma.fill(1.0f);
        beta.fill(0.0f);
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.layerNorm(gamma, beta, epsilon);
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> parameters = new ArrayList<>();
        parameters.add(gamma);
        parameters.add(beta);
        return parameters;
    }

    public float epsilon() {
        return epsilon;
    }

    public Tensor gamma() {
        return gamma;
    }

    public Tensor beta() {
        return beta;
    }

    @Override
    public List<Map.Entry<String, Tensor>> namedParameters() {
        List<Map.Entry<String, Tensor>> entries = new ArrayList<>();
        entries.add(Map.entry("gamma", gamma));
        entries.add(Map.entry("beta", beta));
        return entries;
    }
}
