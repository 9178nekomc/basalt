package basalt.torch;

import basalt.core.BC;
import basalt.nativebridge.NativeContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Dense layer: {@code y = x W + b}. */
public final class Linear extends Module<BC, BC> {
    private final Tensor weight;
    private final Tensor bias;

    public Linear(NativeContext context, int inputSize, int outputSize) {
        this(context, inputSize, outputSize, true, 0L);
    }

    public Linear(NativeContext context, int inputSize, int outputSize, boolean useBias,
                  long seed) {
        float scale = (float) Math.sqrt(6.0 / (inputSize + outputSize));
        weight = Tensor.random(context, new int[]{inputSize, outputSize},
                seed + 17L * inputSize + outputSize, scale, true);
        bias = useBias ? Tensor.parameter(context, 1, outputSize) : null;
        if (bias != null) {
            bias.fill(0.0f);
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        int[] shape = input.shape();
        int last = shape[shape.length - 1];
        if (last != weight.shape()[0]) {
            throw new IllegalArgumentException(
                    "Linear expected last dimension " + weight.shape()[0] + ", got " + last);
        }
        int rows = input.size() / last;
        Tensor flat = shape.length == 2 ? input : input.reshape(rows, last);
        Tensor output = flat.matMul(weight);
        if (bias != null) {
            output = output.add(bias);
        }
        if (shape.length == 2) {
            return output;
        }
        int[] outputShape = shape.clone();
        outputShape[shape.length - 1] = weight.shape()[1];
        return output.reshape(outputShape);
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> parameters = new ArrayList<>();
        parameters.add(weight);
        if (bias != null) {
            parameters.add(bias);
        }
        return parameters;
    }

    @Override
    public List<Map.Entry<String, Tensor>> namedParameters() {
        List<Map.Entry<String, Tensor>> entries = new ArrayList<>();
        entries.add(Map.entry("weight", weight));
        if (bias != null) {
            entries.add(Map.entry("bias", bias));
        }
        return entries;
    }

    public Tensor weight() {
        return weight;
    }

    public Tensor bias() {
        return bias;
    }
}
