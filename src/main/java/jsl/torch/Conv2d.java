package jsl.torch;

import jsl.core.BCHW;
import jsl.nativebridge.NativeContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** 2-D convolution over NCHW tensors, with optional bias shaped {@code [1, C, 1, 1]}. */
public final class Conv2d extends Module<BCHW, BCHW> {
    private final Tensor weight;
    private final Tensor bias;
    private final int kernelHeight;
    private final int kernelWidth;
    private final int strideHeight;
    private final int strideWidth;
    private final int paddingHeight;
    private final int paddingWidth;

    public Conv2d(NativeContext context, int inputChannels, int outputChannels, int kernelHeight,
                  int kernelWidth) {
        this(context, inputChannels, outputChannels, kernelHeight, kernelWidth, 1, 1, 0, 0,
                true, 0L);
    }

    public Conv2d(NativeContext context, int inputChannels, int outputChannels, int kernelHeight,
                  int kernelWidth, int strideHeight, int strideWidth, int paddingHeight,
                  int paddingWidth, boolean useBias, long seed) {
        if (inputChannels <= 0 || outputChannels <= 0 || kernelHeight <= 0 || kernelWidth <= 0) {
            throw new IllegalArgumentException("conv2d dimensions must be positive");
        }
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.strideHeight = strideHeight;
        this.strideWidth = strideWidth;
        this.paddingHeight = paddingHeight;
        this.paddingWidth = paddingWidth;
        int fanIn = inputChannels * kernelHeight * kernelWidth;
        int fanOut = outputChannels * kernelHeight * kernelWidth;
        float scale = (float) Math.sqrt(6.0 / (fanIn + fanOut));
        weight = Tensor.random(context,
                new int[]{outputChannels, inputChannels, kernelHeight, kernelWidth},
                seed + 31L * inputChannels + outputChannels, scale, true);
        bias = useBias ? Tensor.parameter(context, 1, outputChannels, 1, 1) : null;
        if (bias != null) {
            bias.fill(0.0f);
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor output = input.conv2d(weight, kernelHeight, kernelWidth, strideHeight, strideWidth,
                paddingHeight, paddingWidth);
        return bias == null ? output : output.add(bias);
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
}
