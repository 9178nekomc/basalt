package jsl.torch;

import jsl.core.BCHW;
import jsl.nativebridge.NativeContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Depthwise Separable 2-D Convolution (MobileNet / Xception style): a depthwise spatial
 * convolution ({@code groups = inputChannels}) followed by a {@code 1x1} pointwise channel-mixing
 * convolution.
 */
public final class DepthwiseSeparableConv2d extends Module<BCHW, BCHW> {
    private final Conv2d depthwise;
    private final Conv2d pointwise;

    public DepthwiseSeparableConv2d(NativeContext context, int inputChannels, int outputChannels,
                                    int kernelSize, int stride, int padding, long seed) {
        this(context, inputChannels, outputChannels, kernelSize, kernelSize,
                stride, stride, padding, padding, 1, 1, true, seed);
    }

    public DepthwiseSeparableConv2d(NativeContext context, int inputChannels, int outputChannels,
                                    int kernelHeight, int kernelWidth, int strideHeight,
                                    int strideWidth, int paddingHeight, int paddingWidth,
                                    int dilationHeight, int dilationWidth, boolean useBias,
                                    long seed) {
        this.depthwise = new Conv2d(context, inputChannels, inputChannels,
                kernelHeight, kernelWidth, strideHeight, strideWidth,
                paddingHeight, paddingWidth, dilationHeight, dilationWidth,
                inputChannels, false, seed + 17L);
        this.pointwise = new Conv2d(context, inputChannels, outputChannels,
                1, 1, 1, 1, 0, 0, 1, 1, 1, useBias, seed + 53L);
    }

    @Override
    public Tensor forward(Tensor input) {
        return pointwise.forward(depthwise.forward(input));
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> params = new ArrayList<>();
        params.addAll(depthwise.parameters());
        params.addAll(pointwise.parameters());
        return params;
    }

    @Override
    public List<Map.Entry<String, Tensor>> namedParameters() {
        List<Map.Entry<String, Tensor>> entries = new ArrayList<>();
        for (Map.Entry<String, Tensor> e : depthwise.namedParameters()) {
            entries.add(Map.entry("depthwise." + e.getKey(), e.getValue()));
        }
        for (Map.Entry<String, Tensor> e : pointwise.namedParameters()) {
            entries.add(Map.entry("pointwise." + e.getKey(), e.getValue()));
        }
        return entries;
    }

    public Conv2d depthwise() {
        return depthwise;
    }

    public Conv2d pointwise() {
        return pointwise;
    }

    @Override
    public void train() {
        super.train();
        depthwise.train();
        pointwise.train();
    }

    @Override
    public void eval() {
        super.eval();
        depthwise.eval();
        pointwise.eval();
    }

    @Override
    public void close() {
        depthwise.close();
        pointwise.close();
    }
}
