package basalt.torch;

import basalt.core.BCHW;
import basalt.nativebridge.NativeContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 2-D convolution over NCHW tensors, supporting standard, grouped ({@code groups > 1}), depthwise
 * ({@code groups == inputChannels}), and dilated ({@code dilation > 1}) convolutions.
 */
public final class Conv2d extends Module<BCHW, BCHW> {
    private final Tensor weight;
    private final Tensor bias;
    private final int inputChannels;
    private final int outputChannels;
    private final int kernelHeight;
    private final int kernelWidth;
    private final int strideHeight;
    private final int strideWidth;
    private final int paddingHeight;
    private final int paddingWidth;
    private final int dilationHeight;
    private final int dilationWidth;
    private final int groups;

    public Conv2d(NativeContext context, int inputChannels, int outputChannels, int kernelHeight,
                  int kernelWidth) {
        this(context, inputChannels, outputChannels, kernelHeight, kernelWidth, 1, 1, 0, 0,
                1, 1, 1, true, 0L);
    }

    public Conv2d(NativeContext context, int inputChannels, int outputChannels, int kernelHeight,
                  int kernelWidth, int strideHeight, int strideWidth, int paddingHeight,
                  int paddingWidth, boolean useBias, long seed) {
        this(context, inputChannels, outputChannels, kernelHeight, kernelWidth,
                strideHeight, strideWidth, paddingHeight, paddingWidth, 1, 1, 1, useBias, seed);
    }

    public Conv2d(NativeContext context, int inputChannels, int outputChannels, int kernelHeight,
                  int kernelWidth, int strideHeight, int strideWidth, int paddingHeight,
                  int paddingWidth, int dilationHeight, int dilationWidth, int groups,
                  boolean useBias, long seed) {
        if (inputChannels <= 0 || outputChannels <= 0 || kernelHeight <= 0 || kernelWidth <= 0
                || strideHeight <= 0 || strideWidth <= 0 || dilationHeight <= 0
                || dilationWidth <= 0 || groups <= 0) {
            throw new IllegalArgumentException("conv2d dimensions, strides, dilation and groups must be positive");
        }
        if (inputChannels % groups != 0 || outputChannels % groups != 0) {
            throw new IllegalArgumentException(
                    "inputChannels (" + inputChannels + ") and outputChannels (" + outputChannels
                            + ") must be divisible by groups (" + groups + ")");
        }
        this.inputChannels = inputChannels;
        this.outputChannels = outputChannels;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.strideHeight = strideHeight;
        this.strideWidth = strideWidth;
        this.paddingHeight = paddingHeight;
        this.paddingWidth = paddingWidth;
        this.dilationHeight = dilationHeight;
        this.dilationWidth = dilationWidth;
        this.groups = groups;
        int cinPerGroup = inputChannels / groups;
        int fanIn = cinPerGroup * kernelHeight * kernelWidth;
        int fanOut = (outputChannels / groups) * kernelHeight * kernelWidth;
        float scale = (float) Math.sqrt(6.0 / (fanIn + fanOut));
        weight = Tensor.random(context,
                new int[]{outputChannels, cinPerGroup, kernelHeight, kernelWidth},
                seed + 31L * inputChannels + outputChannels, scale, true);
        bias = useBias ? Tensor.parameter(context, 1, outputChannels, 1, 1) : null;
        if (bias != null) {
            bias.fill(0.0f);
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor output = input.conv2d(weight, kernelHeight, kernelWidth, strideHeight, strideWidth,
                paddingHeight, paddingWidth, dilationHeight, dilationWidth, groups);
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

    public Tensor bias() {
        return bias;
    }

    public int inputChannels() {
        return inputChannels;
    }

    public int outputChannels() {
        return outputChannels;
    }

    public int kernelHeight() {
        return kernelHeight;
    }

    public int kernelWidth() {
        return kernelWidth;
    }

    public int strideHeight() {
        return strideHeight;
    }

    public int strideWidth() {
        return strideWidth;
    }

    public int paddingHeight() {
        return paddingHeight;
    }

    public int paddingWidth() {
        return paddingWidth;
    }

    public int dilationHeight() {
        return dilationHeight;
    }

    public int dilationWidth() {
        return dilationWidth;
    }

    public int groups() {
        return groups;
    }
}
