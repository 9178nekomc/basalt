package basalt.nn;

import basalt.core.BCHW;
import basalt.core.Tensor;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

/** NCHW-per-sample (BCHW) convolution with explicit loops and exact input/weight gradients. */
public final class Conv2D implements Layer<BCHW, BCHW> {
    private final int filters;
    private final int kernelHeight;
    private final int kernelWidth;
    private final int strideHeight;
    private final int strideWidth;
    private final Padding padding;
    private final Activation activation;
    private final long seed;
    private int inputChannels = -1;
    private Parameter kernels;
    private Parameter bias;
    private Tensor<BCHW> cachedInput;
    private double[] preActivation;
    private double[] activated;
    private double[] kernelGradient;
    private double[] biasGradient;
    private int outputHeight;
    private int outputWidth;
    private int padTop;
    private int padLeft;
    private boolean gradientReady;

    private Conv2D(Builder builder) {
        this.filters = builder.filters;
        this.kernelHeight = builder.kernelHeight;
        this.kernelWidth = builder.kernelWidth;
        this.strideHeight = builder.strideHeight;
        this.strideWidth = builder.strideWidth;
        this.padding = builder.padding;
        this.activation = builder.activation;
        this.seed = builder.seed;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Tensor<BCHW> forward(Tensor<BCHW> input) {
        gradientReady = false;
        int channels = input.dimension(1);
        if (inputChannels < 0) {
            initialize(channels);
        } else if (channels != inputChannels) {
            throw new IllegalArgumentException(
                    "Conv2D expected " + inputChannels + " channels but got " + channels);
        }
        cachedInput = input;
        int batch = input.dimension(0);
        int height = input.dimension(2);
        int width = input.dimension(3);
        outputHeight = outputDimension(height, kernelHeight, strideHeight);
        outputWidth = outputDimension(width, kernelWidth, strideWidth);
        padTop = padding == Padding.SAME
                ? totalPadding(height, kernelHeight, strideHeight, outputHeight) / 2 : 0;
        padLeft = padding == Padding.SAME
                ? totalPadding(width, kernelWidth, strideWidth, outputWidth) / 2 : 0;
        Tensor<BCHW> output = Tensor.zerosBCHW(batch, filters, outputHeight, outputWidth);
        preActivation = new double[output.size()];
        activated = new double[output.size()];
        double[] x = input.data();
        double[] w = kernels.values();
        double[] b = bias.values();

        for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
            for (int filter = 0; filter < filters; filter++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        double sum = b[filter];
                        for (int channel = 0; channel < channels; channel++) {
                            for (int kh = 0; kh < kernelHeight; kh++) {
                                int ih = oh * strideHeight + kh - padTop;
                                if (ih < 0 || ih >= height) {
                                    continue;
                                }
                                for (int kw = 0; kw < kernelWidth; kw++) {
                                    int iw = ow * strideWidth + kw - padLeft;
                                    if (iw < 0 || iw >= width) {
                                        continue;
                                    }
                                    int inputIndex = ((batchIndex * channels + channel) * height + ih) * width + iw;
                                    int kernelIndex = ((filter * channels + channel) * kernelHeight + kh)
                                            * kernelWidth + kw;
                                    sum += x[inputIndex] * w[kernelIndex];
                                }
                            }
                        }
                        int outputIndex = ((batchIndex * filters + filter) * outputHeight + oh)
                                * outputWidth + ow;
                        preActivation[outputIndex] = sum;
                        activated[outputIndex] = activation.applyScalar(sum);
                    }
                }
            }
        }
        System.arraycopy(activated, 0, output.data(), 0, activated.length);
        return output;
    }

    private int outputDimension(int input, int kernel, int stride) {
        int output = padding == Padding.SAME
                ? (input + stride - 1) / stride
                : (input - kernel) / stride + 1;
        if (output <= 0 || (padding == Padding.VALID && input < kernel)) {
            throw new IllegalArgumentException("Convolution kernel exceeds the input spatial shape");
        }
        return output;
    }

    private static int totalPadding(int input, int kernel, int stride, int output) {
        return Math.max(0, (output - 1) * stride + kernel - input);
    }

    private void initialize(int channels) {
        inputChannels = channels;
        int size = filters * channels * kernelHeight * kernelWidth;
        double[] values = new double[size];
        int fanIn = channels * kernelHeight * kernelWidth;
        int fanOut = filters * kernelHeight * kernelWidth;
        double limit = activation == Activation.RELU
                ? Math.sqrt(6.0 / fanIn)
                : Math.sqrt(6.0 / (fanIn + fanOut));
        Random random = new Random(seed);
        for (int i = 0; i < size; i++) {
            values[i] = (random.nextDouble() * 2.0 - 1.0) * limit;
        }
        kernels = new Parameter(values);
        bias = new Parameter(new double[filters]);
        kernelGradient = new double[size];
        biasGradient = new double[filters];
    }

    @Override
    public Tensor<BCHW> backward(Tensor<BCHW> gradient) {
        gradientReady = false;
        if (cachedInput == null) {
            throw new IllegalStateException("Conv2D backward requires a preceding forward pass");
        }
        int batch = cachedInput.dimension(0);
        int height = cachedInput.dimension(2);
        int width = cachedInput.dimension(3);
        if (gradient.dimension(0) != batch || gradient.dimension(1) != filters
                || gradient.dimension(2) != outputHeight || gradient.dimension(3) != outputWidth) {
            throw new IllegalArgumentException("Conv2D gradient shape mismatch");
        }
        Arrays.fill(kernelGradient, 0.0);
        Arrays.fill(biasGradient, 0.0);
        Tensor<BCHW> inputGradient = Tensor.zerosBCHW(batch, inputChannels, height, width);
        double[] dx = inputGradient.data();
        double[] x = cachedInput.data();
        double[] g = gradient.data();
        double[] w = kernels.values();

        for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
            for (int filter = 0; filter < filters; filter++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        int outputIndex = ((batchIndex * filters + filter) * outputHeight + oh)
                                * outputWidth + ow;
                        double localGradient = g[outputIndex] * activation.derivative(
                                preActivation[outputIndex], activated[outputIndex]);
                        biasGradient[filter] += localGradient;
                        for (int channel = 0; channel < inputChannels; channel++) {
                            for (int kh = 0; kh < kernelHeight; kh++) {
                                int ih = oh * strideHeight + kh - padTop;
                                if (ih < 0 || ih >= height) {
                                    continue;
                                }
                                for (int kw = 0; kw < kernelWidth; kw++) {
                                    int iw = ow * strideWidth + kw - padLeft;
                                    if (iw < 0 || iw >= width) {
                                        continue;
                                    }
                                    int inputIndex = ((batchIndex * inputChannels + channel) * height + ih)
                                            * width + iw;
                                    int kernelIndex = ((filter * inputChannels + channel) * kernelHeight + kh)
                                            * kernelWidth + kw;
                                    kernelGradient[kernelIndex] += localGradient * x[inputIndex];
                                    dx[inputIndex] += localGradient * w[kernelIndex];
                                }
                            }
                        }
                    }
                }
            }
        }
        gradientReady = true;
        return inputGradient;
    }

    @Override
    public void update(OptimizerConfig optimizer) {
        if (kernels == null || !gradientReady) {
            throw new IllegalStateException("Conv2D update requires forward and backward passes");
        }
        kernels.update(kernelGradient, optimizer);
        bias.update(biasGradient, optimizer);
        gradientReady = false;
    }

    int filters() {
        return filters;
    }

    int kernelHeight() {
        return kernelHeight;
    }

    int kernelWidth() {
        return kernelWidth;
    }

    int strideHeight() {
        return strideHeight;
    }

    int strideWidth() {
        return strideWidth;
    }

    Padding padding() {
        return padding;
    }

    Activation activation() {
        return activation;
    }

    long seed() {
        return seed;
    }

    int inputChannels() {
        return inputChannels;
    }

    double[] kernelValues() {
        return kernels.values();
    }

    double[] biasValues() {
        return bias.values();
    }

    void restore(int channels, double[] kernelValues, double[] biasValues) {
        initialize(channels);
        System.arraycopy(kernelValues, 0, kernels.values(), 0, kernelValues.length);
        System.arraycopy(biasValues, 0, bias.values(), 0, biasValues.length);
    }

    public static final class Builder {
        private int filters = 1;
        private int kernelHeight = 3;
        private int kernelWidth = 3;
        private int strideHeight = 1;
        private int strideWidth = 1;
        private Padding padding = Padding.VALID;
        private Activation activation = Activation.RELU;
        private long seed = 42;

        public Builder filters(int filters) {
            this.filters = filters;
            return this;
        }

        public Builder kernelSize(int height, int width) {
            this.kernelHeight = height;
            this.kernelWidth = width;
            return this;
        }

        public Builder strides(int height, int width) {
            this.strideHeight = height;
            this.strideWidth = width;
            return this;
        }

        public Builder padding(Padding padding) {
            this.padding = Objects.requireNonNull(padding, "padding");
            return this;
        }

        public Builder activation(Activation activation) {
            this.activation = Objects.requireNonNull(activation, "activation");
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Conv2D build() {
            if (filters <= 0 || kernelHeight <= 0 || kernelWidth <= 0
                    || strideHeight <= 0 || strideWidth <= 0) {
                throw new IllegalArgumentException("Filters, kernels and strides must be positive");
            }
            if (activation == Activation.SOFTMAX) {
                throw new IllegalArgumentException("Use spatial convolution activations, not SOFTMAX");
            }
            return new Conv2D(this);
        }
    }
}
