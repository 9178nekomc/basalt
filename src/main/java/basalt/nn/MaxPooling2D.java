package basalt.nn;

import basalt.core.BCHW;
import basalt.core.Tensor;

/** Non-overlapping by default, with cached argmax indices for exact backward routing. */
public final class MaxPooling2D implements Layer<BCHW, BCHW> {
    private final int poolHeight;
    private final int poolWidth;
    private final int strideHeight;
    private final int strideWidth;
    private int inputBatch;
    private int inputChannels;
    private int inputHeight;
    private int inputWidth;
    private int outputHeight;
    private int outputWidth;
    private int[] argmax;

    private MaxPooling2D(Builder builder) {
        this.poolHeight = builder.poolHeight;
        this.poolWidth = builder.poolWidth;
        this.strideHeight = builder.strideHeight == 0 ? builder.poolHeight : builder.strideHeight;
        this.strideWidth = builder.strideWidth == 0 ? builder.poolWidth : builder.strideWidth;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Tensor<BCHW> forward(Tensor<BCHW> input) {
        int batch = input.dimension(0);
        int channels = input.dimension(1);
        inputBatch = batch;
        inputChannels = channels;
        inputHeight = input.dimension(2);
        inputWidth = input.dimension(3);
        if (inputHeight < poolHeight || inputWidth < poolWidth) {
            throw new IllegalArgumentException("Pooling window must not exceed the input spatial shape");
        }
        outputHeight = (inputHeight - poolHeight) / strideHeight + 1;
        outputWidth = (inputWidth - poolWidth) / strideWidth + 1;
        Tensor<BCHW> output = Tensor.zerosBCHW(batch, channels, outputHeight, outputWidth);
        argmax = new int[output.size()];
        java.util.Arrays.fill(argmax, -1);
        double[] source = input.data();
        double[] target = output.data();

        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        int outputIndex = ((b * channels + c) * outputHeight + oh) * outputWidth + ow;
                        double maximum = Double.NEGATIVE_INFINITY;
                        for (int kh = 0; kh < poolHeight; kh++) {
                            int ih = oh * strideHeight + kh;
                            for (int kw = 0; kw < poolWidth; kw++) {
                                int iw = ow * strideWidth + kw;
                                int inputIndex = ((b * channels + c) * inputHeight + ih) * inputWidth + iw;
                                double value = source[inputIndex];
                                if (Double.isNaN(value)) {
                                    throw new IllegalArgumentException("MaxPooling2D does not accept NaN input");
                                }
                                if (argmax[outputIndex] < 0 || value > maximum) {
                                    maximum = value;
                                    argmax[outputIndex] = inputIndex;
                                }
                            }
                        }
                        target[outputIndex] = maximum;
                    }
                }
            }
        }
        return output;
    }

    @Override
    public Tensor<BCHW> backward(Tensor<BCHW> gradient) {
        if (argmax == null) {
            throw new IllegalStateException("MaxPooling2D backward requires a preceding forward pass");
        }
        int expectedHeight = gradient.dimension(2);
        int expectedWidth = gradient.dimension(3);
        if (gradient.dimension(0) != inputBatch || gradient.dimension(1) != inputChannels
                || expectedHeight != outputHeight || expectedWidth != outputWidth
                || gradient.size() != argmax.length) {
            throw new IllegalArgumentException("MaxPooling2D gradient shape mismatch");
        }
        Tensor<BCHW> inputGradient = Tensor.zerosBCHW(
                inputBatch, inputChannels, inputHeight, inputWidth);
        double[] target = inputGradient.data();
        double[] source = gradient.data();
        for (int i = 0; i < source.length; i++) {
            if (argmax[i] < 0) {
                throw new IllegalStateException("Pooling argmax cache is incomplete");
            }
            target[argmax[i]] += source[i];
        }
        return inputGradient;
    }

    int poolHeight() {
        return poolHeight;
    }

    int poolWidth() {
        return poolWidth;
    }

    int strideHeight() {
        return strideHeight;
    }

    int strideWidth() {
        return strideWidth;
    }

    public static final class Builder {
        private int poolHeight = 2;
        private int poolWidth = 2;
        private int strideHeight;
        private int strideWidth;

        public Builder poolSize(int height, int width) {
            this.poolHeight = height;
            this.poolWidth = width;
            return this;
        }

        public Builder strides(int height, int width) {
            this.strideHeight = height;
            this.strideWidth = width;
            return this;
        }

        public MaxPooling2D build() {
            if (poolHeight <= 0 || poolWidth <= 0 || strideHeight < 0 || strideWidth < 0
                    || (strideHeight == 0) != (strideWidth == 0)) {
                throw new IllegalArgumentException(
                        "Pool sizes must be positive and strides must be both omitted or both positive");
            }
            return new MaxPooling2D(this);
        }
    }
}
