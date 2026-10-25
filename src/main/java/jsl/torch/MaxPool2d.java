package jsl.torch;

import jsl.core.BCHW;
import java.util.List;

/** Max pooling over NCHW tensors. */
public final class MaxPool2d extends Module<BCHW, BCHW> {
    private final int kernelHeight;
    private final int kernelWidth;
    private final int strideHeight;
    private final int strideWidth;

    public MaxPool2d(int kernelSize) {
        this(kernelSize, kernelSize, kernelSize, kernelSize);
    }

    public MaxPool2d(int kernelHeight, int kernelWidth, int strideHeight, int strideWidth) {
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.strideHeight = strideHeight;
        this.strideWidth = strideWidth;
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.maxPool2d(kernelHeight, kernelWidth, strideHeight, strideWidth);
    }

    @Override
    public List<Tensor> parameters() {
        return List.of();
    }
}
