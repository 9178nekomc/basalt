package basalt.nn;

import basalt.core.BC;
import basalt.core.BCHW;
import basalt.core.Tensor;

/** BCHW -> BC layout transition; channel, height and width are flattened in that order. */
public final class Flatten implements Layer<BCHW, BC> {
    private int batch;
    private int channels;
    private int height;
    private int width;

    private Flatten() {
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Tensor<BC> forward(Tensor<BCHW> input) {
        batch = input.dimension(0);
        channels = input.dimension(1);
        height = input.dimension(2);
        width = input.dimension(3);
        return Tensor.ofBC(batch, channels * height * width, input.data());
    }

    @Override
    public Tensor<BCHW> backward(Tensor<BC> gradient) {
        if (channels == 0) {
            throw new IllegalStateException("Flatten backward requires a preceding forward pass");
        }
        if (gradient.dimension(0) != batch || gradient.dimension(1) != channels * height * width) {
            throw new IllegalArgumentException("Flatten gradient shape mismatch");
        }
        return Tensor.ofBCHW(batch, channels, height, width, gradient.data());
    }

    public static final class Builder {
        public Flatten build() {
            return new Flatten();
        }
    }
}
