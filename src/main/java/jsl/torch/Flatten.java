package jsl.torch;

import jsl.core.BCHW;
import jsl.core.BC;
import java.util.List;

/** Flattens NCHW to [batch, channels * height * width]. */
public final class Flatten extends Module<BCHW, BC> {
    @Override
    public Tensor forward(Tensor input) {
        int[] shape = input.shape();
        if (shape.length != 4) {
            throw new IllegalArgumentException("Flatten expects NCHW input");
        }
        return input.reshape(shape[0], shape[1] * shape[2] * shape[3]);
    }

    @Override
    public List<Tensor> parameters() {
        return List.of();
    }
}
