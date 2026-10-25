package basalt.tests;

import basalt.core.BC;
import basalt.core.Tensor;
import basalt.nn.Conv2D;

/** This file must not compile: Conv2D accepts only Tensor<BCHW>. */
final class InvalidTensorUsage {
    void reject() {
        Conv2D convolution = Conv2D.builder().build();
        Tensor<BC> vectors = Tensor.ofBC(1, 4, 1, 2, 3, 4);
        convolution.forward(vectors);
    }
}
