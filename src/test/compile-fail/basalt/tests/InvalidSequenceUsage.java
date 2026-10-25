package basalt.tests;

import basalt.core.BCHW;
import basalt.core.Tensor;
import basalt.nn.LSTM;

/** This file must not compile: recurrent layers accept only Tensor&lt;BSF&gt;. */
final class InvalidSequenceUsage {
    void reject() {
        LSTM lstm = LSTM.builder().build();
        Tensor<BCHW> images = Tensor.ofBCHW(1, 1, 2, 2, 1, 2, 3, 4);
        lstm.forward(images);
    }
}
