package jsl.tests;

import jsl.core.BCHW;
import jsl.core.Tensor;
import jsl.nn.LSTM;

/** This file must not compile: recurrent layers accept only Tensor&lt;BSF&gt;. */
final class InvalidSequenceUsage {
    void reject() {
        LSTM lstm = LSTM.builder().build();
        Tensor<BCHW> images = Tensor.ofBCHW(1, 1, 2, 2, 1, 2, 3, 4);
        lstm.forward(images);
    }
}
