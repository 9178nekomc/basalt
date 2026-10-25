package jsl.tests;

import jsl.core.BCHW;
import jsl.nativebridge.NativeContext;
import jsl.torch.Linear;
import jsl.torch.Sequential;

/** Must not compile: a Linear (BC -> BC) cannot follow a BCHW output. */
public final class InvalidTypedPipeline {
    private InvalidTypedPipeline() {
    }

    static Object build(NativeContext context) {
        return Sequential.<BCHW>start().add(new Linear(context, 4, 2));
    }
}
