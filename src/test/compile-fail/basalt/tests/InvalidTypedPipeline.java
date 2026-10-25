package basalt.tests;

import basalt.core.BCHW;
import basalt.nativebridge.NativeContext;
import basalt.torch.Linear;
import basalt.torch.Sequential;

/** Must not compile: a Linear (BC -> BC) cannot follow a BCHW output. */
public final class InvalidTypedPipeline {
    private InvalidTypedPipeline() {
    }

    static Object build(NativeContext context) {
        return Sequential.<BCHW>start().add(new Linear(context, 4, 2));
    }
}
