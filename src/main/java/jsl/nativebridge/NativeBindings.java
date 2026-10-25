package jsl.nativebridge;

import java.nio.ByteBuffer;

/**
 * Raw JNI surface. Package-private on purpose: callers must go through {@link NativeContext} and
 * {@link NativeTensorHandle}, which enforce lifecycle, ownership and argument validation.
 *
 * <p>Every method that returns {@code int} returns {@code 0} on success. The Rust side throws
 * {@link IllegalStateException} before returning a negative status, so a Java caller can never
 * silently continue after a rejected operation.</p>
 */
final class NativeBindings {
    private NativeBindings() {
    }

    static native int available();

    static native int version();

    static native long createContext();

    static native int destroyContext(long context);

    static native int contextStats(long context, long[] output);

    static native int clearGraph(long context);

    static native int graphSize(long context);

    static native long createTensor(long context, int[] shape, boolean requiresGrad);

    static native long createTensorFromData(long context, float[] data, int[] shape,
                                            boolean requiresGrad);

    static native int setTensorData(long context, long tensor, float[] data);

    static native int tensorData(long context, long tensor, float[] output);

    static native int tensorGrad(long context, long tensor, float[] output);

    static native int[] tensorShape(long context, long tensor);

    static native int fillTensor(long context, long tensor, float value);

    static native int randomizeTensor(long context, long tensor, long seed, float scale);

    static native int zeroGrad(long context, long tensor);

    static native int releaseTensor(long context, long tensor);

    static native int sgdStep(long context, long tensor, float learningRate, float weightDecay);

    static native int adamStep(long context, long tensor, float learningRate, float beta1,
                               float beta2, float epsilon, float weightDecay, long step);

    static native double tensorNorm(long context, long tensor);

    static native double clipGradNorm(long context, long[] tensors, float maxNorm);

    static native double tensorGradNorm(long context, long tensor);

    static native boolean tensorIsFinite(long context, long tensor);

    static native int executeQueue(long context, ByteBuffer commands, int count, int threads);

    static native int backward(long context, long root, int threads);
}
