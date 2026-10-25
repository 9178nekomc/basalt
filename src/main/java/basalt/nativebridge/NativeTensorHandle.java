package basalt.nativebridge;

import basalt.BasaltRuntimeException;

import java.lang.ref.Cleaner;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A Java handle to one native tensor.
 *
 * <p>The object stores only an opaque {@code long}. The native tensor owns a contiguous
 * {@code Vec<f32>} plus a lazily allocated gradient buffer; no Java array is pinned and no bulk
 * data crosses the JNI boundary unless the caller explicitly asks for {@link #data()} or
 * {@link #grad()}.</p>
 *
 * <p>Release is idempotent and happens exactly once: either through {@link #close()} or through
 * the {@link Cleaner}. The native side additionally rejects an unknown address, so even a buggy
 * double release cannot cause a double free.</p>
 */
public final class NativeTensorHandle implements AutoCloseable {
    private static final Cleaner CLEANER = Cleaner.create();

    private final NativeContext context;
    private final int[] shape;
    private final boolean requiresGrad;
    private final State state;
    private final Cleaner.Cleanable cleanable;

    NativeTensorHandle(NativeContext context, long pointer, int[] shape, boolean requiresGrad) {
        this.context = BasaltRuntimeException.requireNonNull(context, "context");
        this.shape = shape.clone();
        this.requiresGrad = requiresGrad;
        this.state = new State(context, pointer);
        this.cleanable = CLEANER.register(this, state);
    }

    public NativeContext context() {
        return context;
    }

    public int[] shape() {
        return shape.clone();
    }

    public int rank() {
        return shape.length;
    }

    public long size() {
        long elements = 1L;
        for (int dimension : shape) {
            elements *= dimension;
        }
        return elements;
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }

    /** Copies the native data into a new Java array. */
    public float[] data() {
        float[] output = new float[Math.toIntExact(size())];
        try (NativeContext.Guard guard = context.guard()) {
            NativeBindings.tensorData(guard.handle(), ptr(), output);
        }
        return output;
    }

    /** Copies the accumulated gradient; a tensor that never participated in backward yields zeros. */
    public float[] grad() {
        float[] output = new float[Math.toIntExact(size())];
        try (NativeContext.Guard guard = context.guard()) {
            NativeBindings.tensorGrad(guard.handle(), ptr(), output);
        }
        return output;
    }

    public void setData(float[] values) {
        BasaltRuntimeException.requireNonNull(values, "values");
        if (values.length != size()) {
            throw new IllegalArgumentException(
                    "data length " + values.length + " does not match tensor size " + size());
        }
        try (NativeContext.Guard guard = context.guard()) {
            NativeBindings.setTensorData(guard.handle(), ptr(), values);
        }
    }

    public void fill(float value) {
        try (NativeContext.Guard guard = context.guard()) {
            NativeBindings.fillTensor(guard.handle(), ptr(), value);
        }
    }

    public void randomize(long seed, float scale) {
        try (NativeContext.Guard guard = context.guard()) {
            NativeBindings.randomizeTensor(guard.handle(), ptr(), seed, scale);
        }
    }

    public void zeroGrad() {
        try (NativeContext.Guard guard = context.guard()) {
            NativeBindings.zeroGrad(guard.handle(), ptr());
        }
    }

    public void sgd(float learningRate) {
        sgd(learningRate, 0.0f);
    }

    public void sgd(float learningRate, float weightDecay) {
        try (NativeContext.Guard guard = context.guard()) {
            NativeBindings.sgdStep(guard.handle(), ptr(), learningRate, weightDecay);
        }
    }

    public void adam(float learningRate, float beta1, float beta2, float epsilon,
                     float weightDecay, long step) {
        try (NativeContext.Guard guard = context.guard()) {
            NativeBindings.adamStep(guard.handle(), ptr(), learningRate, beta1, beta2,
                    epsilon, weightDecay, step);
        }
    }

    /** L2 norm of the data (useful for diagnostics and gradient clipping). */
    public double norm() {
        try (NativeContext.Guard guard = context.guard()) {
            return NativeBindings.tensorNorm(guard.handle(), ptr());
        }
    }

    public double gradNorm() {
        try (NativeContext.Guard guard = context.guard()) {
            return NativeBindings.tensorGradNorm(guard.handle(), ptr());
        }
    }

    public boolean isFinite() {
        try (NativeContext.Guard guard = context.guard()) {
            return NativeBindings.tensorIsFinite(guard.handle(), ptr());
        }
    }

    long ptr() {
        long pointer = state.pointer.get();
        if (pointer == 0L) {
            throw new IllegalStateException("tensor handle is closed");
        }
        return pointer;
    }

    @Override
    public void close() {
        cleanable.clean();
    }

    @Override
    public String toString() {
        return "NativeTensor" + Arrays.toString(shape)
                + "[requiresGrad=" + requiresGrad + ", live=" + state.pointer.get() + "]";
    }

    private static final class State implements Runnable {
        private final NativeContext context;
        private final AtomicLong pointer;

        private State(NativeContext context, long pointer) {
            this.context = context;
            this.pointer = new AtomicLong(pointer);
        }

        @Override
        public void run() {
            long value = pointer.getAndSet(0L);
            if (value != 0L) {
                context.releaseTensorQuietly(value);
            }
        }
    }
}
