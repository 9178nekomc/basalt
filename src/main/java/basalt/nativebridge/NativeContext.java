package basalt.nativebridge;

import basalt.BasaltRuntimeException;

import java.lang.ref.Cleaner;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Owns a native tensor context: tensor store, autodiff graph, rayon pools and optimizer state.
 *
 * <p>Lifecycle is enforced with a read/write lock. Every JNI call runs under the read lock, while
 * {@link #close()} takes the write lock and destroys the context exactly once. Tensor handles hold
 * a strong reference to their context, so a context cannot be garbage-collected before its
 * tensors; conversely, closing a context invalidates every handle and makes later use fail fast
 * instead of dereferencing freed memory.</p>
 */
public final class NativeContext implements AutoCloseable {
    private static final Cleaner CLEANER = Cleaner.create();

    private final ReentrantReadWriteLock lifecycle = new ReentrantReadWriteLock();
    private final AtomicLong handle;
    private final State state;
    private final Cleaner.Cleanable cleanable;
    private final int threads;

    private NativeContext(int threads) {
        NativeLibrary.requireAvailable();
        long pointer = NativeBindings.createContext();
        if (pointer == 0L) {
            throw new IllegalStateException("native context allocation returned null");
        }
        this.handle = new AtomicLong(pointer);
        this.threads = threads;
        this.state = new State(handle);
        this.cleanable = CLEANER.register(this, state);
    }

    /** Creates a context that uses all available processors for native kernels. */
    public static NativeContext create() {
        return new NativeContext(0);
    }

    /** Creates a context with an explicit native thread count ({@code 0} means auto). */
    public static NativeContext create(int threads) {
        if (threads < 0 || threads > 256) {
            throw new IllegalArgumentException("threads must be in [0, 256]");
        }
        return new NativeContext(threads);
    }

    /** Effective thread hint passed to native kernels; {@code 0} lets Rust choose. */
    public int threads() {
        return threads;
    }

    public boolean isOpen() {
        return handle.get() != 0L;
    }

    /**
     * Acquires the lifecycle read lock and returns the native pointer. The caller must close the
     * guard after the JNI call; try-with-resources is the intended pattern.
     */
    public Guard guard() {
        lifecycle.readLock().lock();
        long pointer = handle.get();
        if (pointer == 0L) {
            lifecycle.readLock().unlock();
            throw new IllegalStateException("native context is closed");
        }
        return new Guard(this, pointer);
    }

    public NativeTensorHandle zeros(int... shape) {
        return create(shape, false);
    }

    /** Zero-initialised trainable tensor ({@code requires_grad = true}). */
    public NativeTensorHandle parameter(int... shape) {
        return create(shape, true);
    }

    public NativeTensorHandle of(float[] data, int... shape) {
        return of(data, false, shape);
    }

    public NativeTensorHandle of(float[] data, boolean requiresGrad, int... shape) {
        BasaltRuntimeException.requireNonNull(data, "data");
        BasaltRuntimeException.requireNonNull(shape, "shape");
        long expected = elementCount(shape);
        if (data.length != expected) {
            throw new IllegalArgumentException(
                    "data length " + data.length + " does not match shape " + expected);
        }
        try (Guard guard = guard()) {
            long pointer = NativeBindings.createTensorFromData(
                    guard.handle(), data, shape, requiresGrad);
            if (pointer == 0L) {
                throw new IllegalStateException("native tensor allocation returned null");
            }
            return new NativeTensorHandle(this, pointer, shape, requiresGrad);
        }
    }

    public NativeTensorHandle random(int[] shape, long seed, float scale, boolean requiresGrad) {
        NativeTensorHandle tensor = create(shape, requiresGrad);
        try {
            tensor.randomize(seed, scale);
            return tensor;
        } catch (RuntimeException failure) {
            tensor.close();
            throw failure;
        }
    }

    private NativeTensorHandle create(int[] shape, boolean requiresGrad) {
        BasaltRuntimeException.requireNonNull(shape, "shape");
        elementCount(shape);
        try (Guard guard = guard()) {
            long pointer = NativeBindings.createTensor(guard.handle(), shape, requiresGrad);
            if (pointer == 0L) {
                throw new IllegalStateException("native tensor allocation returned null");
            }
            return new NativeTensorHandle(this, pointer, shape, requiresGrad);
        }
    }

    private static long elementCount(int[] shape) {
        if (shape.length == 0) {
            throw new IllegalArgumentException("shape must not be empty");
        }
        long elements = 1L;
        for (int dimension : shape) {
            if (dimension <= 0) {
                throw new IllegalArgumentException(
                        "shape dimensions must be positive, received " + dimension);
            }
            elements = Math.multiplyExact(elements, dimension);
        }
        if (elements > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("tensor is too large for a Java float[] view");
        }
        return elements;
    }

    public void backward(NativeTensorHandle root) {
        BasaltRuntimeException.requireNonNull(root, "root");
        if (root.context() != this) {
            throw new IllegalArgumentException("root tensor belongs to a different context");
        }
        try (Guard guard = guard()) {
            NativeBindings.backward(guard.handle(), root.ptr(), threads);
        }
    }

    /** Clips gradients of the given tensors by global L2 norm; returns the pre-clip norm. */
    public double clipGradNorm(java.util.List<NativeTensorHandle> tensors, float maxNorm) {
        long[] pointers = new long[tensors.size()];
        for (int index = 0; index < pointers.length; index++) {
            NativeTensorHandle tensor = tensors.get(index);
            if (tensor.context() != this) {
                throw new IllegalArgumentException("tensor belongs to a different context");
            }
            pointers[index] = tensor.ptr();
        }
        try (Guard guard = guard()) {
            return NativeBindings.clipGradNorm(guard.handle(), pointers, maxNorm);
        }
    }

    public void clearGraph() {
        try (Guard guard = guard()) {
            NativeBindings.clearGraph(guard.handle());
        }
    }

    public int graphSize() {
        try (Guard guard = guard()) {
            return NativeBindings.graphSize(guard.handle());
        }
    }

    /** Returns {@code [liveTensors, liveBytes, graphNodes]}. */
    public long[] stats() {
        long[] output = new long[3];
        try (Guard guard = guard()) {
            NativeBindings.contextStats(guard.handle(), output);
        }
        return output;
    }

    void releaseTensorQuietly(long pointer) {
        if (pointer == 0L || !isOpen()) {
            return;
        }
        try (Guard guard = guard()) {
            NativeBindings.releaseTensor(guard.handle(), pointer);
        } catch (IllegalStateException ignored) {
            // The context was closed between the liveness check and the guard; native teardown
            // has already released every tensor in that case.
        }
    }

    @Override
    public void close() {
        lifecycle.writeLock().lock();
        try {
            long pointer = handle.getAndSet(0L);
            try {
                if (pointer != 0L) {
                    NativeBindings.destroyContext(pointer);
                }
            } finally {
                cleanable.clean();
            }
        } finally {
            lifecycle.writeLock().unlock();
        }
    }

    @Override
    public String toString() {
        return "NativeContext[open=" + isOpen() + ", threads=" + threads + "]";
    }

    /** Read-lock guard; closing it releases the lock, not the context. */
    public static final class Guard implements AutoCloseable {
        private final NativeContext owner;
        private final long pointer;
        private boolean closed;

        private Guard(NativeContext owner, long pointer) {
            this.owner = owner;
            this.pointer = pointer;
        }

        public long handle() {
            if (closed) {
                throw new IllegalStateException("native context guard is closed");
            }
            return pointer;
        }

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                owner.lifecycle.readLock().unlock();
            }
        }
    }

    private static final class State implements Runnable {
        private final AtomicLong handle;

        private State(AtomicLong handle) {
            this.handle = handle;
        }

        @Override
        public void run() {
            long pointer = handle.getAndSet(0L);
            if (pointer == 0L) {
                return;
            }
            try {
                NativeBindings.destroyContext(pointer);
            } catch (RuntimeException ignored) {
                // A Cleaner thread must not propagate; the context registry makes a repeated
                // destroy a no-op, so there is nothing actionable here.
            }
        }
    }
}
