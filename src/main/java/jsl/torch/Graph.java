package jsl.torch;

import jsl.JSLRuntimeException;
import jsl.nativebridge.NativeContext;

/**
 * Minimal Java-side handle for the native autograd graph.
 *
 * <p>{@link Tensor#backward()} is the common entry; this wrapper exists for explicit graph
 * control (inspection, clearing between independent passes, and grouping a training step).</p>
 */
public final class Graph implements AutoCloseable {
    private final NativeContext context;

    public Graph(NativeContext context) {
        this.context = JSLRuntimeException.requireNonNull(context, "context");
    }

    public static Graph of(NativeContext context) {
        return new Graph(context);
    }

    /** Reverse-mode autodiff from a scalar root. */
    public void backward(Tensor root) {
        JSLRuntimeException.requireNonNull(root, "root");
        context.backward(root.handle());
    }

    /** Drops every recorded node; call between independent forward passes. */
    public void clear() {
        context.clearGraph();
    }

    public int size() {
        return context.graphSize();
    }

    /** Returns {@code [liveTensors, liveBytes, graphNodes]}. */
    public long[] stats() {
        return context.stats();
    }

    @Override
    public void close() {
        clear();
    }
}
