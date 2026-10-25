package jsl.nativebridge;

import jsl.JSLRuntimeException;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Packs a batch of tensor operations into one direct {@link ByteBuffer} and executes them with a
 * single JNI call.
 *
 * <p>Wire layout per command (128 bytes, native endianness):</p>
 * <pre>
 * offset 0    u8   op code
 * offset 1    u8   flags
 * offset 2    6B   padding (zero)
 * offset 8    u64  input A
 * offset 16   u64  input B
 * offset 24   u64  input C
 * offset 32   u64  input D
 * offset 40   u64  input E
 * offset 48   u64  output
 * offset 56   i32[16] attributes
 * offset 120  8B   reserved (zero)
 * </pre>
 *
 * <p>Complex ops carry their shape parameters inline in the 16 attribute slots, so convolution
 * and batchnorm need no second allocation or pointer chase.</p>
 */
public final class CommandQueueBuilder implements AutoCloseable {
    public static final int OP_MATMUL = 1;
    public static final int OP_ADD = 2;
    public static final int OP_RELU = 3;
    public static final int OP_SUB = 4;
    public static final int OP_MUL = 5;
    public static final int OP_SIGMOID = 6;
    public static final int OP_TANH = 7;
    public static final int OP_MSE = 8;
    public static final int OP_SOFTMAX = 9;
    public static final int OP_CROSS_ENTROPY = 10;
    public static final int OP_GELU = 11;
    public static final int OP_SILU = 12;
    public static final int OP_LAYER_NORM = 13;
    public static final int OP_BATCH_NORM = 14;
    public static final int OP_CONV2D = 15;
    public static final int OP_MAX_POOL2D = 16;
    public static final int OP_DROPOUT = 17;
    public static final int OP_EMBEDDING = 18;
    public static final int OP_BATCHED_MATMUL = 19;
    public static final int OP_PERMUTE = 20;
    public static final int OP_RESHAPE = 21;
    public static final int OP_GROUP_NORM = 22;

    public static final int COMMAND_BYTES = 128;

    private final NativeContext context;
    private ByteBuffer buffer;
    private int count;

    public CommandQueueBuilder(NativeContext context) {
        this.context = JSLRuntimeException.requireNonNull(context, "context");
        this.buffer = ByteBuffer.allocateDirect(COMMAND_BYTES * 8)
                .order(ByteOrder.nativeOrder());
    }

    public synchronized CommandQueueBuilder matMul(NativeTensorHandle left, NativeTensorHandle right,
                                      NativeTensorHandle output) {
        return binary(OP_MATMUL, left, right, output);
    }

    public synchronized CommandQueueBuilder add(NativeTensorHandle left, NativeTensorHandle right,
                                   NativeTensorHandle output) {
        return binary(OP_ADD, left, right, output);
    }

    public synchronized CommandQueueBuilder sub(NativeTensorHandle left, NativeTensorHandle right,
                                   NativeTensorHandle output) {
        return binary(OP_SUB, left, right, output);
    }

    public synchronized CommandQueueBuilder mul(NativeTensorHandle left, NativeTensorHandle right,
                                   NativeTensorHandle output) {
        return binary(OP_MUL, left, right, output);
    }

    public synchronized CommandQueueBuilder relu(NativeTensorHandle input, NativeTensorHandle output) {
        return unary(OP_RELU, input, output);
    }

    public synchronized CommandQueueBuilder sigmoid(NativeTensorHandle input, NativeTensorHandle output) {
        return unary(OP_SIGMOID, input, output);
    }

    public synchronized CommandQueueBuilder tanh(NativeTensorHandle input, NativeTensorHandle output) {
        return unary(OP_TANH, input, output);
    }

    public synchronized CommandQueueBuilder gelu(NativeTensorHandle input, NativeTensorHandle output) {
        return unary(OP_GELU, input, output);
    }

    public synchronized CommandQueueBuilder silu(NativeTensorHandle input, NativeTensorHandle output) {
        return unary(OP_SILU, input, output);
    }

    public synchronized CommandQueueBuilder mse(NativeTensorHandle prediction, NativeTensorHandle target,
                                   NativeTensorHandle output) {
        return binary(OP_MSE, prediction, target, output);
    }

    public synchronized CommandQueueBuilder softmax(NativeTensorHandle input, NativeTensorHandle output,
                                       int dimension) {
        JSLRuntimeException.requireNonNull(input, "input");
        JSLRuntimeException.requireNonNull(output, "output");
        requireSameContext(input, output);
        begin(OP_SOFTMAX, input.ptr(), 0, 0, 0, 0, output.ptr());
        putInt(0, dimension);
        return this;
    }

    public synchronized CommandQueueBuilder crossEntropy(NativeTensorHandle logits, NativeTensorHandle targets,
                                            NativeTensorHandle output) {
        return binary(OP_CROSS_ENTROPY, logits, targets, output);
    }

    public synchronized CommandQueueBuilder layerNorm(NativeTensorHandle input, NativeTensorHandle gamma,
                                         NativeTensorHandle beta, NativeTensorHandle output,
                                         float epsilon) {
        JSLRuntimeException.requireNonNull(input, "input");
        JSLRuntimeException.requireNonNull(gamma, "gamma");
        JSLRuntimeException.requireNonNull(beta, "beta");
        JSLRuntimeException.requireNonNull(output, "output");
        requireSameContext(input, gamma, beta, output);
        begin(OP_LAYER_NORM, input.ptr(), gamma.ptr(), beta.ptr(), 0, 0, output.ptr());
        putFloat(0, epsilon);
        return this;
    }

    public synchronized CommandQueueBuilder batchNorm2d(NativeTensorHandle input, NativeTensorHandle gamma,
                                           NativeTensorHandle beta,
                                           NativeTensorHandle runningMean,
                                           NativeTensorHandle runningVar,
                                           NativeTensorHandle output, float epsilon,
                                           float momentum, boolean training) {
        JSLRuntimeException.requireNonNull(input, "input");
        JSLRuntimeException.requireNonNull(gamma, "gamma");
        JSLRuntimeException.requireNonNull(beta, "beta");
        JSLRuntimeException.requireNonNull(runningMean, "runningMean");
        JSLRuntimeException.requireNonNull(runningVar, "runningVar");
        JSLRuntimeException.requireNonNull(output, "output");
        requireSameContext(input, gamma, beta, runningMean, runningVar, output);
        begin(OP_BATCH_NORM, input.ptr(), gamma.ptr(), beta.ptr(), runningMean.ptr(),
                runningVar.ptr(), output.ptr());
        putFloat(0, epsilon);
        putFloat(1, momentum);
        putInt(2, training ? 1 : 0);
        return this;
    }

    public synchronized CommandQueueBuilder conv2d(
            NativeTensorHandle input, NativeTensorHandle weight,
            NativeTensorHandle output, int kernelHeight,
            int kernelWidth, int strideHeight, int strideWidth,
            int paddingHeight, int paddingWidth) {
        return conv2d(input, weight, output, kernelHeight, kernelWidth, strideHeight, strideWidth,
                paddingHeight, paddingWidth, 1, 1, 1);
    }

    public synchronized CommandQueueBuilder conv2d(
            NativeTensorHandle input, NativeTensorHandle weight,
            NativeTensorHandle output, int kernelHeight,
            int kernelWidth, int strideHeight, int strideWidth,
            int paddingHeight, int paddingWidth, int dilationHeight,
            int dilationWidth, int groups) {
        JSLRuntimeException.requireNonNull(input, "input");
        JSLRuntimeException.requireNonNull(weight, "weight");
        JSLRuntimeException.requireNonNull(output, "output");
        requireSameContext(input, weight, output);
        begin(OP_CONV2D, input.ptr(), weight.ptr(), 0, 0, 0, output.ptr());
        putInt(0, kernelHeight);
        putInt(1, kernelWidth);
        putInt(2, strideHeight);
        putInt(3, strideWidth);
        putInt(4, paddingHeight);
        putInt(5, paddingWidth);
        putInt(6, dilationHeight);
        putInt(7, dilationWidth);
        putInt(8, groups);
        return this;
    }

    public synchronized CommandQueueBuilder groupNorm(
            NativeTensorHandle input, NativeTensorHandle gamma,
            NativeTensorHandle beta, NativeTensorHandle output,
            int groups, float epsilon) {
        JSLRuntimeException.requireNonNull(input, "input");
        JSLRuntimeException.requireNonNull(gamma, "gamma");
        JSLRuntimeException.requireNonNull(beta, "beta");
        JSLRuntimeException.requireNonNull(output, "output");
        requireSameContext(input, gamma, beta, output);
        begin(OP_GROUP_NORM, input.ptr(), gamma.ptr(), beta.ptr(), 0, 0, output.ptr());
        putInt(0, groups);
        putFloat(1, epsilon);
        return this;
    }

    public synchronized CommandQueueBuilder maxPool2d(NativeTensorHandle input, NativeTensorHandle output,
                                         int kernelHeight, int kernelWidth, int strideHeight,
                                         int strideWidth) {
        JSLRuntimeException.requireNonNull(input, "input");
        JSLRuntimeException.requireNonNull(output, "output");
        requireSameContext(input, output);
        begin(OP_MAX_POOL2D, input.ptr(), 0, 0, 0, 0, output.ptr());
        putInt(0, kernelHeight);
        putInt(1, kernelWidth);
        putInt(2, strideHeight);
        putInt(3, strideWidth);
        return this;
    }

    public synchronized CommandQueueBuilder dropout(NativeTensorHandle input, NativeTensorHandle output,
                                       float probability, boolean training, long seed) {
        JSLRuntimeException.requireNonNull(input, "input");
        JSLRuntimeException.requireNonNull(output, "output");
        requireSameContext(input, output);
        begin(OP_DROPOUT, input.ptr(), 0, 0, 0, 0, output.ptr());
        putFloat(0, probability);
        putInt(1, training ? 1 : 0);
        putInt(2, (int) (seed & 0xffff_ffffL));
        putInt(3, (int) ((seed >>> 32) & 0xffff_ffffL));
        return this;
    }

    public synchronized CommandQueueBuilder embedding(NativeTensorHandle indices, NativeTensorHandle weight,
                                         NativeTensorHandle output) {
        return binary(OP_EMBEDDING, indices, weight, output);
    }

    public synchronized CommandQueueBuilder batchedMatMul(NativeTensorHandle left, NativeTensorHandle right,
                                             NativeTensorHandle output) {
        return binary(OP_BATCHED_MATMUL, left, right, output);
    }

    public synchronized CommandQueueBuilder permute(NativeTensorHandle input, NativeTensorHandle output,
                                       int... permutation) {
        JSLRuntimeException.requireNonNull(input, "input");
        JSLRuntimeException.requireNonNull(output, "output");
        requireSameContext(input, output);
        if (permutation.length < 1 || permutation.length > 4) {
            throw new IllegalArgumentException("permutation rank must be in [1, 4]");
        }
        begin(OP_PERMUTE, input.ptr(), 0, 0, 0, 0, output.ptr());
        putInt(0, permutation.length);
        for (int index = 0; index < permutation.length; index++) {
            putInt(1 + index, permutation[index]);
        }
        return this;
    }

    public synchronized CommandQueueBuilder reshape(NativeTensorHandle input, NativeTensorHandle output,
                                       int... shape) {
        JSLRuntimeException.requireNonNull(input, "input");
        JSLRuntimeException.requireNonNull(output, "output");
        requireSameContext(input, output);
        if (shape.length < 1 || shape.length > 4) {
            throw new IllegalArgumentException("reshape rank must be in [1, 4]");
        }
        begin(OP_RESHAPE, input.ptr(), 0, 0, 0, 0, output.ptr());
        putInt(0, shape.length);
        for (int index = 0; index < shape.length; index++) {
            putInt(1 + index, shape[index]);
        }
        return this;
    }

    private CommandQueueBuilder binary(int op, NativeTensorHandle left, NativeTensorHandle right,
                                       NativeTensorHandle output) {
        JSLRuntimeException.requireNonNull(left, "left");
        JSLRuntimeException.requireNonNull(right, "right");
        JSLRuntimeException.requireNonNull(output, "output");
        requireSameContext(left, right, output);
        begin(op, left.ptr(), right.ptr(), 0, 0, 0, output.ptr());
        return this;
    }

    private CommandQueueBuilder unary(int op, NativeTensorHandle input, NativeTensorHandle output) {
        JSLRuntimeException.requireNonNull(input, "input");
        JSLRuntimeException.requireNonNull(output, "output");
        requireSameContext(input, output);
        begin(op, input.ptr(), 0, 0, 0, 0, output.ptr());
        return this;
    }

    private void begin(int op, long inputA, long inputB, long inputC, long inputD, long inputE,
                       long output) {
        ensureCapacity();
        int base = count * COMMAND_BYTES;
        buffer.putLong(base, 0L);
        buffer.put(base, (byte) op);
        buffer.put(base + 1, (byte) 0);
        buffer.putLong(base + 8, inputA);
        buffer.putLong(base + 16, inputB);
        buffer.putLong(base + 24, inputC);
        buffer.putLong(base + 32, inputD);
        buffer.putLong(base + 40, inputE);
        buffer.putLong(base + 48, output);
        for (int index = 0; index < 16; index++) {
            buffer.putInt(base + 56 + index * 4, 0);
        }
        buffer.putLong(base + 120, 0L);
        count++;
    }

    private void putInt(int attribute, int value) {
        buffer.putInt((count - 1) * COMMAND_BYTES + 56 + attribute * 4, value);
    }

    private void putFloat(int attribute, float value) {
        putInt(attribute, Float.floatToRawIntBits(value));
    }

    private void requireSameContext(NativeTensorHandle... tensors) {
        for (NativeTensorHandle tensor : tensors) {
            if (tensor.context() != context) {
                throw new IllegalArgumentException(
                        "all tensors in a command queue must belong to the same context");
            }
        }
    }

    private void ensureCapacity() {
        int required = (count + 1) * COMMAND_BYTES;
        if (required <= buffer.capacity()) {
            return;
        }
        int capacity = Math.max(buffer.capacity() * 2, required);
        ByteBuffer grown = ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
        ByteBuffer source = buffer.duplicate();
        source.clear();
        source.limit(count * COMMAND_BYTES);
        grown.position(0);
        grown.put(source);
        buffer = grown;
    }

    public synchronized int size() {
        return count;
    }

    public synchronized int capacity() {
        return buffer.capacity() / COMMAND_BYTES;
    }

    public synchronized void execute() {
        execute(context.threads());
    }

    public synchronized void execute(int threads) {
        if (threads < 0 || threads > 256) {
            throw new IllegalArgumentException("threads must be in [0, 256]");
        }
        if (count == 0) {
            return;
        }
        long start = System.nanoTime();
        try (NativeContext.Guard guard = context.guard()) {
            NativeBindings.executeQueue(guard.handle(), buffer, count, threads);
        } finally {
            long elapsed = System.nanoTime() - start;
            jsl.torch.NativeQueueEvent event = new jsl.torch.NativeQueueEvent();
            event.commands = count;
            event.threads = threads;
            event.bytes = count * COMMAND_BYTES;
            event.durationNanos = elapsed;
            event.commit();
            jsl.JslMetrics.recordOperator("native.command_queue", elapsed, count,
                    (long) count * COMMAND_BYTES);
        }
    }

    public synchronized void executeAndClear() {
        execute();
        clear();
    }

    public synchronized void clear() {
        count = 0;
        buffer.clear();
    }

    public synchronized void close() {
        clear();
    }

    @Override
    public String toString() {
        return "CommandQueueBuilder[commands=" + count + ", capacity=" + capacity() + "]";
    }
}
