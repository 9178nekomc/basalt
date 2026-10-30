package basalt.torch;

import basalt.BasaltRuntimeException;
import basalt.nativebridge.CommandQueueBuilder;
import basalt.nativebridge.NativeContext;
import basalt.nativebridge.NativeTensorHandle;
import java.util.Arrays;

/**
 * Eager tensor with reverse-mode autograd, backed by the Rust tensor engine.
 *
 * <p>This is the eager surface: every operation immediately produces a new tensor and
 * records a graph node. Parameters are tensors created with {@link #parameter}; calling
 * {@link #backward()} on a scalar loss accumulates gradients into them.</p>
 *
 * <p>Java-side layout safety is provided by the typed {@link Module} pipeline; {@code Tensor}
 * itself is dynamically shaped, like {@code torch.Tensor}.</p>
 */
public final class Tensor implements AutoCloseable {
    private final NativeContext context;
    private final NativeTensorHandle handle;

    Tensor(NativeContext context, NativeTensorHandle handle) {
        this.context = BasaltRuntimeException.requireNonNull(context, "context");
        this.handle = BasaltRuntimeException.requireNonNull(handle, "handle");
    }

    public static Tensor zeros(NativeContext context, int... shape) {
        return new Tensor(context, context.zeros(shape));
    }

    public static Tensor of(NativeContext context, float[] data, int... shape) {
        return new Tensor(context, context.of(data, false, shape));
    }

    public static Tensor of(NativeContext context, float[] data, boolean requiresGrad,
                            int... shape) {
        return new Tensor(context, context.of(data, requiresGrad, shape));
    }

    /** Trainable zero-initialised tensor. */
    public static Tensor parameter(NativeContext context, int... shape) {
        return new Tensor(context, context.parameter(shape));
    }

    public static Tensor random(NativeContext context, int[] shape, long seed, float scale,
                                boolean requiresGrad) {
        return new Tensor(context, context.random(shape, seed, scale, requiresGrad));
    }

    public static Tensor scalar(NativeContext context, float value) {
        return of(context, new float[]{value}, 1);
    }

    public NativeContext context() {
        return context;
    }

    public NativeTensorHandle handle() {
        return handle;
    }

    public int[] shape() {
        return handle.shape();
    }

    public int rank() {
        return shape().length;
    }

    public int size() {
        return numel(shape());
    }

    public boolean requiresGrad() {
        return handle.requiresGrad();
    }

    public float[] data() {
        return handle.data();
    }

    public float[] grad() {
        return handle.grad();
    }

    public void setData(float[] values) {
        handle.setData(values);
    }

    public void fill(float value) {
        handle.fill(value);
    }

    public void randomize(long seed, float scale) {
        handle.randomize(seed, scale);
    }

    public void zeroGrad() {
        handle.zeroGrad();
    }

    public void backward() {
        context.backward(handle);
    }

    public void sgd(float learningRate, float weightDecay) {
        handle.sgd(learningRate, weightDecay);
    }

    public void adam(float learningRate, float beta1, float beta2, float epsilon,
                     float weightDecay, long step) {
        handle.adam(learningRate, beta1, beta2, epsilon, weightDecay, step);
    }

    public double norm() {
        return handle.norm();
    }

    public double gradNorm() {
        return handle.gradNorm();
    }

    public boolean isFinite() {
        return handle.isFinite();
    }

    public Tensor add(Tensor other) {
        int[] outputShape = broadcastShape(shape(), other.shape());
        Tensor output = zeros(context, outputShape);
        CommandQueueBuilder queue = new CommandQueueBuilder(context);
        queue.add(handle, other.handle, output.handle);
        queue.executeAndClear();
        return output;
    }

    public Tensor sub(Tensor other) {
        int[] outputShape = broadcastShape(shape(), other.shape());
        Tensor output = zeros(context, outputShape);
        CommandQueueBuilder queue = new CommandQueueBuilder(context);
        queue.sub(handle, other.handle, output.handle);
        queue.executeAndClear();
        return output;
    }

    public Tensor mul(Tensor other) {
        int[] outputShape = broadcastShape(shape(), other.shape());
        Tensor output = zeros(context, outputShape);
        CommandQueueBuilder queue = new CommandQueueBuilder(context);
        queue.mul(handle, other.handle, output.handle);
        queue.executeAndClear();
        return output;
    }

    public Tensor matMul(Tensor other) {
        int[] left = shape();
        int[] right = other.shape();
        if (left.length != 2 || right.length != 2 || left[1] != right[0]) {
            throw new IllegalArgumentException(
                    "matMul expects [M,K] x [K,N], received " + Arrays.toString(left)
                            + " x " + Arrays.toString(right));
        }
        Tensor output = zeros(context, left[0], right[1]);
        CommandQueueBuilder queue = new CommandQueueBuilder(context);
        queue.matMul(handle, other.handle, output.handle);
        queue.executeAndClear();
        return output;
    }

    public Tensor batchedMatMul(Tensor other) {
        int[] left = shape();
        int[] right = other.shape();
        if (left.length != 3 || right.length != 3 || left[0] != right[0] || left[2] != right[1]) {
            throw new IllegalArgumentException(
                    "batchedMatMul expects [B,M,K] x [B,K,N], received "
                            + Arrays.toString(left) + " x " + Arrays.toString(right));
        }
        Tensor output = zeros(context, left[0], left[1], right[2]);
        CommandQueueBuilder queue = new CommandQueueBuilder(context);
        queue.batchedMatMul(handle, other.handle, output.handle);
        queue.executeAndClear();
        return output;
    }

    public Tensor relu() {
        return unary(CommandQueueBuilder.OP_RELU);
    }

    public Tensor sigmoid() {
        return unary(CommandQueueBuilder.OP_SIGMOID);
    }

    public Tensor tanh() {
        return unary(CommandQueueBuilder.OP_TANH);
    }

    public Tensor gelu() {
        return unary(CommandQueueBuilder.OP_GELU);
    }

    public Tensor silu() {
        return unary(CommandQueueBuilder.OP_SILU);
    }

    private Tensor unary(int op) {
        Tensor output = zeros(context, shape());
        CommandQueueBuilder queue = new CommandQueueBuilder(context);
        switch (op) {
            case CommandQueueBuilder.OP_RELU -> queue.relu(handle, output.handle);
            case CommandQueueBuilder.OP_SIGMOID -> queue.sigmoid(handle, output.handle);
            case CommandQueueBuilder.OP_TANH -> queue.tanh(handle, output.handle);
            case CommandQueueBuilder.OP_GELU -> queue.gelu(handle, output.handle);
            case CommandQueueBuilder.OP_SILU -> queue.silu(handle, output.handle);
            default -> throw new IllegalStateException("unknown unary op " + op);
        }
        queue.executeAndClear();
        return output;
    }

    public Tensor softmax(int dimension) {
        int[] shape = shape();
        if (dimension < 0) {
            dimension += shape.length;
        }
        if (dimension != shape.length - 1) {
            throw new IllegalArgumentException("softmax currently supports the last dimension");
        }
        Tensor output = zeros(context, shape);
        CommandQueueBuilder queue = new CommandQueueBuilder(context);
        queue.softmax(handle, output.handle, shape[dimension]);
        queue.executeAndClear();
        return output;
    }

    public Tensor crossEntropy(Tensor targets) {
        int[] logits = shape();
        if (logits.length != 2 || targets.size() != logits[0]) {
            throw new IllegalArgumentException("crossEntropy expects [N,C] logits and N targets");
        }
        Tensor output = zeros(context, 1);
        CommandQueueBuilder queue = new CommandQueueBuilder(context);
        queue.crossEntropy(handle, targets.handle, output.handle);
        queue.executeAndClear();
        return output;
    }

    public Tensor mse(Tensor target) {
        if (!Arrays.equals(shape(), target.shape())) {
            throw new IllegalArgumentException("MSE expects equal shapes");
        }
        Tensor output = zeros(context, 1);
        CommandQueueBuilder queue = new CommandQueueBuilder(context);
        queue.mse(handle, target.handle, output.handle);
        queue.executeAndClear();
        return output;
    }

    public Tensor layerNorm(Tensor gamma, Tensor beta, float epsilon) {
        int[] shape = shape();
        int dimension = shape[shape.length - 1];
        if (gamma.size() != dimension || beta.size() != dimension) {
            throw new IllegalArgumentException("layerNorm gamma/beta must match the last dimension");
        }
        Tensor output = zeros(context, shape);
        CommandQueueBuilder queue = new CommandQueueBuilder(context);
        queue.layerNorm(handle, gamma.handle, beta.handle, output.handle, epsilon);
        queue.executeAndClear();
        return output;
    }

    public Tensor batchNorm2d(Tensor gamma, Tensor beta, Tensor runningMean, Tensor runningVar,
                              float epsilon, float momentum, boolean training) {
        int[] shape = shape();
        if (shape.length != 4) {
            throw new IllegalArgumentException("batchNorm2d expects NCHW input");
        }
        Tensor output = zeros(context, shape);
        CommandQueueBuilder queue = new CommandQueueBuilder(context);
        queue.batchNorm2d(handle, gamma.handle, beta.handle, runningMean.handle,
                runningVar.handle, output.handle, epsilon, momentum, training);
        queue.executeAndClear();
        return output;
    }

    public Tensor conv2d(Tensor weight, int kernelHeight, int kernelWidth, int strideHeight,
                         int strideWidth, int paddingHeight, int paddingWidth) {
        return conv2d(weight, kernelHeight, kernelWidth, strideHeight, strideWidth,
                paddingHeight, paddingWidth, 1, 1, 1);
    }

    public Tensor conv2d(Tensor weight, int kernelHeight, int kernelWidth, int strideHeight,
                         int strideWidth, int paddingHeight, int paddingWidth,
                         int dilationHeight, int dilationWidth, int groups) {
        int[] input = shape();
        int[] kernel = weight.shape();
        if (groups <= 0) {
            throw new IllegalArgumentException("groups must be positive");
        }
        if (input.length != 4 || kernel.length != 4
                || input[1] % groups != 0 || kernel[0] % groups != 0
                || input[1] / groups != kernel[1]) {
            throw new IllegalArgumentException("conv2d expects NCHW input and [O, I/groups, kH, kW] weight");
        }
        int effKh = dilationHeight * (kernelHeight - 1) + 1;
        int effKw = dilationWidth * (kernelWidth - 1) + 1;
        int outputHeight = (input[2] + 2 * paddingHeight - effKh) / strideHeight + 1;
        int outputWidth = (input[3] + 2 * paddingWidth - effKw) / strideWidth + 1;
        if (outputHeight <= 0 || outputWidth <= 0) {
            throw new IllegalArgumentException("conv2d output would be empty");
        }
        Tensor output = zeros(context, input[0], kernel[0], outputHeight, outputWidth);
        CommandQueueBuilder queue = new CommandQueueBuilder(context);
        queue.conv2d(handle, weight.handle, output.handle, kernelHeight, kernelWidth,
                strideHeight, strideWidth, paddingHeight, paddingWidth,
                dilationHeight, dilationWidth, groups);
        queue.executeAndClear();
        return output;
    }

    public Tensor groupNorm(Tensor gamma, Tensor beta, int groups, float epsilon) {
        int[] shape = shape();
        if (shape.length < 2 || shape[1] % groups != 0) {
            throw new IllegalArgumentException("groupNorm expects [N, C, ...] with C divisible by groups");
        }
        if (gamma.size() != shape[1] || beta.size() != shape[1]) {
            throw new IllegalArgumentException("groupNorm gamma/beta must match channel dimension " + shape[1]);
        }
        Tensor output = zeros(context, shape);
        CommandQueueBuilder queue = new CommandQueueBuilder(context);
        queue.groupNorm(handle, gamma.handle, beta.handle, output.handle, groups, epsilon);
        queue.executeAndClear();
        return output;
    }

    public Tensor leakyRelu(float negativeSlope) {
        Tensor pos = relu();
        Tensor negInput = mul(scalar(context, -1.0f)).relu().mul(scalar(context, -negativeSlope));
        return pos.add(negInput);
    }

    public Tensor elu(float alpha) {
        // Differentiable approximation via composition or exact elementwise for forward + backward
        Tensor pos = relu();
        Tensor neg = sub(pos); // min(0, x)
        // Use 2nd-order Padé / exact exp for negative branch or silu/tanh composition
        // Even better: for x <= 0, neg is non-positive; we can compute exact forward and use autograd!
        // Note: in our autodiff graph, we can express L1/SmoothL1/LeakyReLU/Softplus directly via graph ops
        // or exact differentiable masks.
        float[] x = data();
        float[] mask = new float[x.length];
        float[] offset = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            if (x[i] > 0.0f) {
                mask[i] = 1.0f;
                offset[i] = 0.0f;
            } else {
                float ex = (float) Math.exp(x[i]);
                mask[i] = alpha * ex;
                offset[i] = alpha * (ex - 1.0f) - mask[i] * x[i];
            }
        }
        return mul(of(context, mask, shape())).add(of(context, offset, shape()));
    }

    public Tensor l1Loss(Tensor target) {
        if (!Arrays.equals(shape(), target.shape())) {
            throw new IllegalArgumentException("L1Loss expects equal shapes");
        }
        Tensor diff = sub(target);
        Tensor absDiff = diff.relu().add(diff.mul(scalar(context, -1.0f)).relu());
        float invN = 1.0f / size();
        Tensor ones = of(context, fillArray(size(), invN), size(), 1);
        return absDiff.reshape(1, size()).matMul(ones).reshape(1);
    }

    public Tensor smoothL1Loss(Tensor target, float beta) {
        if (!Arrays.equals(shape(), target.shape())) {
            throw new IllegalArgumentException("SmoothL1Loss expects equal shapes");
        }
        if (beta <= 0.0f) {
            return l1Loss(target);
        }
        Tensor diff = sub(target);
        float[] d = diff.data();
        int n = d.length;
        float[] quadMask = new float[n];
        float[] linSign = new float[n];
        float[] linOffset = new float[n];
        for (int i = 0; i < n; i++) {
            float abs = Math.abs(d[i]);
            if (abs < beta) {
                quadMask[i] = 1.0f / (float) Math.sqrt(2.0f * beta);
            } else {
                linSign[i] = d[i] >= 0.0f ? 1.0f : -1.0f;
                linOffset[i] = -0.5f * beta;
            }
        }
        Tensor quadPart = diff.mul(of(context, quadMask, shape()));
        Tensor quadSq = quadPart.mul(quadPart);
        Tensor linPart = diff.mul(of(context, linSign, shape())).add(of(context, linOffset, shape()));
        Tensor elem = quadSq.add(linPart);
        Tensor ones = of(context, fillArray(n, 1.0f / n), n, 1);
        return elem.reshape(1, n).matMul(ones).reshape(1);
    }

    public Tensor bceWithLogitsLoss(Tensor target) {
        if (!Arrays.equals(shape(), target.shape())) {
            throw new IllegalArgumentException("BCEWithLogitsLoss expects equal shapes");
        }
        Tensor probs = sigmoid();
        // Numerically stable BCE with exact gradient (sigmoid(x) - y) / N
        float[] x = data();
        float[] y = target.data();
        float[] p = probs.data();
        int n = x.length;
        double sumLoss = 0.0;
        float[] gradScale = new float[n];
        for (int i = 0; i < n; i++) {
            double max0 = Math.max(x[i], 0.0);
            sumLoss += max0 - x[i] * y[i] + Math.log1p(Math.exp(-Math.abs(x[i])));
            gradScale[i] = (p[i] - y[i]) / n;
        }
        float meanLoss = (float) (sumLoss / n);
        // Attach exact first-order gradient to this logit tensor via linear surrogate + constant offset
        Tensor linearTerm = reshape(1, n).matMul(of(context, gradScale, n, 1)).reshape(1);
        float linearVal = linearTerm.data()[0];
        return linearTerm.add(scalar(context, meanLoss - linearVal));
    }

    private static float[] fillArray(int length, float value) {
        float[] arr = new float[length];
        Arrays.fill(arr, value);
        return arr;
    }

    public Tensor maxPool2d(int kernelHeight, int kernelWidth, int strideHeight,
                            int strideWidth) {
        int[] input = shape();
        if (input.length != 4) {
            throw new IllegalArgumentException("maxPool2d expects NCHW input");
        }
        int outputHeight = (input[2] - kernelHeight) / strideHeight + 1;
        int outputWidth = (input[3] - kernelWidth) / strideWidth + 1;
        if (outputHeight <= 0 || outputWidth <= 0) {
            throw new IllegalArgumentException("maxPool2d output would be empty");
        }
        Tensor output = zeros(context, input[0], input[1], outputHeight, outputWidth);
        CommandQueueBuilder queue = new CommandQueueBuilder(context);
        queue.maxPool2d(handle, output.handle, kernelHeight, kernelWidth, strideHeight,
                strideWidth);
        queue.executeAndClear();
        return output;
    }

    public Tensor dropout(float probability, boolean training, long seed) {
        Tensor output = zeros(context, shape());
        CommandQueueBuilder queue = new CommandQueueBuilder(context);
        queue.dropout(handle, output.handle, probability, training, seed);
        queue.executeAndClear();
        return output;
    }

    public Tensor embedding(Tensor weight) {
        int[] weightShape = weight.shape();
        if (weightShape.length != 2) {
            throw new IllegalArgumentException("embedding weight must be rank 2");
        }
        int[] outputShape = Arrays.copyOf(shape(), rank() + 1);
        outputShape[rank()] = weightShape[1];
        Tensor output = zeros(context, outputShape);
        CommandQueueBuilder queue = new CommandQueueBuilder(context);
        queue.embedding(handle, weight.handle, output.handle);
        queue.executeAndClear();
        return output;
    }

    public Tensor permute(int... permutation) {
        int[] input = shape();
        if (permutation.length != input.length) {
            throw new IllegalArgumentException("permutation rank must match the tensor rank");
        }
        int[] outputShape = new int[input.length];
        for (int index = 0; index < input.length; index++) {
            if (permutation[index] < 0 || permutation[index] >= input.length) {
                throw new IllegalArgumentException("permutation index out of range");
            }
            outputShape[index] = input[permutation[index]];
        }
        Tensor output = zeros(context, outputShape);
        CommandQueueBuilder queue = new CommandQueueBuilder(context);
        queue.permute(handle, output.handle, permutation);
        queue.executeAndClear();
        return output;
    }

    public Tensor reshape(int... newShape) {
        if (numel(newShape) != size()) {
            throw new IllegalArgumentException(
                    "reshape element count mismatch: " + size() + " vs " + numel(newShape));
        }
        Tensor output = zeros(context, newShape);
        CommandQueueBuilder queue = new CommandQueueBuilder(context);
        queue.reshape(handle, output.handle, newShape);
        queue.executeAndClear();
        return output;
    }

    public Tensor detachCopy() {
        Tensor copy = zeros(context, shape());
        copy.handle.setData(data());
        return copy;
    }

    @Override
    public void close() {
        handle.close();
    }

    @Override
    public String toString() {
        return "Tensor" + Arrays.toString(shape()) + "[requiresGrad=" + requiresGrad() + "]";
    }

    static int numel(int[] shape) {
        if (shape.length == 0) {
            throw new IllegalArgumentException("shape must not be empty");
        }
        int elements = 1;
        for (int dimension : shape) {
            if (dimension <= 0) {
                throw new IllegalArgumentException("shape dimensions must be positive");
            }
            elements = Math.multiplyExact(elements, dimension);
        }
        return elements;
    }

    static int[] broadcastShape(int[] left, int[] right) {
        int rank = Math.max(left.length, right.length);
        int[] output = new int[rank];
        for (int index = 0; index < rank; index++) {
            int leftDim = index < rank - left.length ? 1 : left[index - (rank - left.length)];
            int rightDim = index < rank - right.length ? 1 : right[index - (rank - right.length)];
            if (leftDim == rightDim) {
                output[index] = leftDim;
            } else if (leftDim == 1) {
                output[index] = rightDim;
            } else if (rightDim == 1) {
                output[index] = leftDim;
            } else {
                throw new IllegalArgumentException(
                        "cannot broadcast " + Arrays.toString(left) + " and "
                                + Arrays.toString(right));
            }
        }
        return output;
    }
}
