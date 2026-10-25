package basalt.nativebridge;

import basalt.BasaltRuntimeException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * A compact multilayer perceptron built entirely from the public tensor/queue API.
 *
 * <p>This class is the "expert tier" reference implementation: it shows how to compose matmul,
 * broadcast add, activation and MSE nodes, run one backward pass, and update parameters with
 * Adam. It is intentionally not a high-level estimator; the classical {@code basalt.ml} models keep
 * their own tuned implementations.</p>
 *
 * <p>Mini-batches are padded by wrapping to the start of the shuffled epoch when the last batch is
 * short. That keeps every native tensor shape static (no reallocation per step) at the cost of a
 * small duplicate-sample bias in the final batch.</p>
 */
public final class NativeMLP implements AutoCloseable {
    public enum Activation {
        RELU, TANH, SIGMOID
    }

    public enum OutputActivation {
        NONE, SIGMOID, TANH
    }

    private final NativeContext context;
    private final int[] layerSizes;
    private final Activation activation;
    private final OutputActivation outputActivation;
    private final long seed;
    private final List<NativeTensorHandle> weights = new ArrayList<>();
    private final List<NativeTensorHandle> biases = new ArrayList<>();
    private final List<NativeTensorHandle> parameters = new ArrayList<>();
    private final CommandQueueBuilder queue;
    private boolean closed;

    public NativeMLP(NativeContext context, int[] layerSizes, Activation activation,
                     OutputActivation outputActivation, long seed) {
        this.context = BasaltRuntimeException.requireNonNull(context, "context");
        BasaltRuntimeException.requireNonNull(layerSizes, "layerSizes");
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("layerSizes must contain input and output sizes");
        }
        this.layerSizes = layerSizes.clone();
        for (int size : this.layerSizes) {
            if (size <= 0) {
                throw new IllegalArgumentException("layer sizes must be positive");
            }
        }
        this.activation = BasaltRuntimeException.requireNonNull(activation, "activation");
        this.outputActivation = BasaltRuntimeException.requireNonNull(outputActivation, "outputActivation");
        this.seed = seed;
        this.queue = new CommandQueueBuilder(context);
        for (int layer = 0; layer + 1 < this.layerSizes.length; layer++) {
            int input = this.layerSizes[layer];
            int output = this.layerSizes[layer + 1];
            float scale = (float) Math.sqrt(6.0 / (input + output));
            NativeTensorHandle weight = context.random(
                    new int[]{input, output}, seed + 1_009L * (layer + 1), scale, true);
            NativeTensorHandle bias = context.parameter(output);
            weights.add(weight);
            biases.add(bias);
            parameters.add(weight);
            parameters.add(bias);
        }
    }

    public int[] layerSizes() {
        return layerSizes.clone();
    }

    public List<NativeTensorHandle> parameters() {
        return Collections.unmodifiableList(parameters);
    }

    public NativeTensorHandle weight(int layer) {
        return weights.get(layer);
    }

    public NativeTensorHandle bias(int layer) {
        return biases.get(layer);
    }

    /**
     * Trains with Adam and returns the mean MSE of the last epoch.
     *
     * @param features rows of input features
     * @param targets  rows of target values
     * @param epochs   full passes over the data
     * @param batchSize static native batch size; the final short batch wraps to the epoch start
     * @param learningRate Adam learning rate
     * @param weightDecay decoupled weight decay
     */
    public float fit(float[][] features, float[][] targets, int epochs, int batchSize,
                     float learningRate, float weightDecay) {
        requireOpen();
        validateDataset(features, targets);
        if (epochs <= 0) {
            throw new IllegalArgumentException("epochs must be positive");
        }
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive");
        }
        int rows = features.length;
        int inputSize = layerSizes[0];
        int outputSize = layerSizes[layerSizes.length - 1];
        int effectiveBatch = Math.min(batchSize, rows);

        List<NativeTensorHandle> batchTensors = new ArrayList<>();
        try {
            NativeTensorHandle input = context.zeros(effectiveBatch, inputSize);
            NativeTensorHandle target = context.zeros(effectiveBatch, outputSize);
            NativeTensorHandle loss = context.zeros(1);
            batchTensors.add(input);
            batchTensors.add(target);
            batchTensors.add(loss);
            List<List<NativeTensorHandle>> layers = allocateLayerTensors(effectiveBatch,
                    batchTensors);

            float[] inputBuffer = new float[effectiveBatch * inputSize];
            float[] targetBuffer = new float[effectiveBatch * outputSize];
            int[] order = new int[rows];
            for (int index = 0; index < rows; index++) {
                order[index] = index;
            }
            Random random = new Random(seed ^ 0x9E3779B97F4A7C15L);
            long step = 0L;
            float epochLoss = 0.0f;
            context.clearGraph();
            for (int epoch = 0; epoch < epochs; epoch++) {
                shuffle(order, random);
                double lossSum = 0.0;
                int batches = 0;
                for (int start = 0; start < rows; start += effectiveBatch) {
                    fillBatch(features, targets, order, start, effectiveBatch,
                            inputBuffer, targetBuffer, inputSize, outputSize);
                    input.setData(inputBuffer);
                    target.setData(targetBuffer);
                    NativeTensorHandle logits = forward(input, layers, effectiveBatch);
                    queue.mse(logits, target, loss).execute();
                    queue.clear();
                    float lossValue = loss.data()[0];
                    if (!Float.isFinite(lossValue)) {
                        throw new IllegalStateException(
                                "training diverged: non-finite loss at step " + step);
                    }
                    lossSum += lossValue;
                    batches++;
                    context.backward(loss);
                    step++;
                    for (NativeTensorHandle parameter : parameters) {
                        parameter.adam(learningRate, 0.9f, 0.999f, 1.0e-8f, weightDecay, step);
                        parameter.zeroGrad();
                    }
                    context.clearGraph();
                }
                epochLoss = (float) (lossSum / Math.max(1, batches));
            }
            return epochLoss;
        } finally {
            for (NativeTensorHandle tensor : batchTensors) {
                tensor.close();
            }
            context.clearGraph();
        }
    }

    public float[] predict(float[][] features) {
        requireOpen();
        BasaltRuntimeException.requireNonNull(features, "features");
        if (features.length == 0) {
            return new float[0];
        }
        int inputSize = layerSizes[0];
        int outputSize = layerSizes[layerSizes.length - 1];
        for (float[] row : features) {
            if (row == null || row.length != inputSize) {
                throw new IllegalArgumentException("every row must have " + inputSize + " features");
            }
        }
        int rows = features.length;
        List<NativeTensorHandle> tensors = new ArrayList<>();
        try {
            context.clearGraph();
            NativeTensorHandle input = context.zeros(rows, inputSize);
            tensors.add(input);
            float[] buffer = new float[rows * inputSize];
            for (int row = 0; row < rows; row++) {
                System.arraycopy(features[row], 0, buffer, row * inputSize, inputSize);
            }
            input.setData(buffer);
            List<List<NativeTensorHandle>> layers = allocateLayerTensors(rows, tensors);
            NativeTensorHandle output = forward(input, layers, rows);
            float[] flat = output.data();
            float[] result = new float[rows * outputSize];
            System.arraycopy(flat, 0, result, 0, result.length);
            return result;
        } finally {
            for (NativeTensorHandle tensor : tensors) {
                tensor.close();
            }
            context.clearGraph();
        }
    }

    /** Convenience wrapper returning one scalar per row. */
    public float[] predictScalar(float[][] features) {
        int outputSize = layerSizes[layerSizes.length - 1];
        if (outputSize != 1) {
            throw new IllegalStateException("predictScalar requires a single output unit");
        }
        return predict(features);
    }

    public double gradientNorm() {
        double sum = 0.0;
        for (NativeTensorHandle parameter : parameters) {
            double norm = parameter.gradNorm();
            sum += norm * norm;
        }
        return Math.sqrt(sum);
    }

    private List<List<NativeTensorHandle>> allocateLayerTensors(
            int rows, List<NativeTensorHandle> owner) {
        List<List<NativeTensorHandle>> layers = new ArrayList<>();
        for (int layer = 0; layer < weights.size(); layer++) {
            int output = layerSizes[layer + 1];
            NativeTensorHandle product = context.zeros(rows, output);
            NativeTensorHandle biased = context.zeros(rows, output);
            NativeTensorHandle activated = context.zeros(rows, output);
            owner.add(product);
            owner.add(biased);
            owner.add(activated);
            layers.add(Arrays.asList(product, biased, activated));
        }
        return layers;
    }

    private NativeTensorHandle forward(NativeTensorHandle input,
                                       List<List<NativeTensorHandle>> layers, int rows) {
        NativeTensorHandle current = input;
        for (int layer = 0; layer < weights.size(); layer++) {
            List<NativeTensorHandle> tensors = layers.get(layer);
            NativeTensorHandle product = tensors.get(0);
            NativeTensorHandle biased = tensors.get(1);
            NativeTensorHandle activated = tensors.get(2);
            queue.matMul(current, weights.get(layer), product);
            queue.add(product, biases.get(layer), biased);
            boolean last = layer == weights.size() - 1;
            Activation selected = last ? toActivation(outputActivation) : activation;
            if (selected == null) {
                current = biased;
            } else {
                switch (selected) {
                    case RELU -> queue.relu(biased, activated);
                    case TANH -> queue.tanh(biased, activated);
                    case SIGMOID -> queue.sigmoid(biased, activated);
                    default -> throw new IllegalStateException("unhandled activation " + selected);
                }
                current = activated;
            }
        }
        queue.execute();
        queue.clear();
        return current;
    }

    private static Activation toActivation(OutputActivation output) {
        return switch (output) {
            case NONE -> null;
            case SIGMOID -> Activation.SIGMOID;
            case TANH -> Activation.TANH;
        };
    }

    private void validateDataset(float[][] features, float[][] targets) {
        BasaltRuntimeException.requireNonNull(features, "features");
        BasaltRuntimeException.requireNonNull(targets, "targets");
        if (features.length == 0 || features.length != targets.length) {
            throw new IllegalArgumentException(
                    "features and targets must be non-empty and have equal row counts");
        }
        int inputSize = layerSizes[0];
        int outputSize = layerSizes[layerSizes.length - 1];
        for (int row = 0; row < features.length; row++) {
            if (features[row] == null || features[row].length != inputSize) {
                throw new IllegalArgumentException(
                        String.format(Locale.ROOT, "feature row %d must have %d values",
                                row, inputSize));
            }
            if (targets[row] == null || targets[row].length != outputSize) {
                throw new IllegalArgumentException(
                        String.format(Locale.ROOT, "target row %d must have %d values",
                                row, outputSize));
            }
        }
    }

    private static void fillBatch(float[][] features, float[][] targets, int[] order, int start,
                                  int batchSize, float[] inputBuffer, float[] targetBuffer,
                                  int inputSize, int outputSize) {
        int rows = features.length;
        for (int offset = 0; offset < batchSize; offset++) {
            int source = order[(start + offset) % rows];
            System.arraycopy(features[source], 0, inputBuffer, offset * inputSize, inputSize);
            System.arraycopy(targets[source], 0, targetBuffer, offset * outputSize, outputSize);
        }
    }

    private static void shuffle(int[] values, Random random) {
        for (int index = values.length - 1; index > 0; index--) {
            int other = random.nextInt(index + 1);
            int temporary = values[index];
            values[index] = values[other];
            values[other] = temporary;
        }
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException("NativeMLP is closed");
        }
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        for (NativeTensorHandle parameter : parameters) {
            parameter.close();
        }
        queue.clear();
    }

    @Override
    public String toString() {
        return "NativeMLP" + Arrays.toString(layerSizes)
                + "[activation=" + activation + ", output=" + outputActivation + "]";
    }
}
