package basalt.nn;

import basalt.core.BSF;
import basalt.core.ModelFormatException;
import basalt.core.ModelKind;
import basalt.core.ModelReader;
import basalt.core.ModelWriter;
import basalt.core.Persistable;
import basalt.core.Tensor;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Elman RNN: {@code h_t = tanh(W x_t + U h_{t-1} + b)}.
 *
 * <p>This layer exists as the control group for the gradient-vanishing experiment. Its BPTT
 * Jacobian is a product of {@code U^T diag(1 - h^2)} factors, whose spectral norm is typically
 * below one, so the gradient of an early timestep decays geometrically with sequence length.</p>
 */
public final class VanillaRNN implements Layer<BSF, BSF>, Persistable {
    private final int hiddenSize;
    private final long seed;
    private int inputFeatures = -1;
    private Parameter inputWeights;
    private Parameter recurrentWeights;
    private Parameter bias;
    private Tensor<BSF> cachedInput;
    private double[] hidden;
    private double[] inputWeightGradient;
    private double[] recurrentWeightGradient;
    private double[] biasGradient;
    private boolean gradientReady;

    private VanillaRNN(Builder builder) {
        hiddenSize = builder.hiddenSize;
        seed = builder.seed;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Tensor<BSF> forward(Tensor<BSF> input) {
        gradientReady = false;
        int features = input.dimension(2);
        if (inputFeatures < 0) {
            initialize(features);
        } else if (features != inputFeatures) {
            throw new IllegalArgumentException("VanillaRNN input feature count mismatch");
        }
        cachedInput = input;
        int batch = input.dimension(0);
        int sequence = input.dimension(1);
        int size = batch * sequence * hiddenSize;
        if (hidden == null || hidden.length != size) {
            hidden = new double[size];
        }
        Arrays.fill(hidden, 0.0);
        double[] x = input.data();
        double[] w = inputWeights.values();
        double[] u = recurrentWeights.values();
        double[] b = bias.values();
        for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
            for (int time = 0; time < sequence; time++) {
                int timeBase = (batchIndex * sequence + time) * hiddenSize;
                int previousBase = timeBase - hiddenSize;
                int inputBase = (batchIndex * sequence + time) * features;
                for (int unit = 0; unit < hiddenSize; unit++) {
                    double sum = b[unit];
                    int inputWeightBase = unit * features;
                    for (int feature = 0; feature < features; feature++) {
                        sum += w[inputWeightBase + feature] * x[inputBase + feature];
                    }
                    if (time > 0) {
                        int recurrentWeightBase = unit * hiddenSize;
                        for (int previous = 0; previous < hiddenSize; previous++) {
                            sum += u[recurrentWeightBase + previous] * hidden[previousBase + previous];
                        }
                    }
                    hidden[timeBase + unit] = Math.tanh(sum);
                }
            }
        }
        return Tensor.ofBSF(batch, sequence, hiddenSize, hidden);
    }

    private void initialize(int features) {
        inputFeatures = features;
        double[] w = new double[hiddenSize * features];
        double[] u = new double[hiddenSize * hiddenSize];
        Random random = new Random(seed);
        double inputLimit = Math.sqrt(6.0 / (features + hiddenSize));
        double recurrentLimit = Math.sqrt(6.0 / (2.0 * hiddenSize));
        for (int i = 0; i < w.length; i++) {
            w[i] = (random.nextDouble() * 2.0 - 1.0) * inputLimit;
        }
        for (int i = 0; i < u.length; i++) {
            u[i] = (random.nextDouble() * 2.0 - 1.0) * recurrentLimit;
        }
        inputWeights = new Parameter(w);
        recurrentWeights = new Parameter(u);
        bias = new Parameter(new double[hiddenSize]);
        inputWeightGradient = new double[w.length];
        recurrentWeightGradient = new double[u.length];
        biasGradient = new double[hiddenSize];
    }

    @Override
    public Tensor<BSF> backward(Tensor<BSF> gradient) {
        gradientReady = false;
        if (cachedInput == null) {
            throw new IllegalStateException("VanillaRNN backward requires forward");
        }
        int batch = cachedInput.dimension(0);
        int sequence = cachedInput.dimension(1);
        if (gradient.dimension(0) != batch || gradient.dimension(1) != sequence
                || gradient.dimension(2) != hiddenSize) {
            throw new IllegalArgumentException("VanillaRNN gradient shape mismatch");
        }
        Arrays.fill(inputWeightGradient, 0.0);
        Arrays.fill(recurrentWeightGradient, 0.0);
        Arrays.fill(biasGradient, 0.0);
        Tensor<BSF> inputGradient = Tensor.zerosBSF(batch, sequence, inputFeatures);
        double[] dx = inputGradient.data();
        double[] x = cachedInput.data();
        double[] g = gradient.data();
        double[] w = inputWeights.values();
        double[] u = recurrentWeights.values();
        int batchHidden = batch * hiddenSize;
        double[] nextHidden = new double[batchHidden];
        double[] currentHidden = new double[batchHidden];

        for (int time = sequence - 1; time >= 0; time--) {
            Arrays.fill(currentHidden, 0.0);
            for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
                int timeBase = (batchIndex * sequence + time) * hiddenSize;
                int previousBase = timeBase - hiddenSize;
                int inputBase = (batchIndex * sequence + time) * inputFeatures;
                int recurrentBatchBase = batchIndex * hiddenSize;
                for (int unit = 0; unit < hiddenSize; unit++) {
                    int index = timeBase + unit;
                    double activated = hidden[index];
                    double hiddenGradient = g[index] + nextHidden[recurrentBatchBase + unit];
                    double preGradient = hiddenGradient * (1.0 - activated * activated);
                    biasGradient[unit] += preGradient;
                    int inputWeightBase = unit * inputFeatures;
                    for (int feature = 0; feature < inputFeatures; feature++) {
                        inputWeightGradient[inputWeightBase + feature] +=
                                preGradient * x[inputBase + feature];
                        dx[inputBase + feature] += preGradient * w[inputWeightBase + feature];
                    }
                    if (time > 0) {
                        int recurrentWeightBase = unit * hiddenSize;
                        for (int previous = 0; previous < hiddenSize; previous++) {
                            recurrentWeightGradient[recurrentWeightBase + previous] +=
                                    preGradient * hidden[previousBase + previous];
                            currentHidden[recurrentBatchBase + previous] +=
                                    preGradient * u[recurrentWeightBase + previous];
                        }
                    }
                }
            }
            double[] temporary = nextHidden;
            nextHidden = currentHidden;
            currentHidden = temporary;
        }
        gradientReady = true;
        return inputGradient;
    }

    @Override
    public void update(OptimizerConfig optimizer) {
        if (!gradientReady) {
            throw new IllegalStateException("VanillaRNN update requires forward and backward");
        }
        inputWeights.update(inputWeightGradient, optimizer);
        recurrentWeights.update(recurrentWeightGradient, optimizer);
        bias.update(biasGradient, optimizer);
        gradientReady = false;
    }

    @Override
    public List<ParameterRef> parameters() {
        if (inputWeights == null) {
            return ParameterRef.none();
        }
        return List.of(
                new ParameterRef("rnn.inputWeights", inputWeights.values(), inputWeightGradient),
                new ParameterRef("rnn.recurrentWeights", recurrentWeights.values(), recurrentWeightGradient),
                new ParameterRef("rnn.bias", bias.values(), biasGradient));
    }

    @Override
    public ModelKind kind() {
        return ModelKind.VANILLA_RNN;
    }

    @Override
    public void write(ModelWriter writer) {
        writer.putInt(hiddenSize);
        writer.putLong(seed);
        writer.putInt(inputFeatures);
        if (inputFeatures > 0) {
            writer.putDoubles(inputWeights.values());
            writer.putDoubles(recurrentWeights.values());
            writer.putDoubles(bias.values());
        }
    }

    public static VanillaRNN read(ModelReader reader) {
        int hidden = reader.getInt();
        long seed = reader.getLong();
        int features = reader.getInt();
        if (hidden <= 0) {
            throw new ModelFormatException("Invalid VanillaRNN hidden size");
        }
        VanillaRNN rnn = new VanillaRNN(new Builder().hiddenSize(hidden).seed(seed));
        if (features > 0) {
            rnn.initialize(features);
            double[] w = reader.getDoubles(hidden * features);
            double[] u = reader.getDoubles(hidden * hidden);
            double[] b = reader.getDoubles(hidden);
            System.arraycopy(w, 0, rnn.inputWeights.values(), 0, w.length);
            System.arraycopy(u, 0, rnn.recurrentWeights.values(), 0, u.length);
            System.arraycopy(b, 0, rnn.bias.values(), 0, b.length);
        } else if (features != -1) {
            throw new ModelFormatException("Invalid VanillaRNN input feature marker");
        }
        return rnn;
    }

    public static final class Builder {
        private int hiddenSize = 16;
        private long seed = 42;

        public Builder hiddenSize(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public VanillaRNN build() {
            if (hiddenSize <= 0) {
                throw new IllegalArgumentException("hiddenSize must be positive");
            }
            return new VanillaRNN(this);
        }
    }
}
