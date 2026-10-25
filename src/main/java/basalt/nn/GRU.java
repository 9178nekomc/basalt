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
 * Many-to-many GRU with explicit backpropagation through time.
 *
 * <p>Gate order is update ({@code z}), reset ({@code r}) and candidate ({@code n}):
 * {@code z = sigmoid(Wz x + Uz h + bz)}, {@code r = sigmoid(Wr x + Ur h + br)},
 * {@code n = tanh(Wn x + Un (r * h) + bn)}, {@code h' = (1 - z) * n + z * h}.</p>
 *
 * <p>The update gate carries the previous state additively, so its Jacobian contains
 * {@code dh'/dh = z + ...}, which stays close to the identity early in training. That is the
 * mechanism that keeps gradients alive over long sequences, in contrast to the product of
 * {@code tanh'} factors in a vanilla RNN.</p>
 */
public final class GRU implements Layer<BSF, BSF>, Persistable {
    private final int hiddenSize;
    private final long seed;
    private int inputFeatures = -1;
    private Parameter inputWeights;
    private Parameter recurrentWeights;
    private Parameter bias;
    private Tensor<BSF> cachedInput;
    private double[] hidden;
    private double[] updateGate;
    private double[] resetGate;
    private double[] candidate;
    private double[] inputWeightGradient;
    private double[] recurrentWeightGradient;
    private double[] biasGradient;
    private boolean gradientReady;

    private GRU(Builder builder) {
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
            throw new IllegalArgumentException("GRU input feature count mismatch");
        }
        cachedInput = input;
        int batch = input.dimension(0);
        int sequence = input.dimension(1);
        int size = batch * sequence * hiddenSize;
        hidden = ensure(hidden, size);
        updateGate = ensure(updateGate, size);
        resetGate = ensure(resetGate, size);
        candidate = ensure(candidate, size);
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
                    double updateSum = b[unit];
                    double resetSum = b[hiddenSize + unit];
                    int updateInputBase = unit * features;
                    int resetInputBase = (hiddenSize + unit) * features;
                    for (int feature = 0; feature < features; feature++) {
                        double value = x[inputBase + feature];
                        updateSum += w[updateInputBase + feature] * value;
                        resetSum += w[resetInputBase + feature] * value;
                    }
                    if (time > 0) {
                        int updateRecurrentBase = unit * hiddenSize;
                        int resetRecurrentBase = (hiddenSize + unit) * hiddenSize;
                        for (int previous = 0; previous < hiddenSize; previous++) {
                            double previousHidden = hidden[previousBase + previous];
                            updateSum += u[updateRecurrentBase + previous] * previousHidden;
                            resetSum += u[resetRecurrentBase + previous] * previousHidden;
                        }
                    }
                    updateGate[timeBase + unit] = sigmoid(updateSum);
                    resetGate[timeBase + unit] = sigmoid(resetSum);
                }
                for (int unit = 0; unit < hiddenSize; unit++) {
                    double candidateSum = b[2 * hiddenSize + unit];
                    int candidateInputBase = (2 * hiddenSize + unit) * features;
                    for (int feature = 0; feature < features; feature++) {
                        candidateSum += w[candidateInputBase + feature] * x[inputBase + feature];
                    }
                    if (time > 0) {
                        int candidateRecurrentBase = (2 * hiddenSize + unit) * hiddenSize;
                        for (int previous = 0; previous < hiddenSize; previous++) {
                            candidateSum += u[candidateRecurrentBase + previous]
                                    * resetGate[timeBase + previous]
                                    * hidden[previousBase + previous];
                        }
                    }
                    double cell = Math.tanh(candidateSum);
                    candidate[timeBase + unit] = cell;
                    double update = updateGate[timeBase + unit];
                    hidden[timeBase + unit] = (1.0 - update) * cell
                            + update * (time > 0 ? hidden[previousBase + unit] : 0.0);
                }
            }
        }
        return Tensor.ofBSF(batch, sequence, hiddenSize, hidden);
    }

    private static double[] ensure(double[] array, int size) {
        return array != null && array.length == size ? array : new double[size];
    }

    private void initialize(int features) {
        inputFeatures = features;
        double[] w = new double[3 * hiddenSize * features];
        double[] u = new double[3 * hiddenSize * hiddenSize];
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
        bias = new Parameter(new double[3 * hiddenSize]);
        inputWeightGradient = new double[w.length];
        recurrentWeightGradient = new double[u.length];
        biasGradient = new double[3 * hiddenSize];
    }

    @Override
    public Tensor<BSF> backward(Tensor<BSF> gradient) {
        gradientReady = false;
        if (cachedInput == null) {
            throw new IllegalStateException("GRU backward requires forward");
        }
        int batch = cachedInput.dimension(0);
        int sequence = cachedInput.dimension(1);
        if (gradient.dimension(0) != batch || gradient.dimension(1) != sequence
                || gradient.dimension(2) != hiddenSize) {
            throw new IllegalArgumentException("GRU gradient shape mismatch");
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
        double[] candidatePre = new double[hiddenSize];
        double[] resetPre = new double[hiddenSize];
        double[] resetWeighted = new double[hiddenSize];

        for (int time = sequence - 1; time >= 0; time--) {
            Arrays.fill(currentHidden, 0.0);
            for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
                int timeBase = (batchIndex * sequence + time) * hiddenSize;
                int previousBase = timeBase - hiddenSize;
                int inputBase = (batchIndex * sequence + time) * inputFeatures;
                int recurrentBatchBase = batchIndex * hiddenSize;
                Arrays.fill(candidatePre, 0.0);
                Arrays.fill(resetPre, 0.0);
                Arrays.fill(resetWeighted, 0.0);

                for (int unit = 0; unit < hiddenSize; unit++) {
                    int index = timeBase + unit;
                    double update = updateGate[index];
                    double reset = resetGate[index];
                    double cell = candidate[index];
                    double previousHidden = time > 0 ? hidden[previousBase + unit] : 0.0;
                    double hiddenGradient = g[index] + nextHidden[recurrentBatchBase + unit];
                    candidatePre[unit] = hiddenGradient * (1.0 - update) * (1.0 - cell * cell);
                    double updateGradient = hiddenGradient * (previousHidden - cell)
                            * update * (1.0 - update);
                    currentHidden[recurrentBatchBase + unit] += hiddenGradient * update;

                    int candidateInputBase = (2 * hiddenSize + unit) * inputFeatures;
                    int updateInputBase = unit * inputFeatures;
                    biasGradient[2 * hiddenSize + unit] += candidatePre[unit];
                    biasGradient[unit] += updateGradient;
                    for (int feature = 0; feature < inputFeatures; feature++) {
                        double inputValue = x[inputBase + feature];
                        inputWeightGradient[candidateInputBase + feature] +=
                                candidatePre[unit] * inputValue;
                        inputWeightGradient[updateInputBase + feature] += updateGradient * inputValue;
                        dx[inputBase + feature] += candidatePre[unit] * w[candidateInputBase + feature]
                                + updateGradient * w[updateInputBase + feature];
                    }
                    if (time > 0) {
                        int candidateRecurrentBase = (2 * hiddenSize + unit) * hiddenSize;
                        int updateRecurrentBase = unit * hiddenSize;
                        for (int previous = 0; previous < hiddenSize; previous++) {
                            double previousValue = hidden[previousBase + previous];
                            recurrentWeightGradient[candidateRecurrentBase + previous] +=
                                    candidatePre[unit] * resetGate[timeBase + previous] * previousValue;
                            recurrentWeightGradient[updateRecurrentBase + previous] +=
                                    updateGradient * previousValue;
                            resetWeighted[previous] += u[candidateRecurrentBase + previous]
                                    * candidatePre[unit];
                            currentHidden[recurrentBatchBase + previous] +=
                                    updateGradient * u[updateRecurrentBase + previous];
                        }
                    }
                }

                for (int unit = 0; unit < hiddenSize; unit++) {
                    if (time > 0) {
                        int previousIndex = previousBase + unit;
                        resetPre[unit] = resetWeighted[unit] * hidden[previousIndex]
                                * resetGate[timeBase + unit] * (1.0 - resetGate[timeBase + unit]);
                        currentHidden[recurrentBatchBase + unit] +=
                                resetWeighted[unit] * resetGate[timeBase + unit];
                    }
                }

                for (int unit = 0; unit < hiddenSize; unit++) {
                    int resetInputBase = (hiddenSize + unit) * inputFeatures;
                    biasGradient[hiddenSize + unit] += resetPre[unit];
                    for (int feature = 0; feature < inputFeatures; feature++) {
                        inputWeightGradient[resetInputBase + feature] +=
                                resetPre[unit] * x[inputBase + feature];
                        dx[inputBase + feature] += resetPre[unit] * w[resetInputBase + feature];
                    }
                    if (time > 0) {
                        int resetRecurrentBase = (hiddenSize + unit) * hiddenSize;
                        for (int previous = 0; previous < hiddenSize; previous++) {
                            recurrentWeightGradient[resetRecurrentBase + previous] +=
                                    resetPre[unit] * hidden[previousBase + previous];
                            currentHidden[recurrentBatchBase + previous] +=
                                    resetPre[unit] * u[resetRecurrentBase + previous];
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
            throw new IllegalStateException("GRU update requires forward and backward");
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
                new ParameterRef("gru.inputWeights", inputWeights.values(), inputWeightGradient),
                new ParameterRef("gru.recurrentWeights", recurrentWeights.values(), recurrentWeightGradient),
                new ParameterRef("gru.bias", bias.values(), biasGradient));
    }

    public int hiddenSize() {
        return hiddenSize;
    }

    private static double sigmoid(double value) {
        if (value >= 0.0) {
            return 1.0 / (1.0 + Math.exp(-value));
        }
        double exponential = Math.exp(value);
        return exponential / (1.0 + exponential);
    }

    @Override
    public ModelKind kind() {
        return ModelKind.GRU;
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

    public static GRU read(ModelReader reader) {
        int hidden = reader.getInt();
        long seed = reader.getLong();
        int features = reader.getInt();
        if (hidden <= 0) {
            throw new ModelFormatException("Invalid GRU hidden size");
        }
        GRU gru = new GRU(new Builder().hiddenSize(hidden).seed(seed));
        if (features > 0) {
            gru.initialize(features);
            double[] w = reader.getDoubles(3 * hidden * features);
            double[] u = reader.getDoubles(3 * hidden * hidden);
            double[] b = reader.getDoubles(3 * hidden);
            System.arraycopy(w, 0, gru.inputWeights.values(), 0, w.length);
            System.arraycopy(u, 0, gru.recurrentWeights.values(), 0, u.length);
            System.arraycopy(b, 0, gru.bias.values(), 0, b.length);
        } else if (features != -1) {
            throw new ModelFormatException("Invalid GRU input feature marker");
        }
        return gru;
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

        public GRU build() {
            if (hiddenSize <= 0) {
                throw new IllegalArgumentException("hiddenSize must be positive");
            }
            return new GRU(this);
        }
    }
}
