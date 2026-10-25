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
 * Many-to-many LSTM with explicit backpropagation through time.
 *
 * <p>Gate order is input, forget, cell, output. The constant error carousel
 * {@code c_t = f_t * c_{t-1} + i_t * g_t} has Jacobian {@code dc_t/dc_{t-1} = f_t}; because the
 * forget gate is initialized near one and is learned, gradients do not have to pass through a
 * product of squashed derivatives. Every per-timestep buffer is retained between calls and reused
 * when the batch shape is unchanged.</p>
 */
public final class LSTM implements Layer<BSF, BSF>, Persistable {
    private final int hiddenSize;
    private final long seed;
    private int inputFeatures = -1;
    private Parameter inputWeights;
    private Parameter recurrentWeights;
    private Parameter bias;
    private Tensor<BSF> cachedInput;
    private double[] hidden;
    private double[] cell;
    private double[] inputGate;
    private double[] forgetGate;
    private double[] cellGate;
    private double[] outputGate;
    private double[] inputWeightGradient;
    private double[] recurrentWeightGradient;
    private double[] biasGradient;
    private double[] nextHidden;
    private double[] currentHidden;
    private double[] nextCell;
    private double[] currentCell;
    private boolean gradientReady;

    private LSTM(Builder builder) {
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
            throw new IllegalArgumentException("LSTM input feature count mismatch");
        }
        cachedInput = input;
        int batch = input.dimension(0);
        int sequence = input.dimension(1);
        int timeSize = batch * sequence * hiddenSize;
        hidden = ensure(hidden, timeSize);
        cell = ensure(cell, timeSize);
        inputGate = ensure(inputGate, timeSize);
        forgetGate = ensure(forgetGate, timeSize);
        cellGate = ensure(cellGate, timeSize);
        outputGate = ensure(outputGate, timeSize);
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
                    for (int gate = 0; gate < 4; gate++) {
                        double sum = b[gate * hiddenSize + unit];
                        int inputWeightBase = (gate * hiddenSize + unit) * features;
                        int recurrentWeightBase = (gate * hiddenSize + unit) * hiddenSize;
                        for (int feature = 0; feature < features; feature++) {
                            sum += w[inputWeightBase + feature] * x[inputBase + feature];
                        }
                        if (time > 0) {
                            for (int previous = 0; previous < hiddenSize; previous++) {
                                sum += u[recurrentWeightBase + previous] * hidden[previousBase + previous];
                            }
                        }
                        double activated = gate == 2 ? Math.tanh(sum) : sigmoid(sum);
                        switch (gate) {
                            case 0 -> inputGate[timeBase + unit] = activated;
                            case 1 -> forgetGate[timeBase + unit] = activated;
                            case 2 -> cellGate[timeBase + unit] = activated;
                            default -> outputGate[timeBase + unit] = activated;
                        }
                    }
                    double previousCell = time > 0 ? cell[previousBase + unit] : 0.0;
                    cell[timeBase + unit] = forgetGate[timeBase + unit] * previousCell
                            + inputGate[timeBase + unit] * cellGate[timeBase + unit];
                    hidden[timeBase + unit] = outputGate[timeBase + unit]
                            * Math.tanh(cell[timeBase + unit]);
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
        double[] w = new double[4 * hiddenSize * features];
        double[] u = new double[4 * hiddenSize * hiddenSize];
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
        bias = new Parameter(new double[4 * hiddenSize]);
        inputWeightGradient = new double[w.length];
        recurrentWeightGradient = new double[u.length];
        biasGradient = new double[4 * hiddenSize];
    }

    @Override
    public Tensor<BSF> backward(Tensor<BSF> gradient) {
        gradientReady = false;
        if (cachedInput == null) {
            throw new IllegalStateException("LSTM backward requires forward");
        }
        int batch = cachedInput.dimension(0);
        int sequence = cachedInput.dimension(1);
        if (gradient.dimension(0) != batch || gradient.dimension(1) != sequence
                || gradient.dimension(2) != hiddenSize) {
            throw new IllegalArgumentException("LSTM gradient shape mismatch");
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
        nextHidden = ensure(nextHidden, batchHidden);
        currentHidden = ensure(currentHidden, batchHidden);
        nextCell = ensure(nextCell, batchHidden);
        currentCell = ensure(currentCell, batchHidden);
        Arrays.fill(nextHidden, 0.0);
        Arrays.fill(nextCell, 0.0);
        double[] gateGradient = new double[4];

        for (int time = sequence - 1; time >= 0; time--) {
            Arrays.fill(currentHidden, 0.0);
            Arrays.fill(currentCell, 0.0);
            for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
                int timeBase = (batchIndex * sequence + time) * hiddenSize;
                int previousBase = timeBase - hiddenSize;
                int inputBase = (batchIndex * sequence + time) * inputFeatures;
                int recurrentBatchBase = batchIndex * hiddenSize;
                for (int unit = 0; unit < hiddenSize; unit++) {
                    int index = timeBase + unit;
                    double tanhCell = Math.tanh(cell[index]);
                    double hiddenGradient = g[index] + nextHidden[recurrentBatchBase + unit];
                    double outputDerivative = hiddenGradient * tanhCell;
                    double cellGradient = nextCell[recurrentBatchBase + unit]
                            + hiddenGradient * outputGate[index] * (1.0 - tanhCell * tanhCell);
                    double previousCell = time > 0 ? cell[previousBase + unit] : 0.0;
                    double forgetDerivative = cellGradient * previousCell;
                    double inputDerivative = cellGradient * cellGate[index];
                    double cellDerivative = cellGradient * inputGate[index];
                    currentCell[recurrentBatchBase + unit] = cellGradient * forgetGate[index];
                    gateGradient[0] = inputDerivative * inputGate[index] * (1.0 - inputGate[index]);
                    gateGradient[1] = forgetDerivative * forgetGate[index] * (1.0 - forgetGate[index]);
                    gateGradient[2] = cellDerivative * (1.0 - cellGate[index] * cellGate[index]);
                    gateGradient[3] = outputDerivative * outputGate[index] * (1.0 - outputGate[index]);

                    for (int gate = 0; gate < 4; gate++) {
                        double local = gateGradient[gate];
                        biasGradient[gate * hiddenSize + unit] += local;
                        int inputWeightBase = (gate * hiddenSize + unit) * inputFeatures;
                        int recurrentWeightBase = (gate * hiddenSize + unit) * hiddenSize;
                        for (int feature = 0; feature < inputFeatures; feature++) {
                            inputWeightGradient[inputWeightBase + feature] += local * x[inputBase + feature];
                            dx[inputBase + feature] += local * w[inputWeightBase + feature];
                        }
                        if (time > 0) {
                            for (int previous = 0; previous < hiddenSize; previous++) {
                                recurrentWeightGradient[recurrentWeightBase + previous] +=
                                        local * hidden[previousBase + previous];
                                currentHidden[recurrentBatchBase + previous] +=
                                        local * u[recurrentWeightBase + previous];
                            }
                        }
                    }
                }
            }
            double[] temporaryHidden = nextHidden;
            nextHidden = currentHidden;
            currentHidden = temporaryHidden;
            double[] temporaryCell = nextCell;
            nextCell = currentCell;
            currentCell = temporaryCell;
        }
        gradientReady = true;
        return inputGradient;
    }

    @Override
    public void update(OptimizerConfig optimizer) {
        if (!gradientReady) {
            throw new IllegalStateException("LSTM update requires forward and backward");
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
                new ParameterRef("lstm.inputWeights", inputWeights.values(), inputWeightGradient),
                new ParameterRef("lstm.recurrentWeights", recurrentWeights.values(), recurrentWeightGradient),
                new ParameterRef("lstm.bias", bias.values(), biasGradient));
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
        return ModelKind.LSTM;
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

    public static LSTM read(ModelReader reader) {
        int hidden = reader.getInt();
        long seed = reader.getLong();
        int features = reader.getInt();
        if (hidden <= 0) {
            throw new ModelFormatException("Invalid LSTM hidden size");
        }
        LSTM lstm = new LSTM(new Builder().hiddenSize(hidden).seed(seed));
        if (features > 0) {
            lstm.initialize(features);
            double[] w = reader.getDoubles(4 * hidden * features);
            double[] u = reader.getDoubles(4 * hidden * hidden);
            double[] b = reader.getDoubles(4 * hidden);
            System.arraycopy(w, 0, lstm.inputWeights.values(), 0, w.length);
            System.arraycopy(u, 0, lstm.recurrentWeights.values(), 0, u.length);
            System.arraycopy(b, 0, lstm.bias.values(), 0, b.length);
        } else if (features != -1) {
            throw new ModelFormatException("Invalid LSTM input feature marker");
        }
        return lstm;
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

        public LSTM build() {
            if (hiddenSize <= 0) {
                throw new IllegalArgumentException("hiddenSize must be positive");
            }
            return new LSTM(this);
        }
    }
}
