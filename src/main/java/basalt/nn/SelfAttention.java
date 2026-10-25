package basalt.nn;

import basalt.core.BSF;
import basalt.core.ModelFormatException;
import basalt.core.ModelKind;
import basalt.core.ModelReader;
import basalt.core.ModelWriter;
import basalt.core.Persistable;
import basalt.core.Tensor;
import basalt.core.VectorMath;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Single-head scaled dot-product self-attention with full Q/K/V and Softmax derivatives.
 *
 * <p>{@code A = softmax(Q K^T / sqrt(d))}, {@code O = A V}. The backward pass applies the exact
 * Softmax Jacobian {@code dS = A * (dA - rowsum(dA * A))} before the scaling factor, then
 * accumulates {@code dQ = dS K}, {@code dK = dS^T Q} and {@code dV = A^T dO}. An optional causal
 * mask restricts row {@code t} to keys {@code 0..t}.</p>
 */
public final class SelfAttention implements Layer<BSF, BSF>, Persistable {
    private final int dimension;
    private final long seed;
    private final boolean causal;
    private final double scale;
    private Parameter queryWeights;
    private Parameter keyWeights;
    private Parameter valueWeights;
    private Tensor<BSF> cachedInput;
    private double[] query;
    private double[] key;
    private double[] value;
    private double[] probabilities;
    private double[] output;
    private double[] queryGradient;
    private double[] keyGradient;
    private double[] valueGradient;
    private double[] scoreGradient;
    private double[] queryWeightGradient;
    private double[] keyWeightGradient;
    private double[] valueWeightGradient;
    private boolean gradientReady;

    private SelfAttention(Builder builder) {
        dimension = builder.dimension;
        seed = builder.seed;
        causal = builder.causal;
        scale = 1.0 / Math.sqrt(dimension);
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Tensor<BSF> forward(Tensor<BSF> input) {
        gradientReady = false;
        if (input.dimension(2) != dimension) {
            throw new IllegalArgumentException(
                    "SelfAttention input feature count must equal model dimension");
        }
        if (queryWeights == null) {
            initialize();
        }
        cachedInput = input;
        int batch = input.dimension(0);
        int sequence = input.dimension(1);
        int vectors = batch * sequence * dimension;
        int attentionSize = batch * sequence * sequence;
        query = ensure(query, vectors);
        key = ensure(key, vectors);
        value = ensure(value, vectors);
        probabilities = ensure(probabilities, attentionSize);
        output = ensure(output, vectors);
        Arrays.fill(output, 0.0);
        project(input.data(), query, queryWeights.values(), batch, sequence);
        project(input.data(), key, keyWeights.values(), batch, sequence);
        project(input.data(), value, valueWeights.values(), batch, sequence);

        for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
            for (int row = 0; row < sequence; row++) {
                int probabilityBase = (batchIndex * sequence + row) * sequence;
                int queryBase = (batchIndex * sequence + row) * dimension;
                int limit = causal ? row + 1 : sequence;
                double maximum = Double.NEGATIVE_INFINITY;
                for (int column = 0; column < limit; column++) {
                    int keyBase = (batchIndex * sequence + column) * dimension;
                    double score = VectorMath.dot(query, queryBase, key, keyBase, dimension) * scale;
                    probabilities[probabilityBase + column] = score;
                    maximum = Math.max(maximum, score);
                }
                for (int column = limit; column < sequence; column++) {
                    probabilities[probabilityBase + column] = 0.0;
                }
                double sum = 0.0;
                for (int column = 0; column < limit; column++) {
                    double exponential = Math.exp(probabilities[probabilityBase + column] - maximum);
                    probabilities[probabilityBase + column] = exponential;
                    sum += exponential;
                }
                for (int column = 0; column < limit; column++) {
                    double probability = probabilities[probabilityBase + column] / sum;
                    probabilities[probabilityBase + column] = probability;
                    int valueBase = (batchIndex * sequence + column) * dimension;
                    for (int feature = 0; feature < dimension; feature++) {
                        output[queryBase + feature] += probability * value[valueBase + feature];
                    }
                }
            }
        }
        return Tensor.ofBSF(batch, sequence, dimension, output);
    }

    private static double[] ensure(double[] array, int size) {
        return array != null && array.length == size ? array : new double[size];
    }

    private void project(double[] input, double[] target, double[] weights, int batch, int sequence) {
        for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
            for (int time = 0; time < sequence; time++) {
                int vectorBase = (batchIndex * sequence + time) * dimension;
                for (int outputFeature = 0; outputFeature < dimension; outputFeature++) {
                    target[vectorBase + outputFeature] = VectorMath.dot(
                            weights, outputFeature * dimension, input, vectorBase, dimension);
                }
            }
        }
    }

    private void initialize() {
        Random random = new Random(seed);
        double limit = Math.sqrt(3.0 / dimension);
        queryWeights = new Parameter(randomWeights(random, limit));
        keyWeights = new Parameter(randomWeights(random, limit));
        valueWeights = new Parameter(randomWeights(random, limit));
        int size = dimension * dimension;
        queryWeightGradient = new double[size];
        keyWeightGradient = new double[size];
        valueWeightGradient = new double[size];
    }

    private double[] randomWeights(Random random, double limit) {
        double[] result = new double[dimension * dimension];
        for (int i = 0; i < result.length; i++) {
            result[i] = (random.nextDouble() * 2.0 - 1.0) * limit;
        }
        return result;
    }

    @Override
    public Tensor<BSF> backward(Tensor<BSF> gradient) {
        gradientReady = false;
        if (cachedInput == null) {
            throw new IllegalStateException("SelfAttention backward requires forward");
        }
        int batch = cachedInput.dimension(0);
        int sequence = cachedInput.dimension(1);
        if (gradient.dimension(0) != batch || gradient.dimension(1) != sequence
                || gradient.dimension(2) != dimension) {
            throw new IllegalArgumentException("SelfAttention gradient shape mismatch");
        }
        int vectors = batch * sequence * dimension;
        int attentionSize = batch * sequence * sequence;
        queryGradient = ensure(queryGradient, vectors);
        keyGradient = ensure(keyGradient, vectors);
        valueGradient = ensure(valueGradient, vectors);
        scoreGradient = ensure(scoreGradient, attentionSize);
        Arrays.fill(queryGradient, 0.0);
        Arrays.fill(keyGradient, 0.0);
        Arrays.fill(valueGradient, 0.0);
        Arrays.fill(scoreGradient, 0.0);
        Arrays.fill(queryWeightGradient, 0.0);
        Arrays.fill(keyWeightGradient, 0.0);
        Arrays.fill(valueWeightGradient, 0.0);
        double[] g = gradient.data();

        for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
            for (int row = 0; row < sequence; row++) {
                int vectorBase = (batchIndex * sequence + row) * dimension;
                int probabilityBase = (batchIndex * sequence + row) * sequence;
                int limit = causal ? row + 1 : sequence;
                for (int column = 0; column < limit; column++) {
                    int valueBase = (batchIndex * sequence + column) * dimension;
                    double probability = probabilities[probabilityBase + column];
                    double probabilityDerivative =
                            VectorMath.dot(g, vectorBase, value, valueBase, dimension);
                    scoreGradient[probabilityBase + column] = probabilityDerivative;
                    for (int feature = 0; feature < dimension; feature++) {
                        valueGradient[valueBase + feature] += probability * g[vectorBase + feature];
                    }
                }
                double softmaxDot = 0.0;
                for (int column = 0; column < limit; column++) {
                    softmaxDot += scoreGradient[probabilityBase + column]
                            * probabilities[probabilityBase + column];
                }
                for (int column = 0; column < limit; column++) {
                    int probabilityIndex = probabilityBase + column;
                    scoreGradient[probabilityIndex] = probabilities[probabilityIndex]
                            * (scoreGradient[probabilityIndex] - softmaxDot) * scale;
                    int keyBase = (batchIndex * sequence + column) * dimension;
                    for (int feature = 0; feature < dimension; feature++) {
                        queryGradient[vectorBase + feature] +=
                                scoreGradient[probabilityIndex] * key[keyBase + feature];
                        keyGradient[keyBase + feature] +=
                                scoreGradient[probabilityIndex] * query[vectorBase + feature];
                    }
                }
            }
        }

        Tensor<BSF> inputGradient = Tensor.zerosBSF(batch, sequence, dimension);
        double[] dx = inputGradient.data();
        double[] x = cachedInput.data();
        double[] wq = queryWeights.values();
        double[] wk = keyWeights.values();
        double[] wv = valueWeights.values();
        for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
            for (int time = 0; time < sequence; time++) {
                int vectorBase = (batchIndex * sequence + time) * dimension;
                for (int outputFeature = 0; outputFeature < dimension; outputFeature++) {
                    double dq = queryGradient[vectorBase + outputFeature];
                    double dk = keyGradient[vectorBase + outputFeature];
                    double dv = valueGradient[vectorBase + outputFeature];
                    int weightBase = outputFeature * dimension;
                    for (int inputFeature = 0; inputFeature < dimension; inputFeature++) {
                        int weightIndex = weightBase + inputFeature;
                        int inputIndex = vectorBase + inputFeature;
                        dx[inputIndex] += wq[weightIndex] * dq + wk[weightIndex] * dk
                                + wv[weightIndex] * dv;
                        queryWeightGradient[weightIndex] += dq * x[inputIndex];
                        keyWeightGradient[weightIndex] += dk * x[inputIndex];
                        valueWeightGradient[weightIndex] += dv * x[inputIndex];
                    }
                }
            }
        }
        gradientReady = true;
        return inputGradient;
    }

    @Override
    public void update(OptimizerConfig optimizer) {
        if (!gradientReady) {
            throw new IllegalStateException("SelfAttention update requires forward and backward");
        }
        queryWeights.update(queryWeightGradient, optimizer);
        keyWeights.update(keyWeightGradient, optimizer);
        valueWeights.update(valueWeightGradient, optimizer);
        gradientReady = false;
    }

    @Override
    public List<ParameterRef> parameters() {
        if (queryWeights == null) {
            return ParameterRef.none();
        }
        return List.of(
                new ParameterRef("attention.query", queryWeights.values(), queryWeightGradient),
                new ParameterRef("attention.key", keyWeights.values(), keyWeightGradient),
                new ParameterRef("attention.value", valueWeights.values(), valueWeightGradient));
    }

    public int dimension() {
        return dimension;
    }

    public boolean causal() {
        return causal;
    }

    @Override
    public ModelKind kind() {
        return ModelKind.SELF_ATTENTION;
    }

    @Override
    public void write(ModelWriter writer) {
        writer.putInt(dimension);
        writer.putLong(seed);
        writer.putByte(causal ? 1 : 0);
        writer.putByte(queryWeights == null ? 0 : 1);
        if (queryWeights != null) {
            writer.putDoubles(queryWeights.values());
            writer.putDoubles(keyWeights.values());
            writer.putDoubles(valueWeights.values());
        }
    }

    public static SelfAttention read(ModelReader reader) {
        int dimension = reader.getInt();
        long seed = reader.getLong();
        int causalFlag = reader.getByte();
        int initialized = reader.getByte();
        if (dimension <= 0 || (causalFlag != 0 && causalFlag != 1)
                || (initialized != 0 && initialized != 1)) {
            throw new ModelFormatException("Invalid SelfAttention header");
        }
        SelfAttention attention = new SelfAttention(new Builder()
                .dimension(dimension).seed(seed).causal(causalFlag == 1));
        if (initialized == 1) {
            attention.initialize();
            int size = dimension * dimension;
            double[] wq = reader.getDoubles(size);
            double[] wk = reader.getDoubles(size);
            double[] wv = reader.getDoubles(size);
            System.arraycopy(wq, 0, attention.queryWeights.values(), 0, size);
            System.arraycopy(wk, 0, attention.keyWeights.values(), 0, size);
            System.arraycopy(wv, 0, attention.valueWeights.values(), 0, size);
        }
        return attention;
    }

    public static final class Builder {
        private int dimension = 16;
        private long seed = 42;
        private boolean causal;

        public Builder dimension(int dimension) {
            this.dimension = dimension;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        /** Restricts each query position to keys at or before it. */
        public Builder causal(boolean causal) {
            this.causal = causal;
            return this;
        }

        public SelfAttention build() {
            if (dimension <= 0) {
                throw new IllegalArgumentException("dimension must be positive");
            }
            return new SelfAttention(this);
        }
    }
}
