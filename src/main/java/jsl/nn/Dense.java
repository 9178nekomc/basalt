package jsl.nn;

import jsl.core.BC;
import jsl.core.ModelReader;
import jsl.core.ModelWriter;
import jsl.core.Tensor;
import jsl.core.VectorMath;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/** Fully connected BC -> BC layer with complete forward and backward passes. */
public final class Dense implements Layer<BC, BC> {
    private final int units;
    private final Activation activation;
    private final long seed;
    private int inputFeatures = -1;
    private Parameter weights;
    private Parameter bias;
    private Tensor<BC> cachedInput;
    private double[] preActivation;
    private double[] activated;
    private double[] weightGradient;
    private double[] biasGradient;
    private boolean gradientReady;

    private Dense(Builder builder) {
        this.units = builder.units;
        this.activation = builder.activation;
        this.seed = builder.seed;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Tensor<BC> forward(Tensor<BC> input) {
        gradientReady = false;
        int features = input.dimension(1);
        if (inputFeatures < 0) {
            initialize(features);
        } else if (features != inputFeatures) {
            throw new IllegalArgumentException(
                    "Dense expected " + inputFeatures + " features but got " + features);
        }
        int batch = input.dimension(0);
        int outputSize = batch * units;
        preActivation = new double[outputSize];
        activated = new double[outputSize];
        double[] x = input.data();
        double[] w = weights.values();
        double[] b = bias.values();
        for (int row = 0; row < batch; row++) {
            int inputOffset = row * features;
            int outputOffset = row * units;
            for (int unit = 0; unit < units; unit++) {
                preActivation[outputOffset + unit] = b[unit] + VectorMath.dot(
                        w, unit * features, x, inputOffset, features);
            }
            System.arraycopy(preActivation, outputOffset, activated, outputOffset, units);
            activation.applyRow(activated, outputOffset, units);
        }
        cachedInput = input;
        return Tensor.ofBC(batch, units, activated);
    }

    private void initialize(int features) {
        inputFeatures = features;
        double[] values = new double[units * features];
        double limit = activation == Activation.RELU
                ? Math.sqrt(6.0 / features)
                : Math.sqrt(6.0 / (features + units));
        Random random = new Random(seed);
        for (int i = 0; i < values.length; i++) {
            values[i] = (random.nextDouble() * 2.0 - 1.0) * limit;
        }
        weights = new Parameter(values);
        bias = new Parameter(new double[units]);
        weightGradient = new double[values.length];
        biasGradient = new double[units];
    }

    @Override
    public Tensor<BC> backward(Tensor<BC> gradient) {
        return backwardInternal(gradient, false);
    }

    Tensor<BC> backwardFromSoftmaxLoss(Tensor<BC> logitGradient) {
        if (activation != Activation.SOFTMAX) {
            throw new IllegalStateException("Softmax logit gradients require a SOFTMAX Dense layer");
        }
        return backwardInternal(logitGradient, true);
    }

    private Tensor<BC> backwardInternal(Tensor<BC> gradient, boolean logitGradient) {
        gradientReady = false;
        if (cachedInput == null) {
            throw new IllegalStateException("Dense backward requires a preceding forward pass");
        }
        int batch = cachedInput.dimension(0);
        if (gradient.dimension(0) != batch || gradient.dimension(1) != units) {
            throw new IllegalArgumentException("Dense gradient shape mismatch");
        }
        Arrays.fill(weightGradient, 0.0);
        Arrays.fill(biasGradient, 0.0);
        Tensor<BC> inputGradient = Tensor.zerosBC(batch, inputFeatures);
        double[] dx = inputGradient.data();
        double[] x = cachedInput.data();
        double[] g = gradient.data();
        double[] w = weights.values();

        for (int row = 0; row < batch; row++) {
            int inputOffset = row * inputFeatures;
            int outputOffset = row * units;
            double softmaxDot = 0.0;
            if (activation == Activation.SOFTMAX && !logitGradient) {
                for (int unit = 0; unit < units; unit++) {
                    int index = outputOffset + unit;
                    softmaxDot += g[index] * activated[index];
                }
            }
            for (int unit = 0; unit < units; unit++) {
                int outputIndex = outputOffset + unit;
                double localGradient;
                if (logitGradient) {
                    localGradient = g[outputIndex];
                } else if (activation == Activation.SOFTMAX) {
                    localGradient = activated[outputIndex] * (g[outputIndex] - softmaxDot);
                } else {
                    localGradient = g[outputIndex] * activation.derivative(
                            preActivation[outputIndex], activated[outputIndex]);
                }
                biasGradient[unit] += localGradient;
                int weightOffset = unit * inputFeatures;
                for (int feature = 0; feature < inputFeatures; feature++) {
                    int inputIndex = inputOffset + feature;
                    weightGradient[weightOffset + feature] += localGradient * x[inputIndex];
                    dx[inputIndex] += localGradient * w[weightOffset + feature];
                }
            }
        }
        gradientReady = true;
        return inputGradient;
    }

    @Override
    public void update(OptimizerConfig optimizer) {
        if (weights == null || !gradientReady) {
            throw new IllegalStateException("Dense update requires forward and backward passes");
        }
        weights.update(weightGradient, optimizer);
        bias.update(biasGradient, optimizer);
        gradientReady = false;
    }

    Activation activation() {
        return activation;
    }

    @Override
    public List<ParameterRef> parameters() {
        if (weights == null) {
            return ParameterRef.none();
        }
        return List.of(
                new ParameterRef("dense.weights", weights.values(), weightGradient),
                new ParameterRef("dense.bias", bias.values(), biasGradient));
    }

    void write(ModelWriter writer) {
        writer.putInt(units);
        writer.putInt(activation.ordinal());
        writer.putLong(seed);
        writer.putInt(inputFeatures);
        if (inputFeatures > 0) {
            writer.putDoubles(weights.values());
            writer.putDoubles(bias.values());
        }
    }

    static Dense read(ModelReader reader) {
        int units = reader.getInt();
        int activationOrdinal = reader.getInt();
        long seed = reader.getLong();
        int features = reader.getInt();
        Activation[] activations = Activation.values();
        if (units <= 0 || activationOrdinal < 0 || activationOrdinal >= activations.length) {
            throw new jsl.core.ModelFormatException("Invalid Dense header");
        }
        Dense dense = Dense.builder().units(units)
                .activation(activations[activationOrdinal]).seed(seed).build();
        if (features > 0) {
            double[] weightValues = reader.getDoubles(Math.multiplyExact(units, features));
            double[] biasValues = reader.getDoubles(units);
            dense.restore(features, weightValues, biasValues);
        } else if (features != -1) {
            throw new jsl.core.ModelFormatException("Invalid Dense input feature marker");
        }
        return dense;
    }

    long seed() {
        return seed;
    }

    int inputFeatures() {
        return inputFeatures;
    }

    double[] weightValues() {
        return weights.values();
    }

    double[] biasValues() {
        return bias.values();
    }

    void restore(int features, double[] weightValues, double[] biasValues) {
        initialize(features);
        System.arraycopy(weightValues, 0, weights.values(), 0, weightValues.length);
        System.arraycopy(biasValues, 0, bias.values(), 0, biasValues.length);
    }

    public int units() {
        return units;
    }

    public static final class Builder {
        private int units = 1;
        private Activation activation = Activation.NONE;
        private long seed = 42;

        public Builder units(int units) {
            this.units = units;
            return this;
        }

        public Builder activation(Activation activation) {
            this.activation = java.util.Objects.requireNonNull(activation, "activation");
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Dense build() {
            if (units <= 0) {
                throw new IllegalArgumentException("units must be positive");
            }
            return new Dense(this);
        }
    }
}
