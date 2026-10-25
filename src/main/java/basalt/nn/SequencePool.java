package basalt.nn;

import basalt.core.BC;
import basalt.core.BSF;
import basalt.core.ModelReader;
import basalt.core.ModelWriter;
import basalt.core.Tensor;

/**
 * Collapses a sequence into one vector per batch element.
 *
 * <p>{@link Pooling#LAST} is the standard many-to-one head for classification;
 * {@link Pooling#MEAN} averages over time and therefore touches every timestep. Backward mirrors
 * the operation exactly: LAST routes the gradient to the final timestep, MEAN divides it evenly.</p>
 */
public final class SequencePool implements Layer<BSF, BC> {
    private final Pooling pooling;
    private int batch;
    private int sequence;
    private int features;

    private SequencePool(Pooling pooling) {
        this.pooling = pooling;
    }

    public static SequencePool of(Pooling pooling) {
        if (pooling == null) {
            throw new IllegalArgumentException("pooling must not be null");
        }
        return new SequencePool(pooling);
    }

    @Override
    public Tensor<BC> forward(Tensor<BSF> input) {
        batch = input.dimension(0);
        sequence = input.dimension(1);
        features = input.dimension(2);
        Tensor<BC> output = Tensor.zerosBC(batch, features);
        double[] source = input.data();
        double[] target = output.data();
        if (pooling == Pooling.LAST) {
            for (int row = 0; row < batch; row++) {
                System.arraycopy(source, (row * sequence + sequence - 1) * features,
                        target, row * features, features);
            }
            return output;
        }
        double inverse = 1.0 / sequence;
        for (int row = 0; row < batch; row++) {
            int base = row * sequence * features;
            for (int time = 0; time < sequence; time++) {
                int offset = base + time * features;
                for (int feature = 0; feature < features; feature++) {
                    target[row * features + feature] += source[offset + feature];
                }
            }
            for (int feature = 0; feature < features; feature++) {
                target[row * features + feature] *= inverse;
            }
        }
        return output;
    }

    @Override
    public Tensor<BSF> backward(Tensor<BC> gradient) {
        if (gradient.dimension(0) != batch || gradient.dimension(1) != features) {
            throw new IllegalArgumentException("SequencePool gradient shape mismatch");
        }
        Tensor<BSF> inputGradient = Tensor.zerosBSF(batch, sequence, features);
        double[] source = gradient.data();
        double[] target = inputGradient.data();
        if (pooling == Pooling.LAST) {
            for (int row = 0; row < batch; row++) {
                System.arraycopy(source, row * features,
                        target, (row * sequence + sequence - 1) * features, features);
            }
            return inputGradient;
        }
        double inverse = 1.0 / sequence;
        for (int row = 0; row < batch; row++) {
            int base = row * sequence * features;
            for (int time = 0; time < sequence; time++) {
                int offset = base + time * features;
                for (int feature = 0; feature < features; feature++) {
                    target[offset + feature] = source[row * features + feature] * inverse;
                }
            }
        }
        return inputGradient;
    }

    public Pooling pooling() {
        return pooling;
    }

    void write(ModelWriter writer) {
        writer.putInt(pooling.ordinal());
    }

    static SequencePool read(ModelReader reader) {
        int ordinal = reader.getInt();
        Pooling[] values = Pooling.values();
        if (ordinal < 0 || ordinal >= values.length) {
            throw new basalt.core.ModelFormatException("Invalid pooling ordinal " + ordinal);
        }
        return new SequencePool(values[ordinal]);
    }

    /** Reduction applied along the time axis. */
    public enum Pooling {
        LAST,
        MEAN
    }
}
