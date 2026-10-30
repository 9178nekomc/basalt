package basalt.zoo;

import basalt.Basalt;
import basalt.BasaltRuntimeException;
import basalt.core.BC;
import basalt.core.BCHW;
import basalt.nn.Activation;
import basalt.torch.Flatten;
import basalt.torch.MaxPool2d;
import basalt.torch.Sequential;

/**
 * Ready-made architectures with sane defaults.
 *
 * <p>This is the middle tier: you still control the shape of the network, but you do not write the
 * wiring, the weight initialisation or the activation placement. Every method returns an ordinary
 * {@link Sequential}, so the result can be extended with {@code add(...)}, handed to a
 * {@link Trainer}, inspected with {@code parameters()}, or saved with {@code stateDict()} like any
 * hand-built module.</p>
 *
 * <p>For full control, ignore this class: {@link basalt.torch.Module} is the only contract a layer
 * has to satisfy, and {@code Sequential.add} accepts anything that implements it.</p>
 */
public final class Zoo {

    /** Hidden width used by {@link #tabularClassifier}. */
    private static final int[] DEFAULT_HIDDEN = {64, 32};

    private Zoo() {
    }

    /**
     * Dense network: {@code inputs -> hidden... -> outputs}, ReLU between layers and no activation
     * on the last one, so it can feed a cross-entropy or MSE loss directly.
     *
     * @param hidden widths of the hidden layers, in order; may be empty for a linear model
     */
    public static Sequential<BC, BC> mlp(Basalt engine, int inputs, int[] hidden, int outputs) {
        BasaltRuntimeException.requireNonNull(engine, "engine");
        if (inputs < 1) {
            throw new IllegalArgumentException("inputs must be at least 1, got " + inputs);
        }
        if (outputs < 1) {
            throw new IllegalArgumentException("outputs must be at least 1, got " + outputs);
        }
        int[] widths = hidden == null ? new int[0] : hidden.clone();
        Sequential<BC, BC> network = engine.sequential();
        int previous = inputs;
        for (int width : widths) {
            if (width < 1) {
                throw new IllegalArgumentException("hidden width must be at least 1, got " + width);
            }
            network = network.add(engine.linear(previous, width));
            network = network.add(engine.activation(Activation.RELU));
            previous = width;
        }
        return network.add(engine.linear(previous, outputs));
    }

    /** Two-hidden-layer dense classifier for tabular data. */
    public static Sequential<BC, BC> tabularClassifier(Basalt engine, int features, int classes) {
        return mlp(engine, features, DEFAULT_HIDDEN, classes);
    }

    /**
     * Small convolutional classifier: one convolution, one pooling step, one dense head.
     *
     * <p>The convolution keeps the spatial size ({@code kernel 3, stride 1, padding 1}) and the
     * pooling halves it, so the flattened width is {@code 8 * (height / 2) * (width / 2)}. Both
     * dimensions must therefore be even.</p>
     */
    public static Sequential<BCHW, BC> cnnClassifier(Basalt engine, int channels, int height,
                                                     int width, int classes) {
        BasaltRuntimeException.requireNonNull(engine, "engine");
        if (channels < 1) {
            throw new IllegalArgumentException("channels must be at least 1, got " + channels);
        }
        if (classes < 1) {
            throw new IllegalArgumentException("classes must be at least 1, got " + classes);
        }
        if (height < 4 || width < 4) {
            throw new IllegalArgumentException("height and width must be at least 4, got "
                    + height + "x" + width);
        }
        if (height % 2 != 0 || width % 2 != 0) {
            throw new IllegalArgumentException("height and width must be even for 2x2 pooling, got "
                    + height + "x" + width);
        }
        int flattened = 8 * (height / 2) * (width / 2);
        return engine.<BCHW>sequential()
                .add(engine.conv2d(channels, 8, 3, 1, 1))
                .add(engine.activation(Activation.RELU))
                .add(new MaxPool2d(2))
                .add(new Flatten())
                .add(engine.linear(flattened, classes));
    }
}
