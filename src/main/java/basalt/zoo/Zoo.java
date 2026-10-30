package basalt.zoo;

import basalt.Basalt;
import basalt.BasaltRuntimeException;
import basalt.core.BC;
import basalt.core.BCHW;
import basalt.nativebridge.NativeContext;
import basalt.nn.Activation;
import basalt.torch.Embedding;
import basalt.torch.Flatten;
import basalt.torch.Linear;
import basalt.torch.MaxPool2d;
import basalt.torch.Module;
import basalt.torch.Sequential;
import basalt.torch.Tensor;

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
     * Windowed character-level language model.
     *
     * <p>Takes {@code window} token ids, embeds them, flattens to one vector, passes it through a
     * bottleneck residual block and projects to the vocabulary. Small enough to memorise a short
     * corpus in seconds on a CPU, structured enough to be a real next-token predictor.</p>
     *
     * <p>The returned module is an ordinary {@link Module}: train it with {@link Trainer}, inspect
     * it with {@code parameters()}, or wrap it further.</p>
     */
    public static Module<BC, BC> charLm(Basalt engine, int vocabulary, int window, int dimension,
                                        int hidden) {
        BasaltRuntimeException.requireNonNull(engine, "engine");
        if (vocabulary < 2) {
            throw new IllegalArgumentException("vocabulary must be at least 2, got " + vocabulary);
        }
        if (window < 1) {
            throw new IllegalArgumentException("window must be at least 1, got " + window);
        }
        if (dimension < 1 || hidden < 1) {
            throw new IllegalArgumentException("dimension and hidden must be at least 1, got "
                    + dimension + " and " + hidden);
        }
        return new WindowedLanguageModel(engine.context(), vocabulary, window, dimension, hidden);
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

    /**
     * The module behind {@link #charLm}: embedding → flatten → bottleneck residual block → head.
     *
     * <p>Kept private so the architecture is free to change without becoming part of the public
     * surface; callers get a {@link Module} and that is all they should depend on.</p>
     */
    private static final class WindowedLanguageModel extends Module<BC, BC> {
        private final Embedding embedding;
        private final Linear expand;
        private final Linear contract;
        private final Linear head;
        private final int window;
        private final int dimension;

        WindowedLanguageModel(NativeContext context, int vocabulary, int window, int dimension,
                              int hidden) {
            this.window = window;
            this.dimension = dimension;
            this.embedding = new Embedding(context, vocabulary, dimension, 17L);
            int flat = window * dimension;
            this.expand = new Linear(context, flat, hidden);
            this.contract = new Linear(context, hidden, flat);
            this.head = new Linear(context, flat, vocabulary);
        }

        @Override
        public Tensor forward(Tensor input) {
            if (input.rank() != 2 || input.shape()[1] != window) {
                throw new IllegalArgumentException("expected [" + input.shape()[0] + ", " + window
                        + "] token ids, got rank " + input.rank() + " shape "
                        + java.util.Arrays.toString(input.shape()));
            }
            int batch = input.shape()[0];
            Tensor embedded = embedding.forward(input);
            Tensor flat = embedded.reshape(batch, window * dimension);
            Tensor hidden = expand.forward(flat).gelu();
            Tensor back = contract.forward(hidden);
            Tensor mixed = back.add(flat).gelu();
            return head.forward(mixed);
        }

        @Override
        public java.util.List<Tensor> parameters() {
            java.util.List<Tensor> all = new java.util.ArrayList<>();
            all.addAll(embedding.parameters());
            all.addAll(expand.parameters());
            all.addAll(contract.parameters());
            all.addAll(head.parameters());
            return all;
        }
    }
}
