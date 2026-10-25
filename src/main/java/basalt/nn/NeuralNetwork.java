package basalt.nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Staged CNN builder. Image layers are legal before Flatten and vector layers after it;
 * invalid dimension transitions therefore fail during compilation rather than training.
 */
public final class NeuralNetwork {
    private NeuralNetwork() {
    }

    public static ImageBuilder builder() {
        return new ImageBuilder();
    }

    public static final class ImageBuilder {
        private final List<Layer<?, ?>> layers = new ArrayList<>();

        private ImageBuilder() {
        }

        public ImageBuilder add(Conv2D layer) {
            layers.add(Objects.requireNonNull(layer, "layer"));
            return this;
        }

        public ImageBuilder add(MaxPooling2D layer) {
            layers.add(Objects.requireNonNull(layer, "layer"));
            return this;
        }

        public VectorBuilder add(Flatten layer) {
            layers.add(Objects.requireNonNull(layer, "layer"));
            return new VectorBuilder(layers);
        }
    }

    public static final class VectorBuilder {
        private final List<Layer<?, ?>> layers;
        private Loss loss = Loss.CATEGORICAL_CROSSENTROPY;
        private OptimizerConfig optimizer = Optimizer.ADAM.withLearningRate(0.001);

        private VectorBuilder(List<Layer<?, ?>> layers) {
            this.layers = new ArrayList<>(layers);
        }

        public VectorBuilder add(Dense layer) {
            layers.add(Objects.requireNonNull(layer, "layer"));
            return this;
        }

        public VectorBuilder loss(Loss loss) {
            this.loss = Objects.requireNonNull(loss, "loss");
            return this;
        }

        public VectorBuilder optimizer(OptimizerConfig optimizer) {
            this.optimizer = Objects.requireNonNull(optimizer, "optimizer");
            return this;
        }

        public CNN build() {
            return new CNN(layers, loss, optimizer);
        }
    }
}
