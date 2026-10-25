package basalt.nn;

import basalt.core.BC;
import basalt.core.BCHW;
import basalt.core.ModelKind;
import basalt.core.ModelWriter;
import basalt.core.Persistable;
import basalt.core.Tensor;
import java.util.List;
import java.util.Objects;

/** A BCHW-input, BC-output neural network built through a layout-safe staged builder. */
public final class CNN implements Persistable {
    private final List<Layer<?, ?>> layers;
    private final Loss loss;
    private final OptimizerConfig optimizer;

    CNN(List<Layer<?, ?>> layers, Loss loss, OptimizerConfig optimizer) {
        this.layers = List.copyOf(layers);
        this.loss = Objects.requireNonNull(loss, "loss");
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer");
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    public Tensor<BC> forward(Tensor<BCHW> input) {
        Objects.requireNonNull(input, "input");
        Tensor current = input;
        for (Layer layer : layers) {
            current = forwardUnchecked(layer, current);
        }
        return current;
    }

    public Tensor<BC> predict(Tensor<BCHW> input) {
        return forward(input);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    public Tensor<BCHW> backward(Tensor<BC> gradient) {
        Objects.requireNonNull(gradient, "gradient");
        Tensor current = gradient;
        for (int index = layers.size() - 1; index >= 0; index--) {
            current = backwardUnchecked(layers.get(index), current);
        }
        return current;
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private Tensor<BCHW> backward(LossEvaluation evaluation) {
        int lastIndex = layers.size() - 1;
        Tensor current;
        if (evaluation.hasSoftmaxLogitGradient() && lastIndex >= 0
                && layers.get(lastIndex) instanceof Dense dense
                && dense.activation() == Activation.SOFTMAX) {
            current = dense.backwardFromSoftmaxLoss(evaluation.softmaxLogitGradient());
            lastIndex--;
        } else {
            current = evaluation.gradient();
        }
        for (int index = lastIndex; index >= 0; index--) {
            current = backwardUnchecked(layers.get(index), current);
        }
        return current;
    }

    public TrainingHistory fit(Tensor<BCHW> features, Tensor<BC> targets, int epochs) {
        Objects.requireNonNull(features, "features");
        Objects.requireNonNull(targets, "targets");
        if (epochs <= 0) {
            throw new IllegalArgumentException("epochs must be positive");
        }
        if (features.dimension(0) != targets.dimension(0)) {
            throw new IllegalArgumentException("Feature and target batches must match");
        }
        double[] losses = new double[epochs];
        for (int epoch = 0; epoch < epochs; epoch++) {
            Tensor<BC> prediction = forward(features);
            LossEvaluation evaluation = loss.evaluate(prediction, targets);
            if (!Double.isFinite(evaluation.loss())) {
                throw new IllegalStateException("Training loss became non-finite at epoch " + epoch);
            }
            losses[epoch] = evaluation.loss();
            backward(evaluation);
            for (Layer<?, ?> layer : layers) {
                layer.update(optimizer);
            }
        }
        return new TrainingHistory(losses);
    }

    public LossEvaluation evaluate(Tensor<BCHW> features, Tensor<BC> targets) {
        return loss.evaluate(forward(features), targets);
    }

    public int layerCount() {
        return layers.size();
    }

    List<Layer<?, ?>> layers() {
        return layers;
    }

    Loss loss() {
        return loss;
    }

    OptimizerConfig optimizer() {
        return optimizer;
    }

    @Override
    public ModelKind kind() {
        return ModelKind.CNN;
    }

    @Override
    public void write(ModelWriter writer) {
        ModelSerializer.writeCNN(writer, this);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private static Tensor forwardUnchecked(Layer layer, Tensor input) {
        return layer.forward(input);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private static Tensor backwardUnchecked(Layer layer, Tensor gradient) {
        return layer.backward(gradient);
    }
}
