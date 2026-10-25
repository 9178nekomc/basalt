package jsl.nn;

import jsl.core.BC;
import jsl.core.BSF;
import jsl.core.ModelFormatException;
import jsl.core.ModelKind;
import jsl.core.ModelReader;
import jsl.core.ModelWriter;
import jsl.core.Persistable;
import jsl.core.Tensor;
import jsl.core.VectorMath;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * End-to-end trainable sequence classifier: {@code BSF -> [recurrent/attention layers] -> pool ->
 * Dense head -> BC}.
 *
 * <p>Training performs a full forward pass, the loss gradient, backpropagation through the head,
 * the pooling layer and every body layer in reverse order, global-norm gradient clipping, and one
 * optimizer step per parameter. Mini-batching slices the batch axis without touching the time or
 * feature axes.</p>
 */
public final class SequenceNetwork implements Persistable {
    private final List<Layer<BSF, BSF>> body;
    private final SequencePool pool;
    private final Dense head;
    private final Loss loss;
    private final OptimizerConfig optimizer;
    private final double gradientClip;
    private LossEvaluation cachedEvaluation;

    private SequenceNetwork(Builder builder) {
        body = List.copyOf(builder.body);
        pool = SequencePool.of(builder.pooling);
        head = builder.head;
        loss = builder.loss;
        optimizer = builder.optimizer;
        gradientClip = builder.gradientClip;
    }

    public static Builder builder() {
        return new Builder();
    }

    public Tensor<BC> predict(Tensor<BSF> features) {
        return forward(features);
    }

    public Tensor<BC> forward(Tensor<BSF> features) {
        Objects.requireNonNull(features, "features");
        Tensor<BSF> current = features;
        for (Layer<BSF, BSF> layer : body) {
            current = layer.forward(current);
        }
        return head.forward(pool.forward(current));
    }

    public LossEvaluation evaluate(Tensor<BSF> features, Tensor<BC> targets) {
        Tensor<BC> prediction = forward(features);
        cachedEvaluation = loss.evaluate(prediction, targets);
        return cachedEvaluation;
    }

    /** Backpropagates the cached loss gradient; returns the gradient with respect to the input. */
    public Tensor<BSF> backward(LossEvaluation evaluation) {
        if (evaluation != cachedEvaluation) {
            throw new IllegalArgumentException("Evaluation does not belong to the latest forward pass");
        }
        Tensor<BC> headGradient;
        if (evaluation.hasSoftmaxLogitGradient() && head.activation() == Activation.SOFTMAX) {
            headGradient = head.backwardFromSoftmaxLoss(evaluation.softmaxLogitGradient());
        } else {
            headGradient = head.backward(evaluation.gradient());
        }
        Tensor<BSF> current = pool.backward(headGradient);
        for (int index = body.size() - 1; index >= 0; index--) {
            current = body.get(index).backward(current);
        }
        return current;
    }

    /** Applies global-norm clipping and one optimizer step; returns the pre-clipping norm. */
    public double update() {
        List<ParameterRef> parameters = parameters();
        double sumSquares = 0.0;
        for (ParameterRef parameter : parameters) {
            sumSquares += VectorMath.sumSquares(parameter.gradient());
        }
        double norm = Math.sqrt(sumSquares);
        if (norm > gradientClip && norm > 0.0) {
            double factor = gradientClip / norm;
            for (ParameterRef parameter : parameters) {
                VectorMath.scaleInPlace(parameter.gradient(), factor);
            }
        }
        for (Layer<BSF, BSF> layer : body) {
            layer.update(optimizer);
        }
        head.update(optimizer);
        return norm;
    }

    public TrainingHistory fit(Tensor<BSF> features, Tensor<BC> targets, int epochs) {
        return fit(features, targets, epochs, features.dimension(0));
    }

    public TrainingHistory fit(Tensor<BSF> features, Tensor<BC> targets, int epochs, int batchSize) {
        Objects.requireNonNull(features, "features");
        Objects.requireNonNull(targets, "targets");
        if (epochs <= 0 || batchSize <= 0) {
            throw new IllegalArgumentException("epochs and batchSize must be positive");
        }
        int rows = features.dimension(0);
        if (targets.dimension(0) != rows) {
            throw new IllegalArgumentException("Feature and target batches must match");
        }
        double[] history = new double[epochs];
        for (int epoch = 0; epoch < epochs; epoch++) {
            double weightedLoss = 0.0;
            for (int from = 0; from < rows; from += batchSize) {
                int to = Math.min(rows, from + batchSize);
                Tensor<BSF> featureBatch = to - from == rows
                        ? features : features.sliceBatch(from, to);
                Tensor<BC> targetBatch = to - from == rows
                        ? targets : targets.sliceBatch(from, to);
                LossEvaluation evaluation = evaluate(featureBatch, targetBatch);
                if (!Double.isFinite(evaluation.loss())) {
                    throw new IllegalStateException(
                            "Sequence loss became non-finite at epoch " + epoch);
                }
                weightedLoss += evaluation.loss() * (to - from);
                backward(evaluation);
                update();
            }
            history[epoch] = weightedLoss / rows;
        }
        return new TrainingHistory(history);
    }

    public List<ParameterRef> parameters() {
        List<ParameterRef> result = new ArrayList<>();
        for (Layer<BSF, BSF> layer : body) {
            result.addAll(layer.parameters());
        }
        result.addAll(head.parameters());
        return result;
    }

    public Loss loss() {
        return loss;
    }

    public OptimizerConfig optimizer() {
        return optimizer;
    }

    public SequencePool.Pooling pooling() {
        return pool.pooling();
    }

    public int bodySize() {
        return body.size();
    }

    @Override
    public ModelKind kind() {
        return ModelKind.SEQUENCE_NETWORK;
    }

    @Override
    public void write(ModelWriter writer) {
        writer.putInt(body.size());
        for (Layer<BSF, BSF> layer : body) {
            if (layer instanceof LSTM lstm) {
                writer.putByte(1);
                lstm.write(writer);
            } else if (layer instanceof GRU gru) {
                writer.putByte(2);
                gru.write(writer);
            } else if (layer instanceof VanillaRNN rnn) {
                writer.putByte(3);
                rnn.write(writer);
            } else if (layer instanceof SelfAttention attention) {
                writer.putByte(4);
                attention.write(writer);
            } else {
                throw new IllegalStateException(
                        "Unsupported sequence layer " + layer.getClass().getName());
            }
        }
        pool.write(writer);
        head.write(writer);
        writer.putInt(loss.ordinal());
        writer.putInt(optimizer.optimizer().ordinal());
        writer.putDouble(optimizer.learningRate());
        writer.putDouble(optimizer.beta1());
        writer.putDouble(optimizer.beta2());
        writer.putDouble(optimizer.epsilon());
        writer.putDouble(gradientClip);
    }

    public static SequenceNetwork read(ModelReader reader) {
        int bodySize = reader.getInt();
        if (bodySize < 0 || bodySize > 1_000) {
            throw new ModelFormatException("Invalid sequence body size " + bodySize);
        }
        Builder builder = new Builder();
        for (int index = 0; index < bodySize; index++) {
            int type = reader.getByte();
            switch (type) {
                case 1 -> builder.add(LSTM.read(reader));
                case 2 -> builder.add(GRU.read(reader));
                case 3 -> builder.add(VanillaRNN.read(reader));
                case 4 -> builder.add(SelfAttention.read(reader));
                default -> throw new ModelFormatException("Unknown sequence layer " + type);
            }
        }
        int poolingOrdinal = reader.getInt();
        SequencePool.Pooling[] poolings = SequencePool.Pooling.values();
        if (poolingOrdinal < 0 || poolingOrdinal >= poolings.length) {
            throw new ModelFormatException("Invalid pooling ordinal " + poolingOrdinal);
        }
        builder.pooling(poolings[poolingOrdinal]);
        builder.head(Dense.read(reader));
        int lossOrdinal = reader.getInt();
        Loss[] losses = Loss.values();
        if (lossOrdinal < 0 || lossOrdinal >= losses.length) {
            throw new ModelFormatException("Invalid loss ordinal " + lossOrdinal);
        }
        builder.loss(losses[lossOrdinal]);
        int optimizerOrdinal = reader.getInt();
        Optimizer[] optimizers = Optimizer.values();
        if (optimizerOrdinal < 0 || optimizerOrdinal >= optimizers.length) {
            throw new ModelFormatException("Invalid optimizer ordinal " + optimizerOrdinal);
        }
        builder.optimizer(new OptimizerConfig(optimizers[optimizerOrdinal],
                reader.getDouble(), reader.getDouble(), reader.getDouble(), reader.getDouble()));
        builder.gradientClip(reader.getDouble());
        return builder.build();
    }

    public static final class Builder {
        private final List<Layer<BSF, BSF>> body = new ArrayList<>();
        private SequencePool.Pooling pooling = SequencePool.Pooling.LAST;
        private Dense head;
        private Loss loss = Loss.CATEGORICAL_CROSSENTROPY;
        private OptimizerConfig optimizer = Optimizer.ADAM.withLearningRate(0.001);
        private double gradientClip = 5.0;

        private Builder() {
        }

        public <L extends Layer<BSF, BSF>> Builder add(L layer) {
            body.add(Objects.requireNonNull(layer, "layer"));
            return this;
        }

        public Builder pooling(SequencePool.Pooling pooling) {
            this.pooling = Objects.requireNonNull(pooling, "pooling");
            return this;
        }

        public Builder head(Dense head) {
            this.head = Objects.requireNonNull(head, "head");
            return this;
        }

        public Builder loss(Loss loss) {
            this.loss = Objects.requireNonNull(loss, "loss");
            return this;
        }

        public Builder optimizer(OptimizerConfig optimizer) {
            this.optimizer = Objects.requireNonNull(optimizer, "optimizer");
            return this;
        }

        public Builder gradientClip(double gradientClip) {
            this.gradientClip = gradientClip;
            return this;
        }

        public SequenceNetwork build() {
            if (head == null) {
                throw new IllegalStateException("A Dense head is required");
            }
            if (!(gradientClip > 0.0)) {
                throw new IllegalArgumentException("gradientClip must be positive");
            }
            if (loss == Loss.CATEGORICAL_CROSSENTROPY && head.activation() != Activation.SOFTMAX) {
                throw new IllegalArgumentException(
                        "Categorical cross-entropy requires a SOFTMAX head");
            }
            if (loss == Loss.BINARY_CROSSENTROPY && head.activation() != Activation.SIGMOID) {
                throw new IllegalArgumentException(
                        "Binary cross-entropy requires a SIGMOID head");
            }
            if (loss == Loss.CATEGORICAL_CROSSENTROPY && head.units() < 2) {
                throw new IllegalArgumentException("Categorical head needs at least two units");
            }
            return new SequenceNetwork(this);
        }
    }
}
