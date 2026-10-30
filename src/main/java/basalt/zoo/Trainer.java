package basalt.zoo;

import basalt.BasaltRuntimeException;
import basalt.nativebridge.NativeContext;
import basalt.torch.Loss;
import basalt.torch.Module;
import basalt.torch.Optimizer;
import basalt.torch.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * The training loop, written once.
 *
 * <p>This is the middle tier's counterpart to {@link Auto}: you still choose the architecture, the
 * optimiser and the loss, but you do not rewrite the batch loop, the graph reset or the epoch
 * bookkeeping. Every step is exactly what a hand-written loop would do — forward, loss, backward,
 * optimiser step, zero the gradients, clear the graph — so a model trained here behaves identically
 * to one trained by hand.</p>
 *
 * <pre>{@code
 * try (Basalt engine = Basalt.create()) {
 *     Sequential<BC, BC> net = Zoo.tabularClassifier(engine, 4, 3);
 *     History history = Trainer.of(engine.context(), net, engine.adamw(net.parameters(), 1e-2f),
 *                     engine.crossEntropy())
 *             .epochs(50)
 *             .batchSize(16)
 *             .fit(engine.tensor(features, rows, 4), engine.tensor(labels, rows));
 * }
 * }</pre>
 *
 * <p>Batches are copied into fresh tensors rather than sliced in place. That costs one array copy
 * per batch and keeps the graph free of views, which is the same trade the rest of the tensor
 * layer makes: no hidden aliasing, no lifetime puzzles.</p>
 */
public final class Trainer {

    private final NativeContext context;
    private final Module<?, ?> module;
    private final Optimizer optimizer;
    private final Loss loss;
    private int epochs = 10;
    private int batchSize = 32;
    private long seed = 42L;
    private boolean shuffle = true;
    private int patience;

    private Trainer(NativeContext context, Module<?, ?> module, Optimizer optimizer, Loss loss) {
        this.context = context;
        this.module = module;
        this.optimizer = optimizer;
        this.loss = loss;
    }

    /**
     * Binds a model, its optimiser and its loss.
     *
     * @throws IllegalArgumentException if any argument is null, or the module exposes no parameters
     */
    public static Trainer of(NativeContext context, Module<?, ?> module, Optimizer optimizer,
                             Loss loss) {
        BasaltRuntimeException.requireNonNull(context, "context");
        BasaltRuntimeException.requireNonNull(module, "module");
        BasaltRuntimeException.requireNonNull(optimizer, "optimizer");
        BasaltRuntimeException.requireNonNull(loss, "loss");
        if (module.parameters().isEmpty()) {
            throw new IllegalArgumentException("module exposes no parameters; nothing to train");
        }
        return new Trainer(context, module, optimizer, loss);
    }

    /** Number of passes over the data. */
    public Trainer epochs(int epochs) {
        if (epochs < 1) {
            throw new IllegalArgumentException("epochs must be at least 1, got " + epochs);
        }
        this.epochs = epochs;
        return this;
    }

    /** Rows per step; the last batch of an epoch may be shorter. */
    public Trainer batchSize(int batchSize) {
        if (batchSize < 1) {
            throw new IllegalArgumentException("batchSize must be at least 1, got " + batchSize);
        }
        this.batchSize = batchSize;
        return this;
    }

    /** Seed for the per-epoch shuffle. */
    public Trainer seed(long seed) {
        this.seed = seed;
        return this;
    }

    /** Whether to reshuffle the rows at the start of every epoch. */
    public Trainer shuffle(boolean shuffle) {
        this.shuffle = shuffle;
        return this;
    }

    /**
     * Stops when the epoch loss has not improved for this many consecutive epochs.
     * {@code 0} (the default) disables early stopping.
     */
    public Trainer earlyStopping(int patience) {
        if (patience < 0) {
            throw new IllegalArgumentException("patience must not be negative");
        }
        this.patience = patience;
        return this;
    }

    /**
     * Trains on the whole tensor.
     *
     * @param features rank 2 or higher; the first dimension is the sample axis
     * @param targets  rank 1 (class indices, or one value per row) or rank 2 matching features
     * @throws IllegalArgumentException on shape mismatch or empty input
     * @throws IllegalStateException    if an epoch loss stops being finite
     */
    public History fit(Tensor features, Tensor targets) {
        BasaltRuntimeException.requireNonNull(features, "features");
        BasaltRuntimeException.requireNonNull(targets, "targets");
        if (features.rank() < 2) {
            throw new IllegalArgumentException("features must have rank >= 2, got rank "
                    + features.rank());
        }
        if (targets.rank() < 1 || targets.rank() > 2) {
            throw new IllegalArgumentException("targets must have rank 1 or 2, got rank "
                    + targets.rank());
        }
        int rows = features.shape()[0];
        if (rows < 1) {
            throw new IllegalArgumentException("features is empty");
        }
        if (targets.shape()[0] != rows) {
            throw new IllegalArgumentException("features has " + rows + " rows but targets has "
                    + targets.shape()[0]);
        }

        int featureWidth = features.size() / rows;
        int targetWidth = targets.size() / rows;
        float[] featureData = features.data();
        float[] targetData = targets.data();

        int[] order = new int[rows];
        for (int row = 0; row < rows; row++) {
            order[row] = row;
        }
        Random random = new Random(seed);
        List<Double> history = new ArrayList<>(epochs);
        module.train();

        double best = Double.NaN;
        int sinceImprovement = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            if (shuffle) {
                for (int index = rows - 1; index > 0; index--) {
                    int swap = random.nextInt(index + 1);
                    int temporary = order[index];
                    order[index] = order[swap];
                    order[swap] = temporary;
                }
            }
            double total = 0.0;
            int batches = 0;
            for (int from = 0; from < rows; from += batchSize) {
                int to = Math.min(from + batchSize, rows);
                int count = to - from;
                Tensor batchFeatures = Tensor.of(context,
                        gather(featureData, order, from, count, featureWidth),
                        prepend(count, features.shape()));
                Tensor batchTargets = Tensor.of(context,
                        gather(targetData, order, from, count, targetWidth),
                        prepend(count, targets.shape()));
                Tensor prediction = module.forward(batchFeatures);
                Tensor value = loss.loss(prediction, batchTargets);
                value.backward();
                optimizer.step();
                optimizer.zeroGrad();
                context.clearGraph();
                total += value.data()[0];
                batches++;
            }
            double mean = batches == 0 ? Double.NaN : total / batches;
            if (!Double.isFinite(mean)) {
                throw new IllegalStateException("loss became " + mean + " at epoch " + epoch
                        + "; lower the learning rate or check the input range");
            }
            history.add(mean);
            if (Double.isNaN(best) || mean < best - 1e-12) {
                best = mean;
                sinceImprovement = 0;
            } else if (patience > 0 && ++sinceImprovement >= patience) {
                break;
            }
        }
        return new History(history);
    }

    private static int[] prepend(int count, int[] shape) {
        int[] result = new int[shape.length];
        result[0] = count;
        System.arraycopy(shape, 1, result, 1, shape.length - 1);
        return result;
    }

    private static float[] gather(float[] data, int[] order, int from, int count, int width) {
        float[] slice = new float[count * width];
        for (int index = 0; index < count; index++) {
            int row = order[from + index];
            System.arraycopy(data, row * width, slice, index * width, width);
        }
        return slice;
    }
}
