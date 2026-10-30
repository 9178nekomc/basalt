package basalt.model;

import basalt.Basalt;
import basalt.core.BC;
import basalt.core.Matrix;
import basalt.ml.DataSplit;
import basalt.nn.Activation;
import basalt.torch.Loss;
import basalt.torch.Module;
import basalt.torch.Optimizer;
import basalt.torch.Sequential;
import basalt.torch.Tensor;
import basalt.zoo.Trainer;
import basalt.zoo.Zoo;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A neural candidate: an architecture plus the training recipe that turns it into a model.
 *
 * <p>Two shapes are supported, matching the two data forms. Tabular candidates read raw feature
 * columns and emit one class or one value per row; sequence candidates read a window of token ids
 * and emit a distribution over the next one.</p>
 *
 * <p>Scoring holds out a slice and trains once, where {@link ClassicalSolver} cross-validates. The
 * asymmetry is deliberate: a five-fold sweep of a network costs five trainings, and for the purpose
 * of ranking candidates a single honest holdout is enough. The price is a noisier estimate on very
 * small datasets, which is why the screening table reports the numbers rather than hiding them.</p>
 */
public final class NeuralSolver implements Solver {

    private static final float HOLDOUT = 0.25f;
    private static final float SEQUENCE_HOLDOUT = 0.2f;

    private final String name;
    private final Task task;
    private final int[] hidden;
    private final int window;
    private final int dimension;
    private final int residualWidth;
    private final int epochs;
    private final int batchSize;
    private final float learningRate;

    private NeuralSolver(String name, Task task, int[] hidden, int window, int dimension,
                         int residualWidth, int epochs, int batchSize, float learningRate) {
        this.name = name;
        this.task = task;
        this.hidden = hidden;
        this.window = window;
        this.dimension = dimension;
        this.residualWidth = residualWidth;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.learningRate = learningRate;
    }

    /**
     * Dense candidate for tabular data.
     *
     * @param task   {@code TABULAR_CLASSIFICATION} or {@code TABULAR_REGRESSION}
     * @param hidden widths of the hidden layers, in order; may be empty for a linear model
     */
    public static NeuralSolver tabular(String name, Task task, int[] hidden, int epochs,
                                       int batchSize, float learningRate) {
        validateCommon(name, epochs, batchSize, learningRate);
        if (task == null || !task.isTabular()) {
            throw new IllegalArgumentException("tabular candidates need a tabular task, got " + task);
        }
        if (hidden == null) {
            throw new IllegalArgumentException("hidden must not be null; use an empty array");
        }
        for (int width : hidden) {
            if (width < 1) {
                throw new IllegalArgumentException("hidden width must be at least 1");
            }
        }
        return new NeuralSolver(name, task, hidden.clone(), 0, 0, 0,
                epochs, batchSize, learningRate);
    }

    /**
     * Windowed language-model candidate for token streams.
     *
     * <p>Reports {@link Task#SEQUENCE_LM} regardless of {@code name}, so it is only ever screened
     * against token data.</p>
     *
     * <p>Be careful reading the score on a tiny corpus. The estimate is cross entropy over windows
     * whose target was held out of training, and a few dozen tokens cannot support that: the model
     * memorises what it sees and is confidently wrong on the tail, so the reported number can sit
     * above {@code ln(vocabulary)}, i.e. worse than guessing uniformly. That is a measurement
     * artefact, not a broken model — check {@code generate(...)} to see what it actually learned.
     * Holdout scoring is meant for corpora large enough that the tail resembles the body.</p>
     */
    public static NeuralSolver sequence(String name, int window, int dimension, int residualWidth,
                                        int epochs, int batchSize, float learningRate) {
        validateCommon(name, epochs, batchSize, learningRate);
        if (window < 1 || dimension < 1 || residualWidth < 1) {
            throw new IllegalArgumentException("window, dimension and residualWidth must be >= 1");
        }
        return new NeuralSolver(name, Task.SEQUENCE_LM, new int[0], window, dimension,
                residualWidth, epochs, batchSize, learningRate);
    }

    private static void validateCommon(String name, int epochs, int batchSize, float learningRate) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("name must not be blank");
        }
        if (epochs < 1) {
            throw new IllegalArgumentException("epochs must be at least 1, got " + epochs);
        }
        if (batchSize < 1) {
            throw new IllegalArgumentException("batchSize must be at least 1, got " + batchSize);
        }
        if (!(learningRate > 0f)) {
            throw new IllegalArgumentException("learningRate must be positive, got " + learningRate);
        }
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public Task task() {
        return task;
    }

    @Override
    public double score(Dataset data, int folds, long seed) {
        requireTask(data);
        try (Basalt engine = Basalt.create()) {
            return task.isTabular()
                    ? scoreTabular(engine, data, seed)
                    : scoreSequence(engine, data, seed);
        }
    }

    @Override
    public Fitted fit(Dataset data, long seed) {
        requireTask(data);
        Basalt engine = Basalt.create();
        try {
            if (task.isTabular()) {
                Labels labels = task == Task.TABULAR_CLASSIFICATION
                        ? Labels.of(data.targets()) : null;
                int outputs = labels == null ? 1 : labels.size();
                Sequential<BC, BC> network = tabularModule(engine, data.features().columns(),
                        outputs);
                trainTabular(engine, network, data.features(), data.targets(), labels, seed,
                        epochs);
                return new NeuralFitted(engine, network, labels, task, 0, 0);
            }
            int vocabulary = data.vocabulary();
            Module<BC, BC> network = Zoo.charLm(engine, vocabulary, window, dimension,
                    residualWidth);
            int[] tokens = data.tokens();
            trainSequence(engine, network, tokens, seed, epochs, vocabulary);
            return new NeuralFitted(engine, network, null, task, window, vocabulary);
        } catch (RuntimeException failure) {
            engine.close();
            throw failure;
        }
    }

    private void requireTask(Dataset data) {
        if (data.task() != task) {
            throw new IllegalArgumentException("solver '" + name + "' solves " + task
                    + " but the dataset poses " + data.task());
        }
    }

    private double scoreTabular(Basalt engine, Dataset data, long seed) {
        Matrix features = data.features();
        double[] targets = data.targets();
        Labels labels = task == Task.TABULAR_CLASSIFICATION ? Labels.of(targets) : null;
        DataSplit.Holdout split = task == Task.TABULAR_CLASSIFICATION
                ? DataSplit.trainTestSplit(features, targets, HOLDOUT, true, true, seed)
                : DataSplit.holdout(features, targets, HOLDOUT, seed);
        if (split.testFeatures().rows() == 0) {
            // too small to split; report the training fit rather than nothing
            split = new DataSplit.Holdout(features, targets, features, targets);
        }
        int outputs = labels == null ? 1 : labels.size();
        Sequential<BC, BC> network = tabularModule(engine, features.columns(), outputs);
        try {
            trainTabular(engine, network, split.trainFeatures(), split.trainTargets(), labels,
                    seed, epochs);
            double[] prediction = predictTabular(engine, network, labels, split.testFeatures());
            return task.scoreOf(task.defaultMetric(), split.testTargets(), prediction);
        } finally {
            network.close();
        }
    }

    private double scoreSequence(Basalt engine, Dataset data, long seed) {
        int[] tokens = data.tokens();
        int vocabulary = data.vocabulary();
        if (tokens.length < window + 2) {
            throw new IllegalArgumentException("need at least " + (window + 2)
                    + " tokens for window " + window + ", got " + tokens.length);
        }
        int cut = Math.max(window + 1, (int) Math.floor(tokens.length * (1 - SEQUENCE_HOLDOUT)));
        cut = Math.min(cut, tokens.length - 1);
        Module<BC, BC> network = Zoo.charLm(engine, vocabulary, window, dimension, residualWidth);
        try {
            trainSequenceRange(engine, network, tokens, 0, cut, seed, epochs, vocabulary);
            double loss = sequenceLoss(engine, network, tokens, cut, vocabulary);
            return -loss;
        } finally {
            network.close();
        }
    }

    private Sequential<BC, BC> tabularModule(Basalt engine, int columns, int outputs) {
        Sequential<BC, BC> network = engine.<BC>sequential();
        int previous = columns;
        for (int width : hidden) {
            network = network.add(engine.linear(previous, width));
            network = network.add(engine.activation(Activation.RELU));
            previous = width;
        }
        return network.add(engine.linear(previous, outputs));
    }

    private void trainTabular(Basalt engine, Sequential<BC, BC> network, Matrix features,
                              double[] targets, Labels labels, long seed, int epochs) {
        int rows = features.rows();
        int columns = features.columns();
        float[] inputs = new float[rows * columns];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                inputs[row * columns + column] = (float) features.get(row, column);
            }
        }
        Tensor featureTensor = engine.tensor(inputs, rows, columns);
        Loss criterion = labels == null ? engine.mse() : engine.crossEntropy();
        Tensor targetTensor;
        if (labels == null) {
            float[] values = new float[rows];
            for (int row = 0; row < rows; row++) {
                values[row] = (float) targets[row];
            }
            targetTensor = engine.tensor(values, rows, 1);
        } else {
            float[] values = new float[rows];
            for (int row = 0; row < rows; row++) {
                values[row] = labels.indexOf(targets[row]);
            }
            targetTensor = engine.tensor(values, rows);
        }
        Optimizer optimizer = engine.adamw(network.parameters(), learningRate);
        Trainer.of(engine.context(), network, optimizer, criterion)
                .epochs(epochs)
                .batchSize(Math.min(batchSize, rows))
                .seed(seed)
                .fit(featureTensor, targetTensor);
    }

    private double[] predictTabular(Basalt engine, Sequential<BC, BC> network, Labels labels,
                                    Matrix features) {
        int rows = features.rows();
        int columns = features.columns();
        float[] inputs = new float[rows * columns];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                inputs[row * columns + column] = (float) features.get(row, column);
            }
        }
        float[] output = network.forward(engine.tensor(inputs, rows, columns)).data();
        double[] prediction = new double[rows];
        if (labels == null) {
            for (int row = 0; row < rows; row++) {
                prediction[row] = output[row];
            }
            return prediction;
        }
        int classes = labels.size();
        for (int row = 0; row < rows; row++) {
            int best = 0;
            for (int candidate = 1; candidate < classes; candidate++) {
                if (output[row * classes + candidate] > output[row * classes + best]) {
                    best = candidate;
                }
            }
            prediction[row] = labels.value(best);
        }
        return prediction;
    }

    private void trainSequence(Basalt engine, Module<BC, BC> network, int[] tokens, long seed,
                               int epochs, int vocabulary) {
        trainSequenceRange(engine, network, tokens, 0, tokens.length, seed, epochs, vocabulary);
    }

    /** Trains on every window whose target index falls in {@code [from, to)}. */
    private void trainSequenceRange(Basalt engine, Module<BC, BC> network, int[] tokens,
                                    int from, int to, long seed, int epochs, int vocabulary) {
        List<Integer> starts = new ArrayList<>();
        for (int start = 0; start + window < tokens.length; start++) {
            int targetIndex = start + window;
            if (targetIndex >= from && targetIndex < to) {
                starts.add(start);
            }
        }
        if (starts.isEmpty()) {
            throw new IllegalArgumentException("no training windows in range [" + from + ", "
                    + to + ") for window " + window);
        }
        int samples = starts.size();
        float[] windows = new float[samples * window];
        float[] targets = new float[samples];
        for (int index = 0; index < samples; index++) {
            int start = starts.get(index);
            for (int offset = 0; offset < window; offset++) {
                windows[index * window + offset] = tokens[start + offset];
            }
            targets[index] = tokens[start + window];
        }
        Optimizer optimizer = engine.adamw(network.parameters(), learningRate);
        Trainer.of(engine.context(), network, optimizer, engine.crossEntropy())
                .epochs(epochs)
                .batchSize(Math.min(batchSize, samples))
                .seed(seed)
                .fit(engine.tensor(windows, samples, window), engine.tensor(targets, samples));
    }

    /** Mean cross entropy over every window whose target index is at least {@code from}. */
    private double sequenceLoss(Basalt engine, Module<BC, BC> network, int[] tokens, int from,
                                int vocabulary) {
        List<Integer> starts = new ArrayList<>();
        for (int start = 0; start + window < tokens.length; start++) {
            if (start + window >= from) {
                starts.add(start);
            }
        }
        if (starts.isEmpty()) {
            return Double.NaN;
        }
        int samples = starts.size();
        float[] windows = new float[samples * window];
        float[] targets = new float[samples];
        for (int index = 0; index < samples; index++) {
            int start = starts.get(index);
            for (int offset = 0; offset < window; offset++) {
                windows[index * window + offset] = tokens[start + offset];
            }
            targets[index] = tokens[start + window];
        }
        Tensor logits = network.forward(engine.tensor(windows, samples, window));
        Tensor loss = engine.crossEntropy().loss(logits, engine.tensor(targets, samples));
        double value = loss.data()[0];
        engine.clearGraph();
        return value;
    }

    @Override
    public String toString() {
        return "NeuralSolver[" + name + ", " + task + "]";
    }

    /** Sorted distinct target values, so classes become dense indices 0..n-1. */
    static final class Labels {
        private final double[] values;
        private final Map<Double, Integer> index;

        private Labels(double[] values) {
            this.values = values;
            this.index = new LinkedHashMap<>();
            for (int i = 0; i < values.length; i++) {
                index.put(values[i], i);
            }
        }

        static Labels of(double[] targets) {
            double[] sorted = Arrays.stream(targets).distinct().sorted().toArray();
            return new Labels(sorted);
        }

        int size() {
            return values.length;
        }

        int indexOf(double value) {
            Integer found = index.get(value);
            if (found == null) {
                throw new IllegalArgumentException("unknown class " + value);
            }
            return found;
        }

        double value(int classIndex) {
            return values[classIndex];
        }
    }

    /** Trained network; owns the runtime that holds its native tensors. */
    private static final class NeuralFitted implements Fitted {
        private final Basalt engine;
        private final Module<BC, BC> network;
        private final Labels labels;
        private final Task task;
        private final int window;
        private final int vocabulary;

        NeuralFitted(Basalt engine, Module<BC, BC> network, Labels labels, Task task, int window,
                     int vocabulary) {
            this.engine = engine;
            this.network = network;
            this.labels = labels;
            this.task = task;
            this.window = window;
            this.vocabulary = vocabulary;
        }

        @Override
        public double[] predict(Matrix features) {
            if (task.isTabular() == false) {
                throw new UnsupportedOperationException("a " + task + " model does not predict rows");
            }
            int rows = features.rows();
            int columns = features.columns();
            float[] inputs = new float[rows * columns];
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    inputs[row * columns + column] = (float) features.get(row, column);
                }
            }
            float[] output = network.forward(engine.tensor(inputs, rows, columns)).data();
            double[] prediction = new double[rows];
            if (labels == null) {
                for (int row = 0; row < rows; row++) {
                    prediction[row] = output[row];
                }
                return prediction;
            }
            int classes = labels.size();
            for (int row = 0; row < rows; row++) {
                int best = 0;
                for (int candidate = 1; candidate < classes; candidate++) {
                    if (output[row * classes + candidate] > output[row * classes + best]) {
                        best = candidate;
                    }
                }
                prediction[row] = labels.value(best);
            }
            return prediction;
        }

        @Override
        public int[] generate(int[] seed, int steps) {
            if (task != Task.SEQUENCE_LM) {
                throw new UnsupportedOperationException("a " + task + " model does not generate");
            }
            if (seed == null || seed.length < window) {
                throw new IllegalArgumentException("seed must hold at least " + window + " tokens");
            }
            if (steps < 1) {
                throw new IllegalArgumentException("steps must be at least 1");
            }
            int[] context = new int[window];
            System.arraycopy(seed, seed.length - window, context, 0, window);
            int[] produced = new int[steps];
            for (int step = 0; step < steps; step++) {
                float[] row = new float[window];
                for (int offset = 0; offset < window; offset++) {
                    row[offset] = context[offset];
                }
                float[] scores = network.forward(engine.tensor(row, 1, window)).data();
                int best = 0;
                for (int candidate = 1; candidate < scores.length; candidate++) {
                    if (scores[candidate] > scores[best]) {
                        best = candidate;
                    }
                }
                produced[step] = best;
                System.arraycopy(context, 1, context, 0, window - 1);
                context[window - 1] = best;
            }
            return produced;
        }

        @Override
        public Task task() {
            return task;
        }

        @Override
        public void close() {
            network.close();
            engine.close();
        }
    }
}
