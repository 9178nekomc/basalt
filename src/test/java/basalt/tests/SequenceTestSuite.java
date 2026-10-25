package basalt.tests;

import basalt.core.BC;
import basalt.core.BSF;
import basalt.core.ModelKind;
import basalt.core.Persistable;
import basalt.core.Tensor;
import basalt.io.ModelCodec;
import basalt.nn.Activation;
import basalt.nn.Dense;
import basalt.nn.GRU;
import basalt.nn.GradientCheck;
import basalt.nn.LSTM;
import basalt.nn.Layer;
import basalt.nn.Loss;
import basalt.nn.Optimizer;
import basalt.nn.ParameterRef;
import basalt.nn.SelfAttention;
import basalt.nn.SequenceNetwork;
import basalt.nn.SequencePool;
import basalt.nn.TrainingHistory;
import basalt.nn.VanillaRNN;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

/** BPTT, attention, sequence training and persistence acceptance tests. */
public final class SequenceTestSuite {
    private static final double EPSILON = 1e-6;
    private static final double GRADIENT_TOLERANCE = 1e-6;
    private static int passed;

    private SequenceTestSuite() {
    }

    public static void main(String[] args) throws Exception {
        run("LSTM input and parameter gradients", SequenceTestSuite::testLstmGradients);
        run("GRU input and parameter gradients", SequenceTestSuite::testGruGradients);
        run("VanillaRNN input and parameter gradients", SequenceTestSuite::testRnnGradients);
        run("Self-Attention full and causal gradients", SequenceTestSuite::testAttentionGradients);
        run("causal attention cannot see the future", SequenceTestSuite::testCausalMask);
        run("SequenceNetwork end-to-end gradients", SequenceTestSuite::testNetworkGradients);
        run("LSTM sequence classification convergence", SequenceTestSuite::testSequenceTraining);
        run("mini-batch sequence training", SequenceTestSuite::testMiniBatchTraining);
        run("LSTM preserves long-range gradients", SequenceTestSuite::testGradientPreservation);
        run("sequence .bsl bit-exact serialization", SequenceTestSuite::testSerialization);
        System.out.println("PASS: " + passed + " sequence test groups");
    }

    private static void testLstmGradients() {
        GradientCheck.Report report = GradientCheck.checkLayer(
                LSTM.builder().hiddenSize(3).seed(7).build(), sequenceInput(2, 3, 2, 11),
                sequenceInput(2, 3, 3, 12), EPSILON);
        check(report.passed(GRADIENT_TOLERANCE), "LSTM " + report);
    }

    private static void testGruGradients() {
        GradientCheck.Report report = GradientCheck.checkLayer(
                GRU.builder().hiddenSize(3).seed(4).build(), sequenceInput(2, 3, 2, 21),
                sequenceInput(2, 3, 3, 22), EPSILON);
        check(report.passed(GRADIENT_TOLERANCE), "GRU " + report);
    }

    private static void testRnnGradients() {
        GradientCheck.Report report = GradientCheck.checkLayer(
                VanillaRNN.builder().hiddenSize(3).seed(5).build(), sequenceInput(2, 3, 2, 31),
                sequenceInput(2, 3, 3, 32), EPSILON);
        check(report.passed(GRADIENT_TOLERANCE), "VanillaRNN " + report);
    }

    private static void testAttentionGradients() {
        GradientCheck.Report full = GradientCheck.checkLayer(
                SelfAttention.builder().dimension(3).seed(6).build(),
                sequenceInput(2, 3, 3, 41), sequenceInput(2, 3, 3, 42), EPSILON);
        check(full.passed(GRADIENT_TOLERANCE), "attention " + full);
        GradientCheck.Report causal = GradientCheck.checkLayer(
                SelfAttention.builder().dimension(3).seed(6).causal(true).build(),
                sequenceInput(2, 3, 3, 41), sequenceInput(2, 3, 3, 42), EPSILON);
        check(causal.passed(GRADIENT_TOLERANCE), "causal attention " + causal);
    }

    private static void testCausalMask() {
        int batch = 1;
        int sequence = 4;
        int dimension = 3;
        SelfAttention attention = SelfAttention.builder().dimension(dimension).seed(8)
                .causal(true).build();
        Random random = new Random(51);
        double[] base = new double[batch * sequence * dimension];
        for (int i = 0; i < base.length; i++) {
            base[i] = random.nextDouble() - 0.5;
        }
        double[] changed = base.clone();
        changed[3 * dimension] += 10.0;
        double[] first = attention.forward(Tensor.ofBSF(batch, sequence, dimension, base)).data();
        double[] second = attention.forward(Tensor.ofBSF(batch, sequence, dimension, changed)).data();
        for (int feature = 0; feature < dimension; feature++) {
            assertClose(first[feature], second[feature], 1e-12,
                    "causal output at t=0 must ignore t=1");
        }
        check(Math.abs(first[3 * dimension] - second[3 * dimension]) > 1e-6,
                "later positions must react to their own input");
    }

    private static void testNetworkGradients() {
        SequenceNetwork network = SequenceNetwork.builder()
                .add(LSTM.builder().hiddenSize(3).seed(2).build())
                .add(SelfAttention.builder().dimension(3).seed(3).build())
                .pooling(SequencePool.Pooling.MEAN)
                .head(Dense.builder().units(3).activation(Activation.SOFTMAX).seed(4).build())
                .loss(Loss.CATEGORICAL_CROSSENTROPY)
                .optimizer(Optimizer.ADAM.withLearningRate(0.01))
                .gradientClip(5.0)
                .build();
        Tensor<BSF> input = sequenceInput(2, 3, 2, 61);
        Tensor<BC> target = Tensor.ofBC(2, 3, 1, 0, 0, 0, 1, 0);
        GradientCheck.Report report = GradientCheck.checkNetwork(network, input, target, EPSILON, 6);
        check(report.passed(GRADIENT_TOLERANCE), "sequence network " + report);
    }

    private static void testSequenceTraining() {
        Dataset data = memoryDataset(64, 8, 2, 3);
        SequenceNetwork network = SequenceNetwork.builder()
                .add(LSTM.builder().hiddenSize(8).seed(2).build())
                .pooling(SequencePool.Pooling.LAST)
                .head(Dense.builder().units(2).activation(Activation.SOFTMAX).seed(3).build())
                .loss(Loss.CATEGORICAL_CROSSENTROPY)
                .optimizer(Optimizer.ADAM.withLearningRate(0.01))
                .gradientClip(5.0)
                .build();
        TrainingHistory history = network.fit(data.features, data.targets, 150);
        check(history.lastLoss() < 0.05, "final loss " + history.lastLoss());
        check(history.lastLoss() < history.losses()[0], "loss must decrease");
        check(accuracy(network.predict(data.features), data.labels) >= 0.99,
                "training accuracy " + accuracy(network.predict(data.features), data.labels));
    }

    private static void testMiniBatchTraining() {
        Dataset data = memoryDataset(48, 8, 2, 13);
        SequenceNetwork network = SequenceNetwork.builder()
                .add(GRU.builder().hiddenSize(8).seed(6).build())
                .pooling(SequencePool.Pooling.LAST)
                .head(Dense.builder().units(2).activation(Activation.SOFTMAX).seed(7).build())
                .optimizer(Optimizer.ADAM.withLearningRate(0.01))
                .gradientClip(5.0)
                .build();
        TrainingHistory history = network.fit(data.features, data.targets, 120, 16);
        for (double loss : history.losses()) {
            check(Double.isFinite(loss), "mini-batch loss must be finite");
        }
        check(accuracy(network.predict(data.features), data.labels) >= 0.9,
                "GRU mini-batch accuracy " + accuracy(network.predict(data.features), data.labels));
    }

    /**
     * Hand-initialized comparison: an LSTM whose forget gate is saturated near one keeps the
     * gradient of timestep 0 alive across 30 steps, while a vanilla RNN with the same spectral
     * scale annihilates it. This isolates the constant error carousel, not the training budget.
     */
    private static void testGradientPreservation() {
        int sequence = 30;
        int hidden = 4;
        Tensor<BSF> zeros = Tensor.ofBSF(1, sequence, 1, new double[sequence]);
        Tensor<BSF> upstream = Tensor.ofBSF(1, sequence, hidden, new double[sequence * hidden]);
        upstream.data()[(sequence - 1) * hidden] = 1.0;

        LSTM lstm = LSTM.builder().hiddenSize(hidden).seed(1).build();
        lstm.forward(zeros);
        for (ParameterRef parameter : lstm.parameters()) {
            Arrays.fill(parameter.values(), 0.1);
        }
        double[] bias = lstm.parameters().get(2).values();
        for (int unit = 0; unit < hidden; unit++) {
            bias[hidden + unit] = 5.0;
        }
        lstm.forward(zeros);
        double lstmGradient = norm(lstm.backward(upstream).data(), 1);

        VanillaRNN rnn = VanillaRNN.builder().hiddenSize(hidden).seed(1).build();
        rnn.forward(zeros);
        for (ParameterRef parameter : rnn.parameters()) {
            Arrays.fill(parameter.values(), 0.1);
        }
        rnn.forward(zeros);
        double rnnGradient = norm(rnn.backward(upstream).data(), 1);

        check(lstmGradient > 0.0 && rnnGradient > 0.0, "both gradients must be non-zero");
        double ratio = lstmGradient / rnnGradient;
        check(ratio > 1e6, "LSTM must preserve long-range gradient: ratio " + ratio);
    }

    private static void testSerialization() throws Exception {
        Path directory = Path.of(System.getProperty("user.dir"), "build");
        Files.createDirectories(directory);
        Tensor<BSF> input = sequenceInput(2, 4, 2, 71);
        java.util.List<Layer<BSF, BSF>> layers = java.util.List.of(
                LSTM.builder().hiddenSize(3).seed(2).build(),
                GRU.builder().hiddenSize(3).seed(3).build(),
                VanillaRNN.builder().hiddenSize(3).seed(4).build(),
                SelfAttention.builder().dimension(2).seed(5).causal(true).build());
        ModelKind[] kinds = {ModelKind.LSTM, ModelKind.GRU, ModelKind.VANILLA_RNN,
                ModelKind.SELF_ATTENTION};
        for (int index = 0; index < layers.size(); index++) {
            Layer<BSF, BSF> layer = layers.get(index);
            double[] expected = layer.forward(input).data().clone();
            Path path = directory.resolve("sequence-" + kinds[index] + ".bsl");
            ModelCodec.save(path, (Persistable) layer);
            Persistable persistable = ModelCodec.load(path);
            check(persistable instanceof Layer, "loaded layer");
            @SuppressWarnings("unchecked")
            Layer<BSF, BSF> loaded = (Layer<BSF, BSF>) persistable;
            check(ModelCodec.peekKind(path) == kinds[index], "kind " + kinds[index]);
            double[] actual = loaded.forward(input).data();
            check(Arrays.equals(expected, actual), kinds[index] + " output must be bit-exact");
        }

        SequenceNetwork network = SequenceNetwork.builder()
                .add(LSTM.builder().hiddenSize(4).seed(2).build())
                .add(SelfAttention.builder().dimension(4).seed(3).build())
                .pooling(SequencePool.Pooling.MEAN)
                .head(Dense.builder().units(2).activation(Activation.SOFTMAX).seed(4).build())
                .optimizer(Optimizer.ADAM.withLearningRate(0.005))
                .gradientClip(3.0)
                .build();
        Tensor<BSF> sequence = sequenceInput(3, 4, 2, 81);
        Tensor<BC> target = Tensor.ofBC(3, 2, 1, 0, 0, 1, 1, 0);
        network.fit(sequence, target, 3);
        Path path = directory.resolve("sequence-network.bsl");
        ModelCodec.save(path, network);
        SequenceNetwork loaded = ModelCodec.load(path, SequenceNetwork.class);
        double[] expected = network.predict(sequence).data();
        double[] actual = loaded.predict(sequence).data();
        check(Arrays.equals(expected, actual), "sequence network output must be bit-exact");
        check(loaded.pooling() == SequencePool.Pooling.MEAN, "pooling restored");
        check(loaded.optimizer().learningRate() == 0.005, "optimizer restored");
    }

    private static Tensor<BSF> sequenceInput(int batch, int sequence, int features, long seed) {
        Random random = new Random(seed);
        double[] values = new double[batch * sequence * features];
        for (int i = 0; i < values.length; i++) {
            values[i] = random.nextDouble() * 2.0 - 1.0;
        }
        return Tensor.ofBSF(batch, sequence, features, values);
    }

    private static Dataset memoryDataset(int samples, int sequence, int features, long seed) {
        Random random = new Random(seed);
        double[] values = new double[samples * sequence * features];
        double[] labels = new double[samples];
        double[] targets = new double[samples * 2];
        for (int sample = 0; sample < samples; sample++) {
            int label = sample % 2;
            labels[sample] = label;
            targets[sample * 2 + label] = 1.0;
            for (int time = 0; time < sequence; time++) {
                for (int feature = 0; feature < features; feature++) {
                    values[(sample * sequence + time) * features + feature]
                            = 0.1 * random.nextGaussian();
                }
            }
            values[(sample * sequence) * features + label] = 1.0;
        }
        return new Dataset(Tensor.ofBSF(samples, sequence, features, values),
                Tensor.ofBC(samples, 2, targets), labels);
    }

    private static double accuracy(Tensor<BC> prediction, double[] labels) {
        int correct = 0;
        for (int row = 0; row < labels.length; row++) {
            int best = 0;
            for (int column = 1; column < prediction.dimension(1); column++) {
                if (prediction.get(row, column) > prediction.get(row, best)) {
                    best = column;
                }
            }
            if (best == (int) labels[row]) {
                correct++;
            }
        }
        return (double) correct / labels.length;
    }

    private static double norm(double[] values, int length) {
        double sum = 0.0;
        for (int i = 0; i < length; i++) {
            sum += values[i] * values[i];
        }
        return Math.sqrt(sum);
    }

    private record Dataset(Tensor<BSF> features, Tensor<BC> targets, double[] labels) {
    }

    private static void run(String name, ThrowingRunnable test) {
        try {
            test.run();
            passed++;
            System.out.println("OK: " + name);
        } catch (Throwable failure) {
            System.err.println("FAILED: " + name);
            failure.printStackTrace();
            System.exit(1);
        }
    }

    private static void assertClose(double expected, double actual, double tolerance, String message) {
        check(Double.isFinite(actual) && Math.abs(expected - actual) <= tolerance,
                message + ": expected " + expected + " but got " + actual);
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
