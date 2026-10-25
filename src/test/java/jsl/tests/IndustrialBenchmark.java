package jsl.tests;

import jsl.core.BCHW;
import jsl.core.BSF;
import jsl.core.Matrix;
import jsl.core.Tensor;
import jsl.data.DataLoader;
import jsl.data.Normalization;
import jsl.ml.BoostingAlgorithm;
import jsl.ml.BoostingObjective;
import jsl.ml.HistogramGradientBoosting;
import jsl.nn.Activation;
import jsl.nn.CNN;
import jsl.nn.Conv2D;
import jsl.nn.Dense;
import jsl.nn.Flatten;
import jsl.nn.LSTM;
import jsl.nn.MaxPooling2D;
import jsl.nn.ModelSerializer;
import jsl.nn.NeuralNetwork;
import jsl.nn.Optimizer;
import jsl.nn.OptimizerConfig;
import jsl.nn.Padding;
import jsl.nn.SelfAttention;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Random;

/** Reproducible microbenchmarks for the industrial report; not a replacement for JMH. */
public final class IndustrialBenchmark {
    private IndustrialBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        int rows = args.length > 0 ? Integer.parseInt(args[0]) : 50_000;
        if (rows < 2) {
            throw new IllegalArgumentException("rows must be at least 2");
        }
        benchmarkLoader(rows);
        benchmarkBoosting();
        benchmarkSequenceLayers();
        benchmarkSerialization();
    }

    private static void benchmarkLoader(int rows) throws Exception {
        int features = 8;
        double[] values = new double[rows * features];
        double[] labels = new double[rows];
        Random random = new Random(17);
        for (int i = 0; i < values.length; i++) {
            values[i] = random.nextDouble() * 20.0 - 10.0;
        }
        for (int row = 0; row < rows; row++) {
            labels[row] = row % 2;
        }
        Path path = Path.of(System.getProperty("user.dir"), "build", "benchmark-data.jsld");
        DataLoader.write(path, new Matrix(rows, features, values), labels);
        Normalization normalization = DataLoader.estimateNormalization(path, features);
        int iterations = 5;
        double checksum = 0.0;
        long start = System.nanoTime();
        for (int iteration = 0; iteration < iterations; iteration++) {
            try (DataLoader.BatchStream stream = DataLoader.builder(path).featureCount(features)
                    .batchSize(1024).normalization(normalization).open()) {
                while (stream.next()) {
                    DataLoader.Batch batch = stream.batch();
                    for (int row = 0; row < batch.rows(); row++) {
                        checksum += batch.label(row) + batch.feature(row, 0);
                    }
                }
            }
        }
        double seconds = (System.nanoTime() - start) / 1e9;
        double megabytes = Files.size(path) * (double) iterations / (1024.0 * 1024.0);
        System.out.printf(Locale.ROOT,
                "DataLoader: rows=%d, features=%d, %.2f MiB/s, %.0f rows/s, checksum=%.3f%n",
                rows, features, megabytes / seconds, rows * (double) iterations / seconds, checksum);
    }

    private static void benchmarkBoosting() {
        int rows = 1_000;
        int features = 8;
        double[] values = new double[rows * features];
        double[] labels = new double[rows];
        Random random = new Random(23);
        for (int row = 0; row < rows; row++) {
            for (int feature = 0; feature < features; feature++) {
                values[row * features + feature] = random.nextDouble() * 2.0 - 1.0;
            }
            labels[row] = values[row * features] + values[row * features + 1] > 0.0 ? 1.0 : 0.0;
        }
        Matrix data = new Matrix(rows, features, values);
        for (BoostingAlgorithm algorithm : BoostingAlgorithm.values()) {
            long start = System.nanoTime();
            HistogramGradientBoosting model = HistogramGradientBoosting.builder()
                    .algorithm(algorithm).objective(BoostingObjective.LOGISTIC)
                    .iterations(30).learningRate(0.15).maxDepth(4).maxLeaves(12)
                    .bins(32).build().fit(data, labels);
            double millis = (System.nanoTime() - start) / 1e6;
            System.out.printf(Locale.ROOT, "%s: 30 trees, %.3f ms, %d trees total%n",
                    algorithm, millis, model.treeCount());
        }
    }

    private static void benchmarkSequenceLayers() {
        int batch = 8;
        int sequence = 16;
        int features = 8;
        int hidden = 16;
        double[] inputValues = new double[batch * sequence * features];
        double[] gradientValues = new double[batch * sequence * hidden];
        Random random = new Random(29);
        for (int i = 0; i < inputValues.length; i++) {
            inputValues[i] = random.nextDouble() - 0.5;
        }
        for (int i = 0; i < gradientValues.length; i++) {
            gradientValues[i] = random.nextDouble() - 0.5;
        }
        Tensor<BSF> input = Tensor.ofBSF(batch, sequence, features, inputValues);
        Tensor<BSF> gradient = Tensor.ofBSF(batch, sequence, hidden, gradientValues);
        LSTM lstm = LSTM.builder().hiddenSize(hidden).build();
        OptimizerConfig optimizer = Optimizer.ADAM.withLearningRate(0.001);
        for (int i = 0; i < 5; i++) {
            lstm.backward(lstm.forward(input));
            lstm.update(optimizer);
        }
        int iterations = 30;
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            lstm.backward(lstm.forward(input));
            lstm.update(optimizer);
        }
        double lstmMillis = (System.nanoTime() - start) / 1e6 / iterations;
        System.out.printf(Locale.ROOT, "LSTM train step: B=%d,T=%d,F=%d,H=%d, %.3f ms/step%n",
                batch, sequence, features, hidden, lstmMillis);

        Tensor<BSF> attentionInput = Tensor.ofBSF(batch, sequence, hidden, new double[batch * sequence * hidden]);
        SelfAttention attention = SelfAttention.builder().dimension(hidden).build();
        for (int i = 0; i < 5; i++) {
            attention.backward(attention.forward(attentionInput));
            attention.update(optimizer);
        }
        start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            attention.backward(attention.forward(attentionInput));
            attention.update(optimizer);
        }
        double attentionMillis = (System.nanoTime() - start) / 1e6 / iterations;
        System.out.printf(Locale.ROOT, "SelfAttention train step: B=%d,T=%d,D=%d, %.3f ms/step%n",
                batch, sequence, hidden, attentionMillis);
    }

    private static void benchmarkSerialization() {
        CNN model = NeuralNetwork.builder()
                .add(Conv2D.builder().filters(4).kernelSize(3, 3)
                        .activation(Activation.RELU).padding(Padding.SAME).build())
                .add(MaxPooling2D.builder().poolSize(2, 2).build())
                .add(Flatten.builder().build())
                .add(Dense.builder().units(4).activation(Activation.SOFTMAX).build())
                .optimizer(Optimizer.ADAM.withLearningRate(0.001)).build();
        Tensor<BCHW> input = Tensor.ofBCHW(8, 1, 4, 4, new double[8 * 16]);
        model.predict(input);
        Path path = Path.of(System.getProperty("user.dir"), "build", "benchmark-model.jsl");
        long start = System.nanoTime();
        ModelSerializer.save(path, model);
        double saveMillis = (System.nanoTime() - start) / 1e6;
        start = System.nanoTime();
        CNN loaded = ModelSerializer.loadCNN(path);
        double loadMillis = (System.nanoTime() - start) / 1e6;
        for (int i = 0; i < 10; i++) {
            loaded.predict(input);
        }
        int iterations = 200;
        start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            loaded.predict(input);
        }
        double inferenceMillis = (System.nanoTime() - start) / 1e6 / iterations;
        try {
            System.out.printf(Locale.ROOT,
                    "Serializer: %.3f ms save, %.3f ms load, %d bytes; CNN batch-8 inference %.3f ms%n",
                    saveMillis, loadMillis, Files.size(path), inferenceMillis);
        } catch (Exception failure) {
            throw new IllegalStateException(failure);
        }
    }
}
