package basalt.tests;

import basalt.core.BCHW;
import basalt.core.BSF;
import basalt.core.MatMul;
import basalt.core.Matrix;
import basalt.core.ModelKind;
import basalt.core.Persistable;
import basalt.core.Tensor;
import basalt.data.DataLoader;
import basalt.data.Normalization;
import basalt.io.ModelCodec;
import basalt.ml.BoostingAlgorithm;
import basalt.ml.BoostingObjective;
import basalt.ml.EigenSolver;
import basalt.ml.HistogramGradientBoosting;
import basalt.ml.KMeans;
import basalt.ml.PCA;
import basalt.ml.RandomForest;
import basalt.ml.TreeTask;
import basalt.nativebridge.CommandQueueBuilder;
import basalt.nativebridge.NativeContext;
import basalt.nativebridge.NativeLibrary;
import basalt.nativebridge.NativeMLP;
import basalt.nativebridge.NativeTensorHandle;
import basalt.nn.Activation;
import basalt.nn.CNN;
import basalt.nn.Conv2D;
import basalt.nn.Dense;
import basalt.nn.Flatten;
import basalt.nn.GRU;
import basalt.nn.LSTM;
import basalt.nn.MaxPooling2D;
import basalt.nn.NeuralNetwork;
import basalt.nn.Optimizer;
import basalt.nn.Padding;
import basalt.nn.ParameterRef;
import basalt.nn.SelfAttention;
import basalt.nn.SequenceNetwork;
import basalt.nn.SequencePool;
import basalt.nn.VanillaRNN;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Reproducible benchmark harness that emits CSV for the Python report generator.
 *
 * <p>Not JMH: each measurement is a warmup followed by repeated timed runs and a median. Results
 * are specific to this machine, JDK build and JVM flags; absolute numbers must not be compared
 * across hosts. The harness prints the environment header so every CSV can be traced to a run.</p>
 */
public final class BasaltBenchmark {
    private static final int WARMUP = 2;
    private static volatile double sink;

    private BasaltBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Path output = Path.of(System.getProperty("user.dir"), "build", "bench");
        Files.createDirectories(output);
        java.util.Set<String> sections = args.length == 0
                ? java.util.Set.of("all")
                : java.util.Set.of(args[0].split(","));
        System.out.printf(Locale.ROOT,
                "Basalt benchmark: java=%s, cpus=%d, maxMemory=%d MiB, sections=%s%n",
                System.getProperty("java.version"), Runtime.getRuntime().availableProcessors(),
                Runtime.getRuntime().maxMemory() >> 20, sections);
        if (enabled(sections, "matmul")) {
            matmul(output);
            matmulThreads(output);
        }
        if (enabled(sections, "loader")) {
            dataLoader(output);
        }
        if (enabled(sections, "boosting")) {
            boosting(output);
        }
        if (enabled(sections, "forest")) {
            forest(output);
        }
        if (enabled(sections, "kmeans")) {
            kmeans(output);
        }
        if (enabled(sections, "pca")) {
            pca(output);
        }
        if (enabled(sections, "sequence")) {
            sequence(output);
        }
        if (enabled(sections, "serialization")) {
            serialization(output);
        }
        if (enabled(sections, "convergence")) {
            convergence(output);
            gradientDecay(output);
        }
        if (enabled(sections, "native")) {
            nativeBench(output);
        }
        System.out.println("Benchmark CSV written to " + output.toAbsolutePath());
    }

    private static boolean enabled(java.util.Set<String> sections, String name) {
        return sections.contains("all") || sections.contains(name);
    }

    // ---------------------------------------------------------------- native tensor engine

    private static void nativeBench(Path output) throws Exception {
        if (!NativeLibrary.available()) {
            System.out.println("native tensor runtime unavailable; skipping native section");
            return;
        }
        nativeGemm(output);
        nativeMlp(output);
    }

    private static void nativeGemm(Path output) throws Exception {
        int[] sizes = {256, 512, 1024};
        int[] repetitions = {5, 5, 3};
        List<String> rows = new ArrayList<>();
        for (int index = 0; index < sizes.length; index++) {
            int size = sizes[index];
            int reps = repetitions[index];
            Random random = new Random(700 + size);
            float[] leftFloat = new float[size * size];
            float[] rightFloat = new float[size * size];
            double[] leftDouble = new double[size * size];
            double[] rightDouble = new double[size * size];
            for (int element = 0; element < leftFloat.length; element++) {
                leftFloat[element] = random.nextFloat() * 2.0f - 1.0f;
                leftDouble[element] = leftFloat[element];
                rightFloat[element] = random.nextFloat() * 2.0f - 1.0f;
                rightDouble[element] = rightFloat[element];
            }
            Matrix left = new Matrix(size, size, leftDouble);
            Matrix right = new Matrix(size, size, rightDouble);
            double javaMs;
            try (MatMul engine = MatMul.create(8)) {
                engine.multiply(left, right);
                javaMs = medianMs(() -> sink += engine.multiply(left, right).get(0, 0), 1, reps);
            }
            double nativeMs;
            try (NativeContext context = NativeContext.create(8)) {
                NativeTensorHandle a = context.of(leftFloat, false, size, size);
                NativeTensorHandle b = context.of(rightFloat, false, size, size);
                NativeTensorHandle c = context.zeros(size, size);
                CommandQueueBuilder queue = new CommandQueueBuilder(context);
                context.clearGraph();
                queue.matMul(a, b, c).execute();
                queue.clear();
                double[] times = new double[reps];
                for (int repetition = 0; repetition < reps; repetition++) {
                    context.clearGraph();
                    queue.matMul(a, b, c);
                    long start = System.nanoTime();
                    queue.execute();
                    times[repetition] = (System.nanoTime() - start) / 1e6;
                    queue.clear();
                    sink += c.data()[0];
                }
                Arrays.sort(times);
                nativeMs = times[reps / 2];
                a.close();
                b.close();
                c.close();
            }
            double gflops = 2.0 * size * size * size / (nativeMs * 1e6);
            rows.add(String.format(Locale.ROOT, "%d,%.4f,%.4f,%.2f,%.2f",
                    size, javaMs, nativeMs, javaMs / nativeMs, gflops));
            System.out.printf(Locale.ROOT,
                    "native gemm %4d: java-double %.2f ms, native-f32 %.2f ms, %.2fx, %.1f GF%n",
                    size, javaMs, nativeMs, javaMs / nativeMs, gflops);
        }
        write(output.resolve("native_gemm.csv"),
                "size,java_double_ms,native_f32_ms,speedup,native_gflops", rows);
    }

    private static void nativeMlp(Path output) {
        int rows = 4_096;
        int features = 32;
        int outputs = 8;
        int epochs = 20;
        int batch = 256;
        Random random = new Random(991);
        float[][] x = new float[rows][features];
        float[][] y = new float[rows][outputs];
        for (int row = 0; row < rows; row++) {
            for (int feature = 0; feature < features; feature++) {
                x[row][feature] = random.nextFloat() * 2.0f - 1.0f;
            }
            for (int unit = 0; unit < outputs; unit++) {
                float value = 0.0f;
                for (int feature = 0; feature < features; feature++) {
                    value += x[row][feature] * ((feature + unit) % 5 - 2) * 0.2f;
                }
                y[row][unit] = (float) Math.tanh(value);
            }
        }
        int[] layers = {features, 64, 32, outputs};
        try (NativeContext warmupContext = NativeContext.create(8);
             NativeMLP warmup = new NativeMLP(warmupContext, layers,
                     NativeMLP.Activation.RELU, NativeMLP.OutputActivation.TANH, 991)) {
            warmup.fit(x, y, 2, batch, 0.01f, 1e-4f);
        }
        float loss;
        double totalMs;
        try (NativeContext context = NativeContext.create(8);
             NativeMLP mlp = new NativeMLP(context, layers, NativeMLP.Activation.RELU,
                     NativeMLP.OutputActivation.TANH, 991)) {
            long start = System.nanoTime();
            loss = mlp.fit(x, y, epochs, batch, 0.01f, 1e-4f);
            totalMs = (System.nanoTime() - start) / 1e6;
            long predictStart = System.nanoTime();
            float[] prediction = mlp.predict(x);
            double predictMs = (System.nanoTime() - predictStart) / 1e6;
            sink += prediction[0];
            List<String> rowsOut = new ArrayList<>();
            rowsOut.add(String.format(Locale.ROOT,
                    "%d,%d,%d,%d,%d,%.2f,%.2f,%.6f,%.2f",
                    rows, features, outputs, epochs, batch, totalMs, totalMs / epochs,
                    loss, predictMs));
            write(output.resolve("native_mlp.csv"),
                    "rows,features,outputs,epochs,batch,total_ms,ms_per_epoch,loss,predict_ms",
                    rowsOut);
            System.out.printf(Locale.ROOT,
                    "native mlp: %d epochs in %.1f ms (%.2f ms/epoch), loss=%.6f, predict %.2f ms%n",
                    epochs, totalMs, totalMs / epochs, loss, predictMs);
        }
    }

    // ---------------------------------------------------------------- matrix multiplication

    private static void matmul(Path output) throws Exception {
        int[] sizes = {64, 128, 256, 512, 1024};
        int[] repetitions = {30, 20, 10, 4, 1};
        List<String> rows = new ArrayList<>();
        for (int index = 0; index < sizes.length; index++) {
            int size = sizes[index];
            Matrix left = randomMatrix(size, size, 100 + size);
            Matrix right = randomMatrix(size, size, 200 + size);
            int reps = repetitions[index];
            double scalarMs = medianMs(() -> sink += MatMul.scalar(left, right).get(0, 0),
                    Math.min(WARMUP, reps), reps);
            double simdMs;
            try (MatMul engine = MatMul.create()) {
                simdMs = medianMs(() -> sink += engine.multiply(left, right).get(0, 0),
                        WARMUP, reps);
            }
            double gflops = 2.0 * size * size * size / (simdMs * 1e6);
            rows.add(String.format(Locale.ROOT, "%d,%.4f,%.4f,%.2f,%.2f",
                    size, scalarMs, simdMs, scalarMs / simdMs, gflops));
            System.out.printf(Locale.ROOT, "matmul %4d: scalar %.2f ms, SIMD %.2f ms, %.2fx%n",
                    size, scalarMs, simdMs, scalarMs / simdMs);
        }
        write(output.resolve("matmul.csv"), "size,scalar_ms,simd_ms,speedup,gflops", rows);
    }

    private static void matmulThreads(Path output) throws Exception {
        int size = 512;
        Matrix left = randomMatrix(size, size, 301);
        Matrix right = randomMatrix(size, size, 302);
        List<String> rows = new ArrayList<>();
        for (int threads : new int[]{1, 2, 4, 8}) {
            try (MatMul engine = MatMul.create(threads)) {
                double ms = medianMs(() -> sink += engine.multiply(left, right).get(0, 0), WARMUP, 4);
                rows.add(String.format(Locale.ROOT, "%d,%.4f,%.2f",
                        threads, ms, 2.0 * size * size * size / (ms * 1e6)));
                System.out.printf(Locale.ROOT, "matmul threads %d: %.2f ms%n", threads, ms);
            }
        }
        write(output.resolve("matmul_threads.csv"), "threads,ms,gflops", rows);
    }

    // ---------------------------------------------------------------- data loader

    private static void dataLoader(Path output) throws Exception {
        int rows = 200_000;
        int features = 8;
        Random random = new Random(7);
        double[] values = new double[rows * features];
        double[] labels = new double[rows];
        for (int i = 0; i < values.length; i++) {
            values[i] = random.nextDouble() * 20.0 - 10.0;
        }
        for (int row = 0; row < rows; row++) {
            labels[row] = row % 2;
        }
        Path path = output.resolve("loader-data.bsld");
        DataLoader.write(path, new Matrix(rows, features, values), labels);
        Normalization normalization = DataLoader.estimateNormalization(path, features);
        double megabytes = Files.size(path) / (1024.0 * 1024.0);
        List<String> result = new ArrayList<>();
        for (String mode : new String[]{"sequential", "mapped", "mapped-shuffle"}) {
            for (int batchSize : new int[]{256, 1024, 4096}) {
                long allocated = 0;
                double seconds;
                if (mode.equals("sequential")) {
                    consumeSequential(path, features, batchSize, normalization);
                    allocated = allocatedBytes(() -> consumeSequentialUnchecked(
                            path, features, batchSize, normalization));
                    seconds = medianSeconds(() -> consumeSequentialUnchecked(
                            path, features, batchSize, normalization), 1, 3);
                } else {
                    boolean shuffle = mode.equals("mapped-shuffle");
                    consumeMapped(path, features, batchSize, normalization, shuffle);
                    allocated = allocatedBytes(() -> consumeMappedUnchecked(
                            path, features, batchSize, normalization, shuffle));
                    seconds = medianSeconds(() -> consumeMappedUnchecked(
                            path, features, batchSize, normalization, shuffle), 1, 3);
                }
                result.add(String.format(Locale.ROOT, "%s,%d,%d,%d,%.1f,%.0f,%.2f",
                        mode, batchSize, rows, features, megabytes / seconds,
                        rows / seconds, allocated / (double) rows));
                System.out.printf(Locale.ROOT, "loader %-14s batch=%4d: %.1f MiB/s, %.2f B/row%n",
                        mode, batchSize, megabytes / seconds, allocated / (double) rows);
            }
        }
        write(output.resolve("loader.csv"),
                "mode,batch_size,rows,features,mb_per_s,rows_per_s,bytes_per_row", result);
    }

    private static void consumeSequential(Path path, int features, int batchSize,
                                          Normalization normalization) throws Exception {
        try (DataLoader.BatchStream stream = DataLoader.builder(path).featureCount(features)
                .batchSize(batchSize).normalization(normalization).open()) {
            while (stream.next()) {
                DataLoader.Batch batch = stream.batch();
                for (int row = 0; row < batch.rows(); row++) {
                    sink += batch.feature(row, 0) + batch.label(row);
                }
            }
        }
    }

    private static void consumeSequentialUnchecked(Path path, int features, int batchSize,
                                                   Normalization normalization) {
        try {
            consumeSequential(path, features, batchSize, normalization);
        } catch (Exception failure) {
            throw new IllegalStateException(failure);
        }
    }

    private static void consumeMapped(Path path, int features, int batchSize,
                                      Normalization normalization, boolean shuffle) {
        try (DataLoader.MappedBatchStream stream = DataLoader.builder(path)
                .featureCount(features).batchSize(batchSize).shuffle(shuffle)
                .normalization(normalization).openMapped()) {
            while (stream.next()) {
                DataLoader.Batch batch = stream.batch();
                for (int row = 0; row < batch.rows(); row++) {
                    sink += batch.feature(row, 0) + batch.label(row);
                }
            }
        }
    }

    private static void consumeMappedUnchecked(Path path, int features, int batchSize,
                                               Normalization normalization, boolean shuffle) {
        consumeMapped(path, features, batchSize, normalization, shuffle);
    }

    // ---------------------------------------------------------------- trees

    private static void boosting(Path output) {
        int rows = 5_000;
        int features = 16;
        Random random = new Random(11);
        double[] values = new double[rows * features];
        double[] labels = new double[rows];
        for (int row = 0; row < rows; row++) {
            double signal = 0.0;
            for (int feature = 0; feature < features; feature++) {
                double value = random.nextGaussian();
                values[row * features + feature] = value;
                signal += Math.sin(value * (feature % 4 + 1));
            }
            labels[row] = signal > 0 ? 1 : 0;
        }
        Matrix data = new Matrix(rows, features, values);
        List<String> result = new ArrayList<>();
        HistogramGradientBoosting.builder().objective(BoostingObjective.LOGISTIC)
                .iterations(3).maxDepth(3).bins(16).threads(8).build().fit(data, labels);
        for (BoostingAlgorithm algorithm : BoostingAlgorithm.values()) {
            for (int threads : new int[]{1, 2, 4, 8}) {
                double ms = medianMs(() -> sink += HistogramGradientBoosting.builder()
                        .algorithm(algorithm).objective(BoostingObjective.LOGISTIC)
                        .iterations(30).learningRate(0.15).maxDepth(5).maxLeaves(20)
                        .bins(64).threads(threads).build().fit(data, labels).treeCount(), 1, 3);
                result.add(String.format(Locale.ROOT, "%s,%d,%d,%d,%.1f,%d",
                        algorithm, threads, rows, features, ms, 30));
                System.out.printf(Locale.ROOT, "boosting %-8s threads=%d: %.1f ms%n",
                        algorithm, threads, ms);
            }
        }
        write(output.resolve("boosting.csv"),
                "algorithm,threads,rows,features,ms,trees", result);

        int largeRows = 20_000;
        int largeFeatures = 64;
        Random largeRandom = new Random(12);
        double[] largeValues = new double[largeRows * largeFeatures];
        double[] largeLabels = new double[largeRows];
        for (int row = 0; row < largeRows; row++) {
            double signal = 0.0;
            for (int feature = 0; feature < largeFeatures; feature++) {
                double value = largeRandom.nextGaussian();
                largeValues[row * largeFeatures + feature] = value;
                signal += Math.sin(value * (feature % 6 + 1));
            }
            largeLabels[row] = signal > 0 ? 1 : 0;
        }
        Matrix large = new Matrix(largeRows, largeFeatures, largeValues);
        List<String> scale = new ArrayList<>();
        HistogramGradientBoosting.builder().objective(BoostingObjective.LOGISTIC)
                .iterations(2).maxDepth(3).bins(16).threads(8).build()
                .fit(large, largeLabels);
        for (int threads : new int[]{1, 2, 4, 8}) {
            double ms = medianMs(() -> sink += HistogramGradientBoosting.builder()
                    .algorithm(BoostingAlgorithm.LIGHTGBM)
                    .objective(BoostingObjective.LOGISTIC)
                    .iterations(20).learningRate(0.15).maxDepth(5).maxLeaves(20)
                    .bins(64).threads(threads).build().fit(large, largeLabels).treeCount(),
                    1, 3);
            scale.add(String.format(Locale.ROOT, "%d,%d,%d,%.1f,%d",
                    threads, largeRows, largeFeatures, ms, 20));
            System.out.printf(Locale.ROOT, "boosting large threads=%d: %.1f ms%n", threads, ms);
        }
        write(output.resolve("boosting_scale.csv"), "threads,rows,features,ms,trees", scale);
    }

    private static void forest(Path output) {
        int rows = 5_000;
        int features = 16;
        Random random = new Random(13);
        double[] values = new double[rows * features];
        double[] labels = new double[rows];
        for (int row = 0; row < rows; row++) {
            double signal = 0.0;
            for (int feature = 0; feature < features; feature++) {
                double value = random.nextGaussian();
                values[row * features + feature] = value;
                signal += value * (feature + 1);
            }
            labels[row] = signal > 0 ? 1 : 0;
        }
        Matrix data = new Matrix(rows, features, values);
        List<String> result = new ArrayList<>();
        for (int threads : new int[]{1, 2, 4, 8}) {
            double ms = medianMs(() -> sink += RandomForest.builder().trees(25).maxDepth(8)
                    .threads(threads).seed(5).build().fit(data, labels).treeCount(), 1, 3);
            result.add(String.format(Locale.ROOT, "%d,25,%d,%d,%.1f,%d",
                    threads, rows, features, ms, 25));
            System.out.printf(Locale.ROOT, "forest threads=%d: %.1f ms%n", threads, ms);
        }
        for (int trees : new int[]{10, 25, 50}) {
            double ms = medianMs(() -> sink += RandomForest.builder().trees(trees).maxDepth(8)
                    .threads(8).seed(5).build().fit(data, labels).treeCount(), 1, 3);
            result.add(String.format(Locale.ROOT, "8,%d,%d,%d,%.1f,%d",
                    trees, rows, features, ms, trees));
            System.out.printf(Locale.ROOT, "forest trees=%d: %.1f ms%n", trees, ms);
        }
        write(output.resolve("forest.csv"), "threads,trees,rows,features,ms,actual_trees", result);
    }

    // ---------------------------------------------------------------- unsupervised

    private static void kmeans(Path output) {
        List<String> result = new ArrayList<>();
        for (int rows : new int[]{5_000, 20_000}) {
            for (int clusters : new int[]{4, 8, 16}) {
                Random random = new Random(17);
                int features = 8;
                double[] values = new double[rows * features];
                for (int row = 0; row < rows; row++) {
                    int cluster = row % clusters;
                    for (int feature = 0; feature < features; feature++) {
                        values[row * features + feature] = cluster * 5.0 + random.nextGaussian();
                    }
                }
                Matrix data = new Matrix(rows, features, values);
                KMeans.builder().clusters(clusters).maxIterations(15).restarts(1)
                        .threads(8).seed(3).build().fit(data);
                double ms = medianMs(() -> sink += KMeans.builder().clusters(clusters)
                        .maxIterations(15).restarts(1).threads(8).seed(3)
                        .build().fit(data).inertia(), 1, 3);
                KMeans model = KMeans.builder().clusters(clusters).maxIterations(15)
                        .restarts(1).threads(8).seed(3).build().fit(data);
                result.add(String.format(Locale.ROOT, "%d,%d,8,%.1f,%.3f,%d",
                        rows, clusters, ms, model.inertia(), model.iterations()));
                System.out.printf(Locale.ROOT, "kmeans rows=%d k=%d: %.1f ms%n", rows, clusters, ms);
            }
        }
        write(output.resolve("kmeans.csv"), "rows,clusters,threads,ms,inertia,iterations", result);
    }

    private static void pca(Path output) {
        List<String> result = new ArrayList<>();
        benchmarkPca(result, "jacobi", 400, 48, 8, EigenSolver.JACOBI, 31);
        benchmarkPca(result, "gram", 120, 1_200, 8, EigenSolver.GRAM, 32);
        benchmarkPca(result, "jacobi-tall", 5_000, 24, 8, EigenSolver.JACOBI, 33);
        write(output.resolve("pca.csv"), "solver,rows,features,components,ms,ratio_sum", result);
    }

    private static void benchmarkPca(List<String> result, String name, int rows, int features,
                                     int components, EigenSolver solver, long seed) {
        Random random = new Random(seed);
        double[] values = new double[rows * features];
        for (int i = 0; i < values.length; i++) {
            values[i] = random.nextGaussian();
        }
        Matrix data = new Matrix(rows, features, values);
        PCA.builder().components(components).solver(solver).build().fit(data);
        double ms = medianMs(() -> sink += PCA.builder().components(components)
                .solver(solver).build().fit(data).explainedVarianceRatio()[0], 1, 3);
        PCA pca = PCA.builder().components(components).solver(solver).build().fit(data);
        double ratioSum = 0.0;
        for (double ratio : pca.explainedVarianceRatio()) {
            ratioSum += ratio;
        }
        result.add(String.format(Locale.ROOT, "%s,%d,%d,%d,%.1f,%.4f",
                name, rows, features, components, ms, ratioSum));
        System.out.printf(Locale.ROOT, "pca %-12s %dx%d: %.1f ms%n", name, rows, features, ms);
    }

    // ---------------------------------------------------------------- sequence models

    private static void sequence(Path output) {
        int batch = 16;
        int hidden = 32;
        int features = 8;
        List<String> result = new ArrayList<>();
        Tensor<BSF> warmupInput = randomSequence(batch, 16, features, 40);
        warmup(LSTM.builder().hiddenSize(hidden).build(), warmupInput);
        warmup(GRU.builder().hiddenSize(hidden).build(), warmupInput);
        warmup(VanillaRNN.builder().hiddenSize(hidden).build(), warmupInput);
        warmup(SelfAttention.builder().dimension(features).build(), warmupInput);
        for (int sequence : new int[]{8, 32, 64}) {
            Tensor<BSF> input = randomSequence(batch, sequence, features, 41);
            sequenceRow(result, "LSTM", LSTM.builder().hiddenSize(hidden).build(),
                    input, batch, sequence, hidden);
            sequenceRow(result, "GRU", GRU.builder().hiddenSize(hidden).build(),
                    input, batch, sequence, hidden);
            sequenceRow(result, "RNN", VanillaRNN.builder().hiddenSize(hidden).build(),
                    input, batch, sequence, hidden);
            sequenceRow(result, "Attention", SelfAttention.builder().dimension(features).build(),
                    input, batch, sequence, hidden);
        }
        write(output.resolve("sequence.csv"), "layer,batch,seq,hidden,ms_per_step", result);
    }

    private static void warmup(basalt.nn.Layer<BSF, BSF> layer, Tensor<BSF> input) {
        for (int i = 0; i < 40; i++) {
            layer.backward(layer.forward(input));
        }
    }

    private static void sequenceRow(List<String> result, String name,
                                    basalt.nn.Layer<BSF, BSF> layer, Tensor<BSF> input,
                                    int batch, int sequence, int hidden) {
        for (int i = 0; i < 10; i++) {
            layer.backward(layer.forward(input));
        }
        int iterations = 15;
        double[] samples = new double[iterations];
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            sink += layer.backward(layer.forward(input)).data()[0];
            samples[i] = (System.nanoTime() - start) / 1e6;
        }
        Arrays.sort(samples);
        double ms = samples[samples.length / 2];
        result.add(String.format(Locale.ROOT, "%s,%d,%d,%d,%.3f",
                name, batch, sequence, hidden, ms));
        System.out.printf(Locale.ROOT, "sequence %-9s B=%d T=%d: %.3f ms/step%n",
                name, batch, sequence, ms);
    }

    // ---------------------------------------------------------------- serialization

    private static void serialization(Path output) throws Exception {
        List<String> result = new ArrayList<>();
        Path directory = Path.of(System.getProperty("user.dir"), "build", "bench", "models");
        Files.createDirectories(directory);

        CNN cnn = NeuralNetwork.builder()
                .add(Conv2D.builder().filters(8).kernelSize(3, 3)
                        .activation(Activation.RELU).padding(Padding.SAME).build())
                .add(MaxPooling2D.builder().poolSize(2, 2).build())
                .add(Flatten.builder().build())
                .add(Dense.builder().units(4).activation(Activation.SOFTMAX).build())
                .optimizer(Optimizer.ADAM.withLearningRate(0.001)).build();
        Tensor<BCHW> images = Tensor.ofBCHW(16, 1, 16, 16, new double[16 * 256]);
        cnn.predict(images);
        serialize(directory, result, "CNN", cnn, () -> cnn.predict(images).data()[0]);

        Random random = new Random(51);
        double[] values = new double[2_000 * 8];
        double[] targets = new double[2_000];
        for (int row = 0; row < 2_000; row++) {
            for (int feature = 0; feature < 8; feature++) {
                values[row * 8 + feature] = random.nextGaussian();
            }
            targets[row] = values[row * 8] > 0 ? 1 : 0;
        }
        Matrix data = new Matrix(2_000, 8, values);
        HistogramGradientBoosting boosting = HistogramGradientBoosting.builder()
                .objective(BoostingObjective.LOGISTIC).iterations(200).maxDepth(5)
                .bins(32).threads(8).build().fit(data, targets);
        serialize(directory, result, "GBDT", boosting, () -> boosting.predict(data)[0]);

        RandomForest forest = RandomForest.builder().trees(100).maxDepth(8)
                .threads(8).build().fit(data, targets);
        serialize(directory, result, "RandomForest", forest, () -> forest.predict(data)[0]);

        Tensor<BSF> sequence = randomSequence(8, 16, 32, 61);
        LSTM lstm = LSTM.builder().hiddenSize(128).build();
        lstm.forward(sequence);
        serialize(directory, result, "LSTM", lstm, () -> lstm.forward(sequence).data()[0]);

        SequenceNetwork network = SequenceNetwork.builder()
                .add(GRU.builder().hiddenSize(16).build())
                .add(SelfAttention.builder().dimension(16).build())
                .pooling(SequencePool.Pooling.MEAN)
                .head(Dense.builder().units(2).activation(Activation.SOFTMAX).build())
                .build();
        Tensor<BSF> networkInput = randomSequence(8, 16, 8, 62);
        Tensor<basalt.core.BC> networkTarget = Tensor.ofBC(8, 2,
                1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1);
        network.fit(networkInput, networkTarget, 2);
        serialize(directory, result, "SequenceNetwork", network,
                () -> network.predict(networkInput).data()[0]);

        write(output.resolve("serialization.csv"),
                "model,kind,bytes,save_ms,load_ms,load_mb_s,infer_us", result);
    }

    private static void serialize(Path directory, List<String> rows, String name,
                                  Persistable model, DoubleSupplier inference) throws Exception {
        Path path = directory.resolve(name + ".bsl");
        double saveMs = medianMs(() -> ModelCodec.save(path, model), 2, 5);
        double loadMs = medianMs(() -> sink += ModelCodec.load(path).kind().code(), 2, 5);
        Persistable loaded = ModelCodec.load(path);
        if (loaded.kind() != model.kind()) {
            throw new IllegalStateException("Kind mismatch for " + name);
        }
        for (int i = 0; i < 20; i++) {
            sink += inference.get();
        }
        double inferUs = medianMs(() -> sink += inference.get(), 5, 50) * 1000.0;
        long bytes = Files.size(path);
        double loadMb = bytes / (1024.0 * 1024.0) / (loadMs / 1000.0);
        rows.add(String.format(Locale.ROOT, "%s,%s,%d,%.3f,%.3f,%.1f,%.1f",
                name, model.kind(), bytes, saveMs, loadMs, loadMb, inferUs));
        System.out.printf(Locale.ROOT,
                "serialize %-16s %7d B, save %.2f ms, load %.2f ms, infer %.1f us%n",
                name, bytes, saveMs, loadMs, inferUs);
    }

    // ---------------------------------------------------------------- convergence and gradients

    private static void convergence(Path output) {
        int rows = 2_000;
        int features = 8;
        Random random = new Random(71);
        double[] values = new double[rows * features];
        double[] targets = new double[rows];
        for (int row = 0; row < rows; row++) {
            double signal = 0.0;
            for (int feature = 0; feature < features; feature++) {
                double value = random.nextGaussian();
                values[row * features + feature] = value;
                signal += Math.sin(value * (feature + 1));
            }
            targets[row] = signal + 0.3 * random.nextGaussian();
        }
        Matrix data = new Matrix(rows, features, values);
        List<String> rowsOut = new ArrayList<>();
        for (BoostingAlgorithm algorithm : BoostingAlgorithm.values()) {
            HistogramGradientBoosting model = HistogramGradientBoosting.builder()
                    .algorithm(algorithm).iterations(80).learningRate(0.1)
                    .maxDepth(6).maxLeaves(8).bins(32).validationFraction(0.2)
                    .threads(8).build().fit(data, targets);
            double[] training = model.lossHistory();
            double[] validation = model.validationHistory();
            for (int iteration = 0; iteration < training.length; iteration++) {
                rowsOut.add(String.format(Locale.ROOT, "%s,%d,%.6f,%.6f",
                        algorithm, iteration, training[iteration],
                        validation.length > iteration ? validation[iteration] : Double.NaN));
            }
        }
        write(output.resolve("convergence_gbdt.csv"),
                "algorithm,iteration,train_loss,val_loss", rowsOut);

        int samples = 64;
        int sequence = 8;
        int sequenceFeatures = 2;
        Random sequenceRandom = new Random(73);
        double[] sequenceValues = new double[samples * sequence * sequenceFeatures];
        double[] sequenceTargets = new double[samples * 2];
        for (int sample = 0; sample < samples; sample++) {
            int label = sample % 2;
            sequenceTargets[sample * 2 + label] = 1.0;
            for (int time = 0; time < sequence; time++) {
                for (int feature = 0; feature < sequenceFeatures; feature++) {
                    sequenceValues[(sample * sequence + time) * sequenceFeatures + feature]
                            = 0.1 * sequenceRandom.nextGaussian();
                }
            }
            sequenceValues[(sample * sequence) * sequenceFeatures + label] = 1.0;
        }
        SequenceNetwork network = SequenceNetwork.builder()
                .add(LSTM.builder().hiddenSize(8).seed(2).build())
                .pooling(SequencePool.Pooling.LAST)
                .head(Dense.builder().units(2).activation(Activation.SOFTMAX).seed(3).build())
                .optimizer(Optimizer.ADAM.withLearningRate(0.01))
                .gradientClip(5.0).build();
        Tensor<BSF> sequenceInput = Tensor.ofBSF(samples, sequence, sequenceFeatures, sequenceValues);
        Tensor<basalt.core.BC> sequenceTarget = Tensor.ofBC(samples, 2, sequenceTargets);
        double[] losses = network.fit(sequenceInput, sequenceTarget, 120).losses();
        List<String> sequenceRows = new ArrayList<>();
        for (int epoch = 0; epoch < losses.length; epoch++) {
            sequenceRows.add(String.format(Locale.ROOT, "%d,%.8f", epoch, losses[epoch]));
        }
        write(output.resolve("convergence_sequence.csv"), "epoch,loss", sequenceRows);
    }

    /**
     * Measures the norm of the gradient arriving at timestep zero when only the last timestep
     * receives upstream gradient. The LSTM forget gate is saturated at five, the RNN uses the same
     * weight scale; everything else is identical.
     */
    private static void gradientDecay(Path output) {
        int hidden = 4;
        int maximum = 40;
        List<String> rows = new ArrayList<>();
        for (int sequence = 1; sequence <= maximum; sequence++) {
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
            double lstmNorm = Math.abs(lstm.backward(upstream).data()[0]);

            VanillaRNN rnn = VanillaRNN.builder().hiddenSize(hidden).seed(1).build();
            rnn.forward(zeros);
            for (ParameterRef parameter : rnn.parameters()) {
                Arrays.fill(parameter.values(), 0.1);
            }
            rnn.forward(zeros);
            double rnnNorm = Math.abs(rnn.backward(upstream).data()[0]);
            rows.add(String.format(Locale.ROOT, "%d,%.12e,%.12e", sequence, lstmNorm, rnnNorm));
        }
        write(output.resolve("gradient_decay.csv"), "timestep,lstm_norm,rnn_norm", rows);
    }

    // ---------------------------------------------------------------- utilities

    private static Tensor<BSF> randomSequence(int batch, int sequence, int features, long seed) {
        Random random = new Random(seed);
        double[] values = new double[batch * sequence * features];
        for (int i = 0; i < values.length; i++) {
            values[i] = random.nextDouble() * 2.0 - 1.0;
        }
        return Tensor.ofBSF(batch, sequence, features, values);
    }

    private static Matrix randomMatrix(int rows, int columns, long seed) {
        Random random = new Random(seed);
        double[] values = new double[rows * columns];
        for (int i = 0; i < values.length; i++) {
            values[i] = random.nextDouble() * 2.0 - 1.0;
        }
        return new Matrix(rows, columns, values);
    }

    private static double medianMs(Runnable task, int warmup, int repetitions) {
        for (int i = 0; i < warmup; i++) {
            task.run();
        }
        double[] samples = new double[repetitions];
        for (int i = 0; i < repetitions; i++) {
            long start = System.nanoTime();
            task.run();
            samples[i] = (System.nanoTime() - start) / 1e6;
        }
        Arrays.sort(samples);
        return samples[samples.length / 2];
    }

    private static double medianSeconds(Runnable task, int warmup, int repetitions) {
        return medianMs(task, warmup, repetitions) / 1000.0;
    }

    @SuppressWarnings("deprecation")
    private static long allocatedBytes(Runnable task) {
        try {
            com.sun.management.ThreadMXBean bean =
                    (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
            long threadId = Thread.currentThread().getId();
            long before = bean.getThreadAllocatedBytes(threadId);
            task.run();
            long after = bean.getThreadAllocatedBytes(threadId);
            return Math.max(0L, after - before);
        } catch (UnsupportedOperationException | ClassCastException failure) {
            return -1L;
        }
    }

    private static void write(Path path, String header, List<String> rows) {
        StringBuilder builder = new StringBuilder(header).append('\n');
        for (String row : rows) {
            builder.append(row).append('\n');
        }
        try {
            Files.writeString(path, builder.toString());
        } catch (Exception failure) {
            throw new IllegalStateException("Cannot write " + path, failure);
        }
    }

    @FunctionalInterface
    private interface DoubleSupplier {
        double get();
    }
}
