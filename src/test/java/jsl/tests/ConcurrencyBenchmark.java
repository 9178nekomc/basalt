package jsl.tests;

import jsl.concurrent.InferenceEngine;
import jsl.core.Matrix;
import jsl.ml.Estimator;
import jsl.ml.GaussianNaiveBayes;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Compares sequential, fixed platform-thread and virtual-thread orchestration for many small
 * inference requests, plus an I/O-bound scenario. Writes {@code build/compare/vthreads.csv}.
 */
public final class ConcurrencyBenchmark {
    private static final int TASKS = 4_000;
    private static final int BATCH_ROWS = 16;
    private static final int FEATURES = 32;
    private static final int PARALLELISM = 8;

    private ConcurrencyBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Path directory = Path.of(System.getProperty("user.dir"), "build", "compare");
        Files.createDirectories(directory);
        Estimator model = trainModel();
        List<Matrix> batches = batches();
        List<String> rows = new ArrayList<>();

        warmup(model, batches);

        rows.add(measureCpu("sequential", () -> {
            for (Matrix batch : batches) {
                model.predict(batch);
            }
            return null;
        }));
        rows.add(measureCpu("platform-8", () -> runPlatform(model, batches)));
        rows.add(measureCpu("virtual-8", () -> runVirtual(model, batches)));

        rows.add(measureIo("sequential", () -> {
            for (int i = 0; i < TASKS; i++) {
                Thread.sleep(1);
            }
            return null;
        }));
        rows.add(measureIo("platform-8", () -> runPlatformIo()));
        rows.add(measureIo("virtual", () -> runVirtualIo()));

        Path path = directory.resolve("vthreads.csv");
        StringBuilder builder = new StringBuilder(
                "scenario,mode,tasks,wall_ms,tasks_per_s,avg_ms\n");
        for (String row : rows) {
            builder.append(row).append('\n');
        }
        Files.writeString(path, builder.toString());
        System.out.println("wrote " + path);
    }

    private static Estimator trainModel() {
        int rows = 500;
        Random random = new Random(3);
        double[] values = new double[rows * FEATURES];
        double[] labels = new double[rows];
        for (int row = 0; row < rows; row++) {
            double signal = 0.0;
            for (int feature = 0; feature < FEATURES; feature++) {
                double value = random.nextGaussian();
                values[row * FEATURES + feature] = value;
                signal += value * (feature % 3 - 1);
            }
            labels[row] = signal > 0 ? 1 : 0;
        }
        return GaussianNaiveBayes.builder().build()
                .fit(new Matrix(rows, FEATURES, values), labels);
    }

    private static List<Matrix> batches() {
        Random random = new Random(5);
        List<Matrix> result = new ArrayList<>(TASKS);
        for (int task = 0; task < TASKS; task++) {
            double[] values = new double[BATCH_ROWS * FEATURES];
            for (int i = 0; i < values.length; i++) {
                values[i] = random.nextGaussian();
            }
            result.add(new Matrix(BATCH_ROWS, FEATURES, values));
        }
        return result;
    }

    private static void warmup(Estimator model, List<Matrix> batches) {
        for (int i = 0; i < 200; i++) {
            model.predict(batches.get(i % batches.size()));
        }
    }

    private static Void runPlatform(Estimator model, List<Matrix> batches) throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(PARALLELISM);
        try {
            List<Future<double[]>> futures = new ArrayList<>(batches.size());
            for (Matrix batch : batches) {
                futures.add(pool.submit(() -> model.predict(batch)));
            }
            for (Future<double[]> future : futures) {
                future.get();
            }
        } finally {
            pool.shutdown();
        }
        return null;
    }

    private static Void runVirtual(Estimator model, List<Matrix> batches) {
        try (InferenceEngine engine = new InferenceEngine(PARALLELISM)) {
            engine.predict(model, batches);
        }
        return null;
    }

    private static Void runPlatformIo() throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(PARALLELISM);
        try {
            List<Future<?>> futures = new ArrayList<>(TASKS);
            for (int i = 0; i < TASKS; i++) {
                futures.add(pool.submit(() -> {
                    Thread.sleep(1);
                    return null;
                }));
            }
            for (Future<?> future : futures) {
                future.get();
            }
        } finally {
            pool.shutdown();
        }
        return null;
    }

    private static Void runVirtualIo() {
        List<Callable<Void>> tasks = new ArrayList<>(TASKS);
        for (int i = 0; i < TASKS; i++) {
            tasks.add(() -> {
                Thread.sleep(1);
                return null;
            });
        }
        try (InferenceEngine engine = new InferenceEngine(PARALLELISM)) {
            engine.runIo(tasks);
        }
        return null;
    }

    private static String measureCpu(String mode, Callable<Void> action) throws Exception {
        long start = System.nanoTime();
        action.call();
        double wall = (System.nanoTime() - start) / 1e6;
        return String.format(Locale.ROOT, "cpu,%s,%d,%.3f,%.1f,%.3f",
                mode, TASKS, wall, TASKS / (wall / 1000.0), wall / TASKS);
    }

    private static String measureIo(String mode, Callable<Void> action) throws Exception {
        long start = System.nanoTime();
        action.call();
        double wall = (System.nanoTime() - start) / 1e6;
        return String.format(Locale.ROOT, "io,%s,%d,%.3f,%.1f,%.3f",
                mode, TASKS, wall, TASKS / (wall / 1000.0), wall / TASKS);
    }
}
