package basalt.tests;

import basalt.core.Matrix;
import basalt.ml.GaussianNaiveBayes;
import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Measures the overhead of always-on production profiling.
 *
 * <p>Java has JFR (Java Flight Recorder) built into the JVM: a sampling profiler whose overhead is
 * designed for production (typically &lt;2%). Python's stdlib profiler (cProfile) instruments every
 * Python-level call, so its overhead depends on how fine-grained the application code is. This
 * harness runs the same model-inference workload twice in child JVMs - once plain, once with
 * {@code -XX:StartFlightRecording} - and writes both timings to
 * {@code build/compare/profiling_java.csv}.</p>
 */
public final class ProfilingBenchmark {
    private static final int REQUESTS = 4_000;
    private static final int BATCH_ROWS = 16;
    private static final int FEATURES = 32;

    private ProfilingBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 2 && "--child".equals(args[0])) {
            runWorkload(args[1]);
            return;
        }
        Path directory = Path.of(System.getProperty("user.dir"), "build", "compare");
        Files.createDirectories(directory);
        String javaHome = System.getProperty("java.home");
        String executable = System.getProperty("os.name", "").toLowerCase().contains("win")
                ? "java.exe" : "java";
        String javaBin = Path.of(javaHome, "bin", executable).toString();
        String root = Path.of(System.getProperty("user.dir")).toAbsolutePath().toString();
        String classpath = root + "/build/classes;" + root + "/build/test-classes";

        double baseline = runChild(javaBin, classpath, List.of(), "baseline");
        Path recording = directory.resolve("basalt-profiling.jfr");
        double jfr = runChild(javaBin, classpath,
                List.of("-XX:StartFlightRecording=filename=" + recording + ",settings=default"),
                "jfr");

        List<String> rows = new ArrayList<>();
        rows.add(String.format(Locale.ROOT, "Basalt + JFR (always-on),%.4f,%.4f,%.2f",
                baseline, jfr, overhead(baseline, jfr)));
        Path output = directory.resolve("profiling_java.csv");
        StringBuilder builder = new StringBuilder("scenario,baseline_ms,profiled_ms,overhead_percent\n");
        for (String row : rows) {
            builder.append(row).append('\n');
        }
        Files.writeString(output, builder.toString());
        System.out.printf(Locale.ROOT,
                "Basalt baseline=%.1f ms, JFR=%.1f ms, overhead=%.2f%%%n", baseline, jfr,
                overhead(baseline, jfr));
        System.out.println("wrote " + output);
        System.out.println("recording: " + recording + " (" + Files.size(recording) + " bytes)");
    }

    private static double runChild(String javaBin, String classpath, List<String> jvmFlags,
                                   String mode) throws Exception {
        List<String> command = new ArrayList<>();
        command.add(javaBin);
        command.add("--add-modules=jdk.incubator.vector");
        command.add("-Xmx2g");
        command.addAll(jvmFlags);
        command.add("-cp");
        command.add(classpath);
        command.add("basalt.tests.ProfilingBenchmark");
        command.add("--child");
        command.add(mode);
        Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
        double totalMs = Double.NaN;
        try (BufferedReader reader = process.inputReader()) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("RESULT ")) {
                    for (String part : line.split(" ")) {
                        if (part.startsWith("total_ms=")) {
                            totalMs = Double.parseDouble(part.substring("total_ms=".length()));
                        }
                    }
                }
            }
        }
        int exit = process.waitFor();
        if (exit != 0 || Double.isNaN(totalMs)) {
            throw new IllegalStateException("child JVM failed for mode " + mode);
        }
        return totalMs;
    }

    private static double overhead(double baseline, double profiled) {
        return (profiled - baseline) / baseline * 100.0;
    }

    /** Inference over 4000 pre-built batches with a per-request feature-scaling step. */
    private static void runWorkload(String mode) {
        GaussianNaiveBayes model = trainModel();
        List<double[]> batches = new ArrayList<>(REQUESTS);
        Random random = new Random(5);
        for (int request = 0; request < REQUESTS; request++) {
            double[] values = new double[BATCH_ROWS * FEATURES];
            for (int i = 0; i < values.length; i++) {
                values[i] = random.nextGaussian();
            }
            batches.add(values);
        }
        double sink = 0.0;
        for (int warmup = 0; warmup < 200; warmup++) {
            sink += orchestrate(model, batches.get(warmup));
        }
        long start = System.nanoTime();
        for (int request = 0; request < REQUESTS; request++) {
            sink += orchestrate(model, batches.get(request));
        }
        double totalMs = (System.nanoTime() - start) / 1e6;
        if (sink == Double.POSITIVE_INFINITY) {
            throw new IllegalStateException("workload collapsed");
        }
        System.out.printf(Locale.ROOT, "RESULT mode=%s total_ms=%.3f sink=%.3f%n",
                mode, totalMs, sink);
    }

    private static double orchestrate(GaussianNaiveBayes model, double[] batch) {
        double[] scaled = new double[batch.length];
        for (int i = 0; i < batch.length; i++) {
            scaled[i] = batch[i] * 0.5 + 0.1;
        }
        double[] prediction = model.predict(new Matrix(BATCH_ROWS, FEATURES, scaled));
        double score = 0.0;
        for (double value : prediction) {
            score += value;
        }
        return score;
    }

    private static GaussianNaiveBayes trainModel() {
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
}
