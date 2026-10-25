package jsl.tests;

import jsl.core.Matrix;
import jsl.core.Persistable;
import jsl.io.ModelCodec;
import jsl.ml.BoostingObjective;
import jsl.ml.Estimator;
import jsl.ml.GaussianNaiveBayes;
import jsl.ml.HistogramGradientBoosting;
import jsl.ml.KMeans;
import jsl.ml.PCA;
import jsl.ml.RandomForest;
import jsl.ml.TreeTask;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.function.Supplier;

/**
 * Runs the JSL side of the scikit-learn comparison on the exact JSLD files produced by
 * {@code report/compare_sklearn.py}. Results are written to {@code build/compare/jsl.csv}.
 *
 * <p>Hyperparameters are aligned with the Python script: 100 trees / depth 8 / sqrt features for
 * the forests, 100 iterations / 31 leaves / depth 6 / 64 bins / L2=1 for boosting, 8 clusters with
 * 4 K-Means++ restarts, 8 PCA components. Every model is warmed up once before five timed fits;
 * the median is reported.</p>
 */
public final class SklearnComparison {
    private static final int REPETITIONS = 5;
    private static final int THREADS = 8;
    private static volatile double sink;

    private SklearnComparison() {
    }

    public static void main(String[] args) throws Exception {
        Path directory = Path.of(System.getProperty("user.dir"), "build", "compare");
        Dataset classification = load(directory.resolve("clf_train.jsld"));
        Dataset classificationTest = load(directory.resolve("clf_test.jsld"));
        Dataset regression = load(directory.resolve("reg_train.jsld"));
        Dataset regressionTest = load(directory.resolve("reg_test.jsld"));
        List<String> rows = new ArrayList<>();

        Measurement forest = measure(() -> RandomForest.builder()
                        .trees(100).maxDepth(8).minSamplesLeaf(1).bins(64)
                        .threads(THREADS).seed(42).build(),
                classification.features, classification.labels, classificationTest.features);
        record(directory, rows, "clf", "RandomForest", forest,
                accuracy(classificationTest.labels,
                        forest.model.predict(classificationTest.features)),
                "accuracy");

        Measurement boosting = measure(() -> HistogramGradientBoosting.builder()
                        .algorithm(jsl.ml.BoostingAlgorithm.LIGHTGBM)
                        .objective(BoostingObjective.LOGISTIC)
                        .iterations(100).learningRate(0.1).maxDepth(6).maxLeaves(31)
                        .bins(64).minSamplesLeaf(20).lambda(1.0).threads(THREADS).seed(42).build(),
                classification.features, classification.labels, classificationTest.features);
        record(directory, rows, "clf", "HistGradientBoosting", boosting,
                accuracy(classificationTest.labels,
                        boosting.model.predict(classificationTest.features)),
                "accuracy");

        Measurement bayes = measure(() -> GaussianNaiveBayes.builder().build(),
                classification.features, classification.labels, classificationTest.features);
        record(directory, rows, "clf", "GaussianNB", bayes,
                accuracy(classificationTest.labels,
                        bayes.model.predict(classificationTest.features)),
                "accuracy");

        Measurement kmeans = measure(() -> KMeans.builder().clusters(8).restarts(4)
                        .maxIterations(300).threads(THREADS).seed(42).build(),
                classification.features, classification.labels, classificationTest.features);
        record(directory, rows, "unsupervised", "KMeans", kmeans,
                ((KMeans) kmeans.model).inertia(), "inertia");

        Measurement pca = measure(() -> PCA.builder().components(8).build(),
                classification.features, classification.labels, classificationTest.features);
        double ratioSum = 0.0;
        for (double ratio : ((PCA) pca.model).explainedVarianceRatio()) {
            ratioSum += ratio;
        }
        record(directory, rows, "unsupervised", "PCA", pca, ratioSum, "evr_sum");

        Measurement ridge = measure(() -> jsl.ml.Ridge.builder().alpha(1.0).build(),
                regression.features, regression.labels, regressionTest.features);
        record(directory, rows, "reg", "Ridge", ridge,
                jsl.ml.Metrics.r2(regressionTest.labels,
                        ridge.model.predict(regressionTest.features)), "r2");

        Measurement lasso = measure(() -> jsl.ml.Lasso.builder().alpha(0.1)
                        .maxIterations(2000).build(),
                regression.features, regression.labels, regressionTest.features);
        record(directory, rows, "reg", "Lasso", lasso,
                jsl.ml.Metrics.r2(regressionTest.labels,
                        lasso.model.predict(regressionTest.features)), "r2");

        Measurement svm = measure(() -> jsl.ml.LinearSVM.builder().lambda(0.01)
                        .epochs(20).seed(42).build(),
                classification.features, classification.labels, classificationTest.features);
        record(directory, rows, "clf", "LinearSVM", svm,
                accuracy(classificationTest.labels,
                        svm.model.predict(classificationTest.features)), "accuracy");

        Measurement forestRegression = measure(() -> RandomForest.builder()
                        .task(TreeTask.REGRESSION)
                        .trees(100).maxDepth(8).minSamplesLeaf(1).bins(64)
                        .threads(THREADS).seed(42).build(),
                regression.features, regression.labels, regressionTest.features);
        record(directory, rows, "reg", "RandomForestRegressor", forestRegression,
                rSquared(regressionTest.labels,
                        forestRegression.model.predict(regressionTest.features)), "r2");

        Measurement boostingRegression = measure(() -> HistogramGradientBoosting.builder()
                        .algorithm(jsl.ml.BoostingAlgorithm.LIGHTGBM)
                        .objective(BoostingObjective.SQUARED_ERROR)
                        .iterations(100).learningRate(0.1).maxDepth(6).maxLeaves(31)
                        .bins(64).minSamplesLeaf(20).lambda(1.0).threads(THREADS).seed(42).build(),
                regression.features, regression.labels, regressionTest.features);
        record(directory, rows, "reg", "HistGradientBoostingRegressor", boostingRegression,
                rSquared(regressionTest.labels,
                        boostingRegression.model.predict(regressionTest.features)), "r2");

        Path path = directory.resolve("jsl.csv");
        StringBuilder builder = new StringBuilder(
                "task,model,fit_ms,predict_ms,score,score_name,model_bytes\n");
        for (String row : rows) {
            builder.append(row).append('\n');
        }
        Files.writeString(path, builder.toString());
        System.out.println("wrote " + path);
    }

    private static Measurement measure(Supplier<Estimator> factory, Matrix features,
                                       double[] labels, Matrix testFeatures) {
        Estimator warmup = factory.get();
        warmup.fit(features, labels);
        warmup.predict(testFeatures);
        double[] fitTimes = new double[REPETITIONS];
        double[] predictTimes = new double[REPETITIONS];
        Estimator last = null;
        for (int repetition = 0; repetition < REPETITIONS; repetition++) {
            last = factory.get();
            long start = System.nanoTime();
            last.fit(features, labels);
            fitTimes[repetition] = (System.nanoTime() - start) / 1e6;
            start = System.nanoTime();
            double[] prediction = last.predict(testFeatures);
            predictTimes[repetition] = (System.nanoTime() - start) / 1e6;
            sink += prediction[0];
        }
        Arrays.sort(fitTimes);
        Arrays.sort(predictTimes);
        return new Measurement(fitTimes[REPETITIONS / 2], predictTimes[REPETITIONS / 2], last);
    }

    private static void record(Path directory, List<String> rows, String task, String model,
                               Measurement measurement, double score, String scoreName) {
        long bytes = 0;
        if (measurement.model instanceof Persistable persistable) {
            try {
                Path path = directory.resolve("jsl-" + model + ".jsl");
                ModelCodec.save(path, persistable);
                bytes = Files.size(path);
            } catch (Exception failure) {
                throw new IllegalStateException(failure);
            }
        }
        rows.add(String.format(Locale.ROOT, "%s,%s,%.4f,%.4f,%.8f,%s,%d",
                task, model, measurement.fitMs, measurement.predictMs, score, scoreName, bytes));
        System.out.printf(Locale.ROOT,
                "jsl %-14s %-28s fit=%8.2f ms pred=%7.2f ms %s=%.5f size=%d%n",
                task, model, measurement.fitMs, measurement.predictMs, scoreName, score, bytes);
    }

    private static double accuracy(double[] expected, double[] actual) {
        int correct = 0;
        for (int index = 0; index < expected.length; index++) {
            if (expected[index] == actual[index]) {
                correct++;
            }
        }
        return (double) correct / expected.length;
    }

    private static double rSquared(double[] expected, double[] actual) {
        double mean = 0.0;
        for (double value : expected) {
            mean += value;
        }
        mean /= expected.length;
        double residual = 0.0;
        double total = 0.0;
        for (int index = 0; index < expected.length; index++) {
            residual += (expected[index] - actual[index]) * (expected[index] - actual[index]);
            total += (expected[index] - mean) * (expected[index] - mean);
        }
        return 1.0 - residual / total;
    }

    private static Dataset load(Path path) throws Exception {
        byte[] bytes = Files.readAllBytes(path);
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        int magic = buffer.getInt();
        int version = buffer.getInt();
        int features = buffer.getInt();
        int reserved = buffer.getInt();
        if (magic != 0x4A534C44 || version != 1 || reserved != 0 || features <= 0) {
            throw new IllegalArgumentException("Invalid JSLD file " + path);
        }
        int recordValues = features + 1;
        int rows = (bytes.length - 16) / (recordValues * Double.BYTES);
        double[] values = new double[rows * features];
        double[] labels = new double[rows];
        DoubleBuffer doubles = buffer.asDoubleBuffer();
        for (int row = 0; row < rows; row++) {
            doubles.get(values, row * features, features);
            labels[row] = doubles.get();
        }
        return new Dataset(new Matrix(rows, features, values), labels);
    }

    private record Dataset(Matrix features, double[] labels) {
    }

    private record Measurement(double fitMs, double predictMs, Estimator model) {
    }
}
