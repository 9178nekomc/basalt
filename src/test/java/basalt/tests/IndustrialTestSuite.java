package basalt.tests;

import basalt.core.Matrix;
import basalt.core.ModelFormatException;
import basalt.core.ModelKind;
import basalt.core.Persistable;
import basalt.data.DataLoader;
import basalt.data.Normalization;
import basalt.io.ModelCodec;
import basalt.ml.BoostingAlgorithm;
import basalt.ml.BoostingObjective;
import basalt.ml.DecisionTree;
import basalt.ml.EigenSolver;
import basalt.ml.Estimator;
import basalt.ml.GaussianNaiveBayes;
import basalt.ml.HistogramGradientBoosting;
import basalt.ml.KMeans;
import basalt.ml.PCA;
import basalt.ml.RandomForest;
import basalt.ml.TreeTask;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/** Numerical, convergence, persistence and data-pipeline acceptance tests for the ML layer. */
public final class IndustrialTestSuite {
    private static int passed;

    private IndustrialTestSuite() {
    }

    public static void main(String[] args) throws Exception {
        run("random forest classification, OOB and probabilities", IndustrialTestSuite::testRandomForest);
        run("random forest and tree regression", IndustrialTestSuite::testTreeRegression);
        run("XGBoost and LightGBM histogram cores", IndustrialTestSuite::testBoosting);
        run("gradient boosting early stopping", IndustrialTestSuite::testEarlyStopping);
        run("K-Means++ clustering and restarts", IndustrialTestSuite::testKMeans);
        run("PCA Jacobi and Gram duality", IndustrialTestSuite::testPca);
        run("Gaussian Naive Bayes probabilities and scale invariance",
                IndustrialTestSuite::testGaussianNaiveBayes);
        run("DataLoader sequential, mapped and shuffled epochs", IndustrialTestSuite::testDataLoader);
        run("all-model .bsl bit-exact serialization", IndustrialTestSuite::testSerialization);
        System.out.println("PASS: " + passed + " industrial test groups");
    }

    private static void testRandomForest() {
        Matrix features = classificationData();
        double[] labels = classificationLabels(features.rows());
        RandomForest forest = RandomForest.builder().trees(21).maxDepth(5)
                .threads(2).seed(11).oob(true).build().fit(features, labels);
        double accuracy = accuracy(labels, forest.predict(features));
        check(accuracy >= 0.95, "random forest training accuracy " + accuracy);
        check(forest.oobScore() > 0.5, "OOB accuracy " + forest.oobScore());
        Matrix probabilities = forest.predictProbability(features);
        for (int row = 0; row < probabilities.rows(); row++) {
            assertClose(1.0, probabilities.get(row, 0) + probabilities.get(row, 1),
                    1e-12, "forest probability normalization");
        }
        double[] importances = forest.featureImportances();
        check(importances.length == 2, "importance length");
        assertClose(1.0, importances[0] + importances[1], 1e-9, "importance normalization");
    }

    private static void testTreeRegression() {
        int rows = 160;
        double[] x = new double[rows * 2];
        double[] y = new double[rows];
        for (int row = 0; row < rows; row++) {
            double first = -2.0 + 4.0 * row / (rows - 1);
            double second = ((row * 37) % 101) / 100.0 - 0.5;
            x[row * 2] = first;
            x[row * 2 + 1] = second;
            y[row] = 2.0 * first - second + 0.5;
        }
        Matrix data = new Matrix(rows, 2, x);
        DecisionTree tree = DecisionTree.builder().task(TreeTask.REGRESSION)
                .maxDepth(6).bins(64).build().fit(data, y);
        double[] prediction = tree.predict(data);
        check(rSquared(y, prediction) > 0.98, "regression tree R2 " + rSquared(y, prediction));

        RandomForest forest = RandomForest.builder().task(TreeTask.REGRESSION)
                .trees(25).maxDepth(6).threads(2).oob(true).build().fit(data, y);
        double[] forestPrediction = forest.predict(data);
        check(rSquared(y, forestPrediction) > 0.95,
                "regression forest R2 " + rSquared(y, forestPrediction));
        check(forest.oobScore() > 0.8, "regression OOB R2 " + forest.oobScore());
    }

    private static void testBoosting() {
        Matrix features = classificationData();
        double[] labels = classificationLabels(features.rows());
        for (BoostingAlgorithm algorithm : BoostingAlgorithm.values()) {
            HistogramGradientBoosting model = HistogramGradientBoosting.builder()
                    .algorithm(algorithm).objective(BoostingObjective.LOGISTIC)
                    .iterations(25).learningRate(0.3).maxDepth(4).maxLeaves(8)
                    .bins(16).threads(2).build().fit(features, labels);
            double accuracy = accuracy(labels, model.predict(features));
            check(accuracy >= 0.95, algorithm + " accuracy " + accuracy);
            double[] history = model.lossHistory();
            check(history[history.length - 1] < history[0], algorithm + " loss must decrease");
        }
        double[] x = new double[60];
        double[] y = new double[60];
        for (int i = 0; i < 60; i++) {
            x[i] = -1.0 + 2.0 * i / 59.0;
            y[i] = 2.0 * x[i] + 1.0;
        }
        double[] manyBins = new double[129];
        double[] manyBinTargets = new double[129];
        for (int i = 0; i < manyBins.length; i++) {
            manyBins[i] = i;
            manyBinTargets[i] = i;
        }
        HistogramGradientBoosting unsignedBins = HistogramGradientBoosting.builder()
                .iterations(2).bins(129).maxDepth(2).threads(2).build()
                .fit(new Matrix(129, 1, manyBins), manyBinTargets);
        check(unsignedBins.predict(new Matrix(129, 1, manyBins)).length == 129,
                "unsigned bins above 127");

        HistogramGradientBoosting regression = HistogramGradientBoosting.builder()
                .objective(BoostingObjective.SQUARED_ERROR).iterations(40)
                .learningRate(0.2).maxDepth(3).bins(16).threads(2).build()
                .fit(new Matrix(60, 1, x), y);
        double[] prediction = regression.predict(new Matrix(60, 1, x));
        for (int i = 0; i < y.length; i++) {
            assertClose(y[i], prediction[i], 0.25, "boosting regression " + i);
        }
    }

    private static void testEarlyStopping() {
        int rows = 320;
        int features = 5;
        Random random = new Random(31);
        double[] values = new double[rows * features];
        double[] targets = new double[rows];
        for (int row = 0; row < rows; row++) {
            double signal = 0.0;
            for (int feature = 0; feature < features; feature++) {
                double value = random.nextGaussian();
                values[row * features + feature] = value;
                signal += Math.sin(value * (feature + 1));
            }
            targets[row] = signal + 0.4 * random.nextGaussian();
        }
        HistogramGradientBoosting model = HistogramGradientBoosting.builder()
                .iterations(300).learningRate(0.1).maxDepth(4).bins(32)
                .validationFraction(0.25).earlyStoppingRounds(5).threads(2)
                .featureFraction(0.8).seed(9).build()
                .fit(new Matrix(rows, features, values), targets);
        check(model.treeCount() < 300, "early stopping must truncate the ensemble");
        check(model.treeCount() == model.lossHistory().length, "loss history must match trees");
        check(model.validationHistory().length == model.treeCount(),
                "validation history must match trees");
        double[] validation = model.validationHistory();
        check(validation.length == 0 || validation[validation.length - 1] <= validation[0],
                "validation loss must not increase overall");
    }

    private static void testKMeans() {
        Matrix data = Matrix.fromRows(new double[][]{
                {-5, -5}, {-4.8, -5.1}, {-5.2, -4.9}, {-4.9, -5.2},
                {5, 5}, {5.1, 4.8}, {4.9, 5.2}, {5.2, 4.9}});
        KMeans means = KMeans.builder().clusters(2).seed(3).build().fit(data);
        double[] clusters = means.predict(data);
        check(clusters[0] == clusters[1] && clusters[1] == clusters[2]
                && clusters[2] == clusters[3], "negative cluster purity");
        check(clusters[4] == clusters[5] && clusters[5] == clusters[6]
                && clusters[6] == clusters[7], "positive cluster purity");
        check(clusters[0] != clusters[4], "clusters must be distinct");
        check(means.inertia() < 0.5, "KMeans inertia " + means.inertia());
        KMeans duplicates = KMeans.builder().clusters(3).maxIterations(7).build()
                .fit(Matrix.fromRows(new double[][]{{1}, {1}, {1}, {1}, {1}, {1}}));
        assertClose(0.0, duplicates.inertia(), 0.0, "duplicate-point inertia");
        check(duplicates.iterations() <= 7, "iteration counter must not exceed maxIterations");

        Random random = new Random(17);
        double[] values = new double[600 * 2];
        for (int row = 0; row < 600; row++) {
            int cluster = row % 4;
            values[row * 2] = cluster * 8.0 + random.nextGaussian();
            values[row * 2 + 1] = cluster * 8.0 + random.nextGaussian();
        }
        Matrix spread = new Matrix(600, 2, values);
        double single = KMeans.builder().clusters(4).restarts(1).seed(5).threads(2)
                .build().fit(spread).inertia();
        KMeans restarted = KMeans.builder().clusters(4).restarts(6).seed(5).threads(2)
                .build().fit(spread);
        check(restarted.inertia() <= single + 1e-9,
                "restarts must not worsen inertia: " + restarted.inertia() + " vs " + single);
        check(restarted.restarts() == 6, "restart counter");
        double[] distances = restarted.transform(spread);
        assertClose(restarted.inertia(), Arrays.stream(distances).sum(), 1e-6,
                "transform must match inertia");
    }

    private static void testPca() {
        Matrix data = Matrix.fromRows(new double[][]{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}});
        PCA pca = PCA.builder().components(1).build().fit(data);
        Matrix transformed = pca.transform(data);
        Matrix restored = pca.inverseTransform(transformed);
        assertArray(data.data(), restored.data(), 1e-10, "rank-one PCA reconstruction");
        assertClose(1.0, pca.explainedVarianceRatio()[0], 1e-12, "explained variance ratio");
        check(pca.predict(data).length == data.rows(), "PCA Estimator flattened output");

        int rows = 10;
        int features = 24;
        Random random = new Random(23);
        double[][] scores = new double[rows][3];
        double[][] loadings = new double[3][features];
        for (int row = 0; row < rows; row++) {
            for (int component = 0; component < 3; component++) {
                scores[row][component] = random.nextGaussian();
            }
        }
        for (int component = 0; component < 3; component++) {
            for (int feature = 0; feature < features; feature++) {
                loadings[component][feature] = random.nextGaussian();
            }
        }
        double[] values = new double[rows * features];
        for (int row = 0; row < rows; row++) {
            for (int feature = 0; feature < features; feature++) {
                double value = 0.0;
                for (int component = 0; component < 3; component++) {
                    value += scores[row][component] * loadings[component][feature];
                }
                values[row * features + feature] = value;
            }
        }
        Matrix wide = new Matrix(rows, features, values);
        PCA gram = PCA.builder().components(3).solver(EigenSolver.GRAM).build().fit(wide);
        PCA jacobi = PCA.builder().components(3).solver(EigenSolver.JACOBI).build().fit(wide);
        check(gram.solver() == EigenSolver.GRAM, "explicit Gram solver");
        check(PCA.builder().components(3).build().fit(wide).solver() == EigenSolver.GRAM,
                "AUTO must select Gram for wide data");
        Matrix gramProjection = gram.transform(wide);
        Matrix jacobiProjection = jacobi.transform(wide);
        for (int row = 0; row < rows; row++) {
            for (int component = 0; component < 3; component++) {
                assertClose(Math.abs(jacobiProjection.get(row, component)),
                        Math.abs(gramProjection.get(row, component)), 1e-9,
                        "Gram/Jacobi projection " + row + "," + component);
            }
        }
        Matrix reconstruction = gram.inverseTransform(gramProjection);
        assertArray(values, reconstruction.data(), 1e-9, "rank-three reconstruction");
        double ratioSum = 0.0;
        for (double ratio : gram.explainedVarianceRatio()) {
            ratioSum += ratio;
        }
        check(ratioSum > 0.999, "three components must explain rank-three data: " + ratioSum);

        PCA whiten = PCA.builder().components(3).solver(EigenSolver.GRAM).whiten(true)
                .build().fit(wide);
        Matrix white = whiten.transform(wide);
        for (int component = 0; component < 3; component++) {
            double sum = 0.0;
            double mean = 0.0;
            for (int row = 0; row < rows; row++) {
                mean += white.get(row, component);
            }
            mean /= rows;
            for (int row = 0; row < rows; row++) {
                double delta = white.get(row, component) - mean;
                sum += delta * delta;
            }
            assertClose(1.0, sum / (rows - 1), 1e-9, "whitened variance " + component);
        }
    }

    private static void testGaussianNaiveBayes() {
        Matrix features = Matrix.fromRows(new double[][]{
                {0, 0.2}, {0.2, -0.1}, {-0.2, 0.1}, {0.1, 0},
                {5, 5.2}, {5.2, 4.9}, {4.8, 5.1}, {5.1, 5}});
        double[] labels = {0, 0, 0, 0, 1, 1, 1, 1};
        GaussianNaiveBayes model = GaussianNaiveBayes.builder().build().fit(features, labels);
        assertArray(labels, model.predict(features), 0.0, "Gaussian NB predictions");
        Matrix probabilities = model.predictProbability(features);
        for (int row = 0; row < probabilities.rows(); row++) {
            assertClose(1.0, probabilities.get(row, 0) + probabilities.get(row, 1),
                    1e-12, "Gaussian NB probability normalization");
        }
        Matrix extreme = model.predictProbability(Matrix.fromRows(new double[][]{{1e200, 0}}));
        assertClose(1.0, extreme.get(0, 0) + extreme.get(0, 1), 0.0,
                "underflow probability fallback");

        double[] scaledValues = features.data().clone();
        for (int i = 0; i < scaledValues.length; i++) {
            scaledValues[i] *= 1e6;
        }
        Matrix scaled = new Matrix(features.rows(), features.columns(), scaledValues);
        GaussianNaiveBayes scaledModel = GaussianNaiveBayes.builder().build().fit(scaled, labels);
        assertArray(labels, scaledModel.predict(scaled), 0.0,
                "variance smoothing must be scale invariant");
    }

    private static void testDataLoader() throws Exception {
        Path directory = Path.of(System.getProperty("user.dir"), "build");
        Files.createDirectories(directory);
        Path path = directory.resolve("test-data.bsld");
        Matrix data = Matrix.fromRows(new double[][]{{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}});
        double[] labels = {0, 1, 0, 1, 0};
        DataLoader.write(path, data, labels);
        check(DataLoader.rowCount(path, 2) == 5, "row count");
        Normalization normalization = DataLoader.estimateNormalization(path, 2);
        assertArray(new double[]{3, 30}, normalization.means(), 1e-12, "loader means");
        assertArray(new double[]{Math.sqrt(2.5), Math.sqrt(250)},
                normalization.standardDeviations(), 1e-12, "loader standard deviations");
        int rows = 0;
        double[] featureIdentity;
        double[] labelIdentity;
        try (DataLoader.BatchStream stream = DataLoader.builder(path).featureCount(2)
                .batchSize(2).normalization(normalization).open()) {
            DataLoader.Batch batch = stream.batch();
            featureIdentity = batch.features();
            labelIdentity = batch.labels();
            check(stream.next(), "first batch");
            check(stream.batch() == batch && batch.features() == featureIdentity
                    && batch.labels() == labelIdentity, "batch identity must be stable");
            double sum = 0.0;
            double sumSquares = 0.0;
            while (true) {
                for (int row = 0; row < batch.rows(); row++) {
                    for (int feature = 0; feature < 2; feature++) {
                        double value = batch.feature(row, feature);
                        sum += value;
                        sumSquares += value * value;
                    }
                    assertClose(labels[rows + row], batch.label(row), 0.0, "streamed label");
                }
                rows += batch.rows();
                check(stream.batch() == batch && batch.features() == featureIdentity
                        && batch.labels() == labelIdentity, "no replacement buffers");
                if (!stream.next()) {
                    break;
                }
            }
            check(rows == 5, "all rows streamed");
            assertClose(0.0, sum / 10.0, 1e-12, "normalized streaming mean");
            assertClose(0.8, sumSquares / 10.0, 1e-12, "normalized streaming second moment");

            stream.rewind();
            int replayed = 0;
            while (stream.next()) {
                replayed += batch.rows();
            }
            check(replayed == 5, "rewind must replay every row");
        }

        try (DataLoader.MappedBatchStream mapped = DataLoader.builder(path).featureCount(2)
                .batchSize(2).shuffle(true).seed(7).normalization(normalization).openMapped()) {
            check(mapped.totalRows() == 5, "mapped row count");
            double[] features = mapped.batch().features();
            double[] targets = mapped.batch().labels();
            Set<Double> firstEpoch = new HashSet<>();
            double sum = 0.0;
            while (mapped.next()) {
                check(mapped.batch().features() == features && mapped.batch().labels() == targets,
                        "mapped buffers must be stable");
                for (int row = 0; row < mapped.batch().rows(); row++) {
                    firstEpoch.add(mapped.batch().label(row));
                    sum += mapped.batch().feature(row, 0);
                }
            }
            check(firstEpoch.size() == 2, "shuffle must preserve all labels");
            assertClose(0.0, sum / 5.0, 1e-12, "mapped normalized mean");
            mapped.rewind();
            check(mapped.epoch() == 1, "epoch counter");
            Set<Double> secondEpoch = new HashSet<>();
            while (mapped.next()) {
                for (int row = 0; row < mapped.batch().rows(); row++) {
                    secondEpoch.add(mapped.batch().label(row));
                }
            }
            check(secondEpoch.equals(firstEpoch), "shuffled epochs must see the same rows");
        }
    }

    private static void testSerialization() throws Exception {
        Path directory = Path.of(System.getProperty("user.dir"), "build");
        Files.createDirectories(directory);
        int rows = 120;
        int features = 4;
        Random random = new Random(5);
        double[] values = new double[rows * features];
        double[] targets = new double[rows];
        for (int row = 0; row < rows; row++) {
            for (int feature = 0; feature < features; feature++) {
                values[row * features + feature] = random.nextGaussian();
            }
            targets[row] = 2.0 * values[row * features] - values[row * features + 1]
                    + 0.05 * random.nextGaussian();
        }
        Matrix data = new Matrix(rows, features, values);
        double[] labels = new double[rows];
        for (int row = 0; row < rows; row++) {
            labels[row] = targets[row] > 0 ? 1 : 0;
        }

        HistogramGradientBoosting boosting = HistogramGradientBoosting.builder()
                .iterations(15).maxDepth(3).bins(16).threads(2).build().fit(data, targets);
        roundTrip(directory, "boosting", boosting, data);

        RandomForest forest = RandomForest.builder().task(TreeTask.REGRESSION)
                .trees(9).maxDepth(4).threads(2).build().fit(data, targets);
        roundTrip(directory, "forest", forest, data);

        KMeans kmeans = KMeans.builder().clusters(3).threads(2).build().fit(data);
        roundTrip(directory, "kmeans", kmeans, data);

        PCA pca = PCA.builder().components(2).build().fit(data);
        roundTrip(directory, "pca", pca, data);

        GaussianNaiveBayes bayes = GaussianNaiveBayes.builder().build().fit(data, labels);
        roundTrip(directory, "bayes", bayes, data);

        Path valid = directory.resolve("corrupt-source.bsl");
        ModelCodec.save(valid, kmeans);
        byte[] bytes = Files.readAllBytes(valid);
        bytes[bytes.length - 1] ^= 0x5A;
        Path corrupt = directory.resolve("corrupt.bsl");
        Files.write(corrupt, bytes);
        expectThrows(ModelFormatException.class, () -> ModelCodec.load(corrupt));

        Path truncated = directory.resolve("truncated.bsl");
        Files.write(truncated, Arrays.copyOf(bytes, 10));
        expectThrows(ModelFormatException.class, () -> ModelCodec.load(truncated));

        expectThrows(ModelFormatException.class, () -> ModelCodec.load(valid, basalt.nn.CNN.class));
        check(ModelCodec.peekKind(valid) == ModelKind.KMEANS, "peek kind");
    }

    private static void roundTrip(Path directory, String name, Persistable model, Matrix data) {
        Path path = directory.resolve(name + ".bsl");
        ModelCodec.save(path, model);
        Persistable loaded = ModelCodec.load(path);
        check(loaded.kind() == model.kind(), name + " kind");
        double[] expected = ((Estimator) model).predict(data);
        double[] actual = ((Estimator) loaded).predict(data);
        check(Arrays.equals(expected, actual), name + " predictions must be bit-exact");
    }

    private static Matrix classificationData() {
        double[] values = new double[80 * 2];
        for (int row = 0; row < 80; row++) {
            double x = -2.0 + 4.0 * (row % 40) / 39.0;
            values[row * 2] = x;
            values[row * 2 + 1] = row < 40 ? -0.2 : 0.2;
        }
        return new Matrix(80, 2, values);
    }

    private static double[] classificationLabels(int rows) {
        double[] labels = new double[rows];
        for (int row = 0; row < rows; row++) {
            labels[row] = row < rows / 2 ? 0.0 : 1.0;
        }
        return labels;
    }

    private static double accuracy(double[] expected, double[] actual) {
        int correct = 0;
        for (int i = 0; i < expected.length; i++) {
            if (expected[i] == actual[i]) {
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
        for (int i = 0; i < expected.length; i++) {
            residual += (expected[i] - actual[i]) * (expected[i] - actual[i]);
            total += (expected[i] - mean) * (expected[i] - mean);
        }
        return 1.0 - residual / total;
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

    private static void assertArray(double[] expected, double[] actual, double tolerance, String message) {
        check(expected.length == actual.length, message + " length mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertClose(expected[i], actual[i], tolerance, message + " at " + i);
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

    private static void expectThrows(Class<? extends Throwable> type, Runnable action) {
        try {
            action.run();
        } catch (Throwable failure) {
            if (type.isInstance(failure)) {
                return;
            }
            throw new AssertionError("Expected " + type.getName() + " but got " + failure, failure);
        }
        throw new AssertionError("Expected " + type.getName() + " but nothing was thrown");
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
