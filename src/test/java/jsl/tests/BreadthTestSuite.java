package jsl.tests;

import jsl.core.Matrix;
import jsl.ml.DBSCAN;
import jsl.ml.DataSplit;
import jsl.ml.Estimator;
import jsl.ml.JSL;
import jsl.ml.KNeighborsRegressor;
import jsl.ml.Lasso;
import jsl.ml.LinearSVM;
import jsl.ml.Metrics;
import jsl.ml.Ridge;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/** Acceptance tests for the breadth expansion: metrics, splits, linear models, SVM, DBSCAN. */
public final class BreadthTestSuite {
    private static int passed;

    private BreadthTestSuite() {
    }

    public static void main(String[] args) {
        run("classification and regression metrics", BreadthTestSuite::testMetrics);
        run("Ridge recovers linear coefficients", BreadthTestSuite::testRidge);
        run("Lasso produces sparse coefficients", BreadthTestSuite::testLasso);
        run("Linear SVM separates two classes", BreadthTestSuite::testSvm);
        run("DBSCAN finds two dense clusters", BreadthTestSuite::testDbscan);
        run("KNN regression predicts local means", BreadthTestSuite::testKnnRegression);
        run("holdout and k-fold splits are disjoint", BreadthTestSuite::testSplits);
        run("JSL facade dispatch and cross-validation", BreadthTestSuite::testFacade);
        System.out.println("PASS: " + passed + " breadth test groups");
    }

    private static void testMetrics() {
        double[] expected = {0, 1, 1, 0};
        double[] predicted = {0, 1, 0, 0};
        assertClose(0.75, Metrics.accuracy(expected, predicted), 1e-12, "accuracy");
        assertClose(1.0, Metrics.precision(expected, predicted, 1), 1e-12, "precision");
        assertClose(0.5, Metrics.recall(expected, predicted, 1), 1e-12, "recall");
        assertClose(2.0 / 3.0, Metrics.f1(expected, predicted, 1), 1e-12, "f1");
        int[][] matrix = Metrics.confusionMatrix(expected, predicted);
        check(matrix.length == 2 && matrix[0][0] == 2 && matrix[0][1] == 0
                && matrix[1][0] == 1 && matrix[1][1] == 1, "confusion matrix");
        double[] scores = {0.1, 0.9, 0.8, 0.2};
        assertClose(1.0, Metrics.rocAuc(expected, scores), 1e-12, "ROC AUC");
        double[] truth = {1, 2, 3};
        double[] estimate = {1.1, 1.9, 3.2};
        assertClose(0.97, Metrics.r2(truth, estimate), 1e-9, "R2");
        assertClose(Math.sqrt(0.02), Metrics.rmse(truth, estimate), 1e-9, "RMSE");
    }

    private static void testRidge() {
        int rows = 200;
        Random random = new Random(7);
        double[] values = new double[rows * 2];
        double[] targets = new double[rows];
        for (int row = 0; row < rows; row++) {
            double first = random.nextGaussian();
            double second = random.nextGaussian();
            values[row * 2] = first;
            values[row * 2 + 1] = second;
            targets[row] = 3.0 + 2.0 * first - 1.0 * second;
        }
        Ridge ridge = Ridge.builder().alpha(1e-6).build()
                .fit(new Matrix(rows, 2, values), targets);
        assertClose(2.0, ridge.weights()[0], 1e-3, "ridge weight 0");
        assertClose(-1.0, ridge.weights()[1], 1e-3, "ridge weight 1");
        assertClose(3.0, ridge.intercept(), 1e-3, "ridge intercept");
        assertClose(1.0, Metrics.r2(targets,
                ridge.predict(new Matrix(rows, 2, values))), 1e-9, "ridge R2");
    }

    private static void testLasso() {
        int rows = 300;
        Random random = new Random(11);
        double[] values = new double[rows * 3];
        double[] targets = new double[rows];
        for (int row = 0; row < rows; row++) {
            double signal = random.nextGaussian();
            values[row * 3] = signal;
            values[row * 3 + 1] = random.nextGaussian();
            values[row * 3 + 2] = random.nextGaussian();
            targets[row] = 3.0 * signal + 1.0;
        }
        Lasso lasso = Lasso.builder().alpha(0.01).maxIterations(2_000).build()
                .fit(new Matrix(rows, 3, values), targets);
        assertClose(3.0, lasso.weights()[0], 0.05, "lasso signal weight");
        assertClose(0.0, lasso.weights()[1], 0.02, "lasso noise weight 1");
        assertClose(0.0, lasso.weights()[2], 0.02, "lasso noise weight 2");
    }

    private static void testSvm() {
        Matrix features = Matrix.fromRows(new double[][]{
                {-3, -1}, {-2, -2}, {-1, -3}, {-2, -1},
                {3, 1}, {2, 2}, {1, 3}, {2, 1}});
        double[] labels = {0, 0, 0, 0, 1, 1, 1, 1};
        LinearSVM svm = LinearSVM.builder().lambda(0.01).epochs(40).seed(3).build()
                .fit(features, labels);
        assertArray(labels, svm.predict(features), 0.0, "SVM training accuracy");
    }

    private static void testDbscan() {
        Random random = new Random(13);
        double[] values = new double[40 * 2];
        for (int row = 0; row < 20; row++) {
            values[row * 2] = random.nextGaussian() * 0.1;
            values[row * 2 + 1] = random.nextGaussian() * 0.1;
        }
        for (int row = 20; row < 40; row++) {
            values[row * 2] = 10.0 + random.nextGaussian() * 0.1;
            values[row * 2 + 1] = 10.0 + random.nextGaussian() * 0.1;
        }
        DBSCAN dbscan = DBSCAN.builder().eps(0.5).minPoints(3).build()
                .fit(new Matrix(40, 2, values), null);
        check(dbscan.clusterCount() == 2, "DBSCAN cluster count " + dbscan.clusterCount());
        check(dbscan.noiseCount() == 0, "DBSCAN noise " + dbscan.noiseCount());
        int[] labels = dbscan.labels();
        for (int row = 1; row < 20; row++) {
            check(labels[row] == labels[0], "first blob purity");
        }
        for (int row = 21; row < 40; row++) {
            check(labels[row] == labels[20], "second blob purity");
        }
        check(labels[0] != labels[20], "blobs must be distinct");
    }

    private static void testKnnRegression() {
        Matrix features = Matrix.fromRows(new double[][]{{0}, {1}, {2}, {10}, {11}, {12}});
        double[] targets = {0, 2, 4, 20, 22, 24};
        KNeighborsRegressor regressor = KNeighborsRegressor.builder().k(3).build()
                .fit(features, targets);
        double[] prediction = regressor.predict(Matrix.fromRows(new double[][]{{1.2}, {10.8}}));
        assertClose(2.0, prediction[0], 1.0, "KNN regression near 1.2");
        assertClose(22.0, prediction[1], 1.0, "KNN regression near 10.8");
    }

    private static void testSplits() {
        int rows = 103;
        double[] values = new double[rows * 2];
        double[] targets = new double[rows];
        for (int row = 0; row < rows; row++) {
            values[row * 2] = row;
            values[row * 2 + 1] = -row;
            targets[row] = row;
        }
        Matrix features = new Matrix(rows, 2, values);
        DataSplit.Holdout holdout = DataSplit.holdout(features, targets, 0.25, 7);
        check(holdout.trainFeatures().rows() + holdout.testFeatures().rows() == rows,
                "holdout sizes");
        check(holdout.trainTargets().length == holdout.trainFeatures().rows(),
                "holdout train target length");
        check(holdout.testTargets().length == holdout.testFeatures().rows(),
                "holdout test target length");

        List<DataSplit.Fold> folds = DataSplit.kFold(features, targets, 5, 11);
        check(folds.size() == 5, "fold count");
        Set<Double> testRows = new HashSet<>();
        int testTotal = 0;
        for (DataSplit.Fold fold : folds) {
            check(fold.trainFeatures().rows() + fold.testFeatures().rows() == rows,
                    "fold sizes");
            for (int row = 0; row < fold.testFeatures().rows(); row++) {
                check(testRows.add(fold.testFeatures().get(row, 0)), "test rows must be disjoint");
                testTotal++;
            }
        }
        check(testTotal == rows, "every row appears once in test folds");
    }

    private static void testFacade() {
        int rows = 160;
        Random random = new Random(17);
        double[] values = new double[rows * 2];
        double[] targets = new double[rows];
        for (int row = 0; row < rows; row++) {
            double first = random.nextGaussian();
            double second = random.nextGaussian();
            values[row * 2] = first;
            values[row * 2 + 1] = second;
            targets[row] = 1.5 + 2.0 * first - 0.5 * second;
        }
        Matrix features = new Matrix(rows, 2, values);
        Estimator ridge = JSL.fit(JSL.Algorithm.RIDGE, features, targets);
        double[] prediction = JSL.predict(ridge, features);
        check(Metrics.r2(targets, prediction) > 0.99, "facade ridge R2");

        double score = JSL.crossValidate(JSL.Algorithm.RIDGE, features, targets, 5,
                JSL.Metric.R2);
        check(score > 0.99, "cross-validated R2 " + score);

        Estimator kmeans = JSL.fit(JSL.Algorithm.KMEANS, features, null);
        double[] clusters = JSL.predict(kmeans, features);
        check(clusters.length == rows, "facade KMeans prediction length");
    }

    private static void run(String name, Runnable test) {
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
}
