package jsl.tests;

import jsl.core.BC;
import jsl.core.BCHW;
import jsl.core.MatMul;
import jsl.core.Matrix;
import jsl.core.Tensor;
import jsl.ml.DecisionTree;
import jsl.ml.DistanceMetric;
import jsl.ml.ImpurityCriterion;
import jsl.ml.KNearestNeighbors;
import jsl.ml.LinearRegression;
import jsl.ml.LogisticRegression;
import jsl.ml.TreeTask;
import jsl.nn.Activation;
import jsl.nn.CNN;
import jsl.nn.Conv2D;
import jsl.nn.Dense;
import jsl.nn.Flatten;
import jsl.nn.Loss;
import jsl.nn.LossEvaluation;
import jsl.nn.MaxPooling2D;
import jsl.nn.NeuralNetwork;
import jsl.nn.Optimizer;
import jsl.nn.Padding;
import jsl.nn.TrainingHistory;
import java.util.Arrays;
import java.util.Random;

/** Zero-dependency unit and smoke tests; any assertion failure exits non-zero. */
public final class TestSuite {
    private static int passed;

    private TestSuite() {
    }

    public static void main(String[] args) {
        run("matrix scalar/SIMD/parallel correctness", TestSuite::testMatrix);
        run("tensor contiguous storage and indexing", TestSuite::testTensor);
        run("linear regression convergence", TestSuite::testLinearRegression);
        run("logistic regression convergence", TestSuite::testLogisticRegression);
        run("KNN SIMD distances and voting", TestSuite::testKnn);
        run("decision tree Gini and entropy", TestSuite::testDecisionTree);
        run("pooling and flatten backward", TestSuite::testSpatialLayers);
        run("dense numerical gradients and softmax", TestSuite::testDense);
        run("convolution numerical gradients", TestSuite::testConvolution);
        run("CNN builder, forward, backward and Adam", TestSuite::testCnn);
        run("invalid inputs fail fast", TestSuite::testValidation);
        System.out.println("PASS: " + passed + " test groups");
    }

    private static void testMatrix() {
        Matrix left = Matrix.fromRows(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Matrix right = Matrix.fromRows(new double[][]{{7, 8}, {9, 10}, {11, 12}});
        Matrix result = left.matMul(right);
        assertArray(new double[]{58, 64, 139, 154}, result.data(), 1e-12, "known matmul");

        int[][] shapes = {{1, 1, 1}, {3, 5, 2}, {17, 19, 13}, {64, 65, 63}};
        Random random = new Random(7);
        try (MatMul engine = MatMul.create()) {
            for (int[] shape : shapes) {
                int m = shape[0];
                int k = shape[1];
                int n = shape[2];
                double[] aValues = new double[m * k];
                double[] bValues = new double[k * n];
                for (int i = 0; i < aValues.length; i++) {
                    aValues[i] = random.nextDouble() * 2.0 - 1.0;
                }
                for (int i = 0; i < bValues.length; i++) {
                    bValues[i] = random.nextDouble() * 2.0 - 1.0;
                }
                Matrix a = new Matrix(m, k, aValues);
                Matrix b = new Matrix(k, n, bValues);
                Matrix expected = MatMul.scalar(a, b);
                Matrix actual = engine.multiply(a, b);
                assertArray(expected.data(), actual.data(), 1e-10 * k,
                        "random matmul " + m + "x" + k + " by " + k + "x" + n);
            }
        }
    }

    private static void testTensor() {
        double[] values = {1, 2, 3, 4, 5, 6, 7, 8};
        Tensor<BCHW> tensor = Tensor.ofBCHW(1, 2, 2, 2, values);
        values[0] = 99;
        assertClose(1.0, tensor.get(0, 0, 0, 0), 0.0, "constructor copies input");
        tensor.set(12.0, 0, 1, 1, 1);
        assertClose(12.0, tensor.data()[7], 0.0, "row-major index");
        assertClose(8, tensor.size(), 0.0, "tensor size");
        expectThrows(IllegalArgumentException.class,
                () -> Tensor.ofBC(2, 2, 1, 2, 3));
    }

    private static void testLinearRegression() {
        int samples = 120;
        double[] x = new double[samples * 2];
        double[] y = new double[samples];
        for (int row = 0; row < samples; row++) {
            double first = -1.0 + 2.0 * row / (samples - 1);
            double second = ((row * 37) % 101) / 100.0 - 0.5;
            x[row * 2] = first;
            x[row * 2 + 1] = second;
            y[row] = 2.5 + 3.0 * first - 1.5 * second;
        }
        LinearRegression model = LinearRegression.builder()
                .learningRate(0.05)
                .epochs(10_000)
                .build()
                .fit(new Matrix(samples, 2, x), y);
        double[] history = model.lossHistory();
        check(history[history.length - 1] < history[0] * 1e-6,
                "linear loss must decrease substantially: first=" + history[0]
                        + ", last=" + history[history.length - 1]);
        double[] predictions = model.predict(new Matrix(samples, 2, x));
        for (int i = 0; i < y.length; i++) {
            assertClose(y[i], predictions[i], 0.01, "linear prediction " + i);
        }
    }

    private static void testLogisticRegression() {
        Matrix x = Matrix.fromRows(new double[][]{{-3}, {-2}, {-1}, {1}, {2}, {3}});
        double[] y = {0, 0, 0, 1, 1, 1};
        LogisticRegression model = LogisticRegression.builder()
                .learningRate(0.2)
                .epochs(2_000)
                .build()
                .fit(x, y);
        assertArray(y, model.predict(x), 0.0, "logistic classes");
        double[] probabilities = model.predictProbability(x);
        check(probabilities[0] < 0.05 && probabilities[5] > 0.95,
                "logistic probabilities must separate training classes");
        double[] history = model.lossHistory();
        check(history[history.length - 1] < history[0], "logistic loss must decrease");
    }

    private static void testKnn() {
        Matrix train = Matrix.fromRows(new double[][]{{0}, {0.2}, {0.4}, {3}, {3.2}, {3.4}});
        double[] labels = {0, 0, 0, 1, 1, 1};
        KNearestNeighbors knn = KNearestNeighbors.builder()
                .k(3)
                .metric(DistanceMetric.EUCLIDEAN)
                .build()
                .fit(train, labels);
        Matrix queries = Matrix.fromRows(new double[][]{{0.1}, {3.1}});
        assertArray(new double[]{0, 1}, knn.predict(queries), 0.0, "KNN Euclidean");
        knn = KNearestNeighbors.builder().k(1).metric(DistanceMetric.MANHATTAN).build().fit(train, labels);
        assertArray(new double[]{0, 1}, knn.predict(queries), 0.0, "KNN Manhattan");
    }

    private static void testDecisionTree() {
        Matrix x = Matrix.fromRows(new double[][]{{-3}, {-2}, {-1}, {1}, {2}, {3}});
        double[] y = {0, 0, 0, 1, 1, 1};
        double tiny = 1e-300;
        double adjacent = Math.nextUp(tiny);
        DecisionTree adjacentTree = DecisionTree.builder().maxDepth(2).build()
                .fit(Matrix.fromRows(new double[][]{{tiny}, {adjacent}}), new double[]{0, 1});
        assertArray(new double[]{0, 1},
                adjacentTree.predict(Matrix.fromRows(new double[][]{{tiny}, {adjacent}})),
                0.0, "adjacent floating-point split");

        for (ImpurityCriterion criterion : ImpurityCriterion.values()) {
            if (criterion.task() != TreeTask.CLASSIFICATION) {
                continue;
            }
            DecisionTree tree = DecisionTree.builder()
                    .criterion(criterion)
                    .maxDepth(3)
                    .minSamplesLeaf(1)
                    .build()
                    .fit(x, y);
            assertArray(y, tree.predict(x), 0.0, "tree " + criterion);
            assertArray(new double[]{0, 1}, tree.predict(Matrix.fromRows(new double[][]{{-100}, {100}})),
                    0.0, "tree extrapolation " + criterion);
        }
    }

    private static void testSpatialLayers() {
        Tensor<BCHW> input = Tensor.ofBCHW(1, 1, 2, 2, 1, 3, 2, 4);
        MaxPooling2D pool = MaxPooling2D.builder().poolSize(2, 2).build();
        Tensor<BCHW> output = pool.forward(input);
        assertArray(new double[]{4}, output.data(), 0.0, "max pool output");
        Tensor<BCHW> inputGradient = pool.backward(Tensor.ofBCHW(1, 1, 1, 1, 2));
        assertArray(new double[]{0, 0, 0, 2}, inputGradient.data(), 0.0, "max pool gradient");
        expectThrows(IllegalArgumentException.class,
                () -> pool.backward(Tensor.ofBCHW(2, 1, 1, 1, 1, 1)));

        Tensor<BCHW> negativeInfinity = Tensor.ofBCHW(1, 1, 2, 2,
                Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY,
                Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);
        pool.forward(negativeInfinity);
        assertArray(new double[]{2, 0, 0, 0},
                pool.backward(Tensor.ofBCHW(1, 1, 1, 1, 2)).data(), 0.0,
                "negative-infinity pool gradient");

        Flatten flatten = Flatten.builder().build();
        Tensor<BC> flat = flatten.forward(input);
        check(flat.dimension(0) == 1 && flat.dimension(1) == 4, "flatten output shape");
        Tensor<BCHW> restored = flatten.backward(flat);
        assertArray(input.data(), restored.data(), 0.0, "flatten backward");
        expectThrows(IllegalArgumentException.class,
                () -> flatten.backward(Tensor.ofBC(2, 4, new double[8])));
    }

    private static void testDense() {
        Dense dense = Dense.builder().units(2).activation(Activation.NONE).seed(9).build();
        Tensor<BC> input = Tensor.ofBC(1, 2, 0.4, -0.7);
        Tensor<BC> output = dense.forward(input);
        Tensor<BC> outputGradient = Tensor.ofBC(1, 2, 0.2, -0.3);
        double[] analytic = dense.backward(outputGradient).data().clone();
        double epsilon = 1e-6;
        for (int feature = 0; feature < 2; feature++) {
            double[] plusValues = input.data().clone();
            double[] minusValues = input.data().clone();
            plusValues[feature] += epsilon;
            minusValues[feature] -= epsilon;
            double[] plus = dense.forward(Tensor.ofBC(1, 2, plusValues)).data();
            double[] minus = dense.forward(Tensor.ofBC(1, 2, minusValues)).data();
            double numerical = 0.0;
            for (int unit = 0; unit < 2; unit++) {
                numerical += (plus[unit] - minus[unit]) * outputGradient.data()[unit] / (2.0 * epsilon);
            }
            assertClose(numerical, analytic[feature], 1e-8, "dense input gradient " + feature);
        }

        Dense softmax = Dense.builder().units(3).activation(Activation.SOFTMAX).build();
        Tensor<BC> probabilities = softmax.forward(Tensor.ofBC(2, 2, 1, 2, 3, 4));
        for (int row = 0; row < 2; row++) {
            double sum = 0.0;
            for (int column = 0; column < 3; column++) {
                sum += probabilities.get(row, column);
            }
            assertClose(1.0, sum, 1e-12, "softmax normalization");
        }
        Tensor<BC> ones = Tensor.ofBC(2, 3, 1, 1, 1, 1, 1, 1);
        double[] gradient = softmax.backward(ones).data();
        for (double value : gradient) {
            assertClose(0.0, value, 1e-12, "softmax Jacobian times ones");
        }

        LossEvaluation crossEntropy = Loss.CATEGORICAL_CROSSENTROPY.evaluate(
                Tensor.ofBC(1, 2, 0.0, 1.0), Tensor.ofBC(1, 2, 1.0, 0.0));
        check(crossEntropy.hasSoftmaxLogitGradient(), "CCE supplies stable logit gradient");
        assertArray(new double[]{-1.0, 1.0}, crossEntropy.softmaxLogitGradient().data(),
                0.0, "saturated Softmax-CCE gradient");
    }

    private static void testConvolution() {
        Conv2D convolution = Conv2D.builder()
                .filters(1)
                .kernelSize(2, 2)
                .activation(Activation.NONE)
                .padding(Padding.VALID)
                .seed(5)
                .build();
        double[] values = {0.2, -0.4, 0.6, 0.8};
        Tensor<BCHW> input = Tensor.ofBCHW(1, 1, 2, 2, values);
        Tensor<BCHW> output = convolution.forward(input);
        check(output.dimension(2) == 1 && output.dimension(3) == 1, "valid convolution shape");
        double[] analytic = convolution.backward(Tensor.ofBCHW(1, 1, 1, 1, 0.7)).data().clone();
        double epsilon = 1e-6;
        for (int i = 0; i < values.length; i++) {
            double[] plus = values.clone();
            double[] minus = values.clone();
            plus[i] += epsilon;
            minus[i] -= epsilon;
            double plusOutput = convolution.forward(Tensor.ofBCHW(1, 1, 2, 2, plus)).data()[0];
            double minusOutput = convolution.forward(Tensor.ofBCHW(1, 1, 2, 2, minus)).data()[0];
            double numerical = (plusOutput - minusOutput) * 0.7 / (2.0 * epsilon);
            assertClose(numerical, analytic[i], 1e-8, "convolution input gradient " + i);
        }

        Conv2D same = Conv2D.builder().filters(2).kernelSize(3, 3).padding(Padding.SAME).build();
        Tensor<BCHW> sameOutput = same.forward(Tensor.ofBCHW(1, 1, 4, 4, new double[16]));
        check(sameOutput.dimension(1) == 2 && sameOutput.dimension(2) == 4
                && sameOutput.dimension(3) == 4, "same convolution shape");
    }

    private static void testCnn() {
        CNN model = NeuralNetwork.builder()
                .add(Conv2D.builder().filters(2).kernelSize(3, 3)
                        .activation(Activation.RELU).padding(Padding.SAME).build())
                .add(MaxPooling2D.builder().poolSize(2, 2).build())
                .add(Flatten.builder().build())
                .add(Dense.builder().units(2).activation(Activation.SOFTMAX).build())
                .loss(Loss.CATEGORICAL_CROSSENTROPY)
                .optimizer(Optimizer.ADAM.withLearningRate(0.001))
                .build();
        double[] images = new double[2 * 1 * 4 * 4];
        for (int i = 0; i < images.length; i++) {
            images[i] = (i % 7) / 7.0;
        }
        Tensor<BCHW> input = Tensor.ofBCHW(2, 1, 4, 4, images);
        Tensor<BC> target = Tensor.ofBC(2, 2, 1, 0, 0, 1);
        Tensor<BC> output = model.forward(input);
        check(output.dimension(0) == 2 && output.dimension(1) == 2, "CNN output shape");
        for (int row = 0; row < 2; row++) {
            assertClose(1.0, output.get(row, 0) + output.get(row, 1), 1e-12,
                    "CNN softmax row " + row);
        }
        LossEvaluation evaluation = model.evaluate(input, target);
        Tensor<BCHW> gradient = model.backward(evaluation.gradient());
        check(Arrays.equals(input.shape(), gradient.shape()), "CNN input gradient shape");
        for (double value : gradient.data()) {
            check(Double.isFinite(value), "CNN gradients must be finite");
        }
        TrainingHistory history = model.fit(input, target, 3);
        check(history.epochs() == 3 && Double.isFinite(history.lastLoss()), "CNN Adam training");
        check(model.layerCount() == 4, "CNN layer count");
    }

    private static void testValidation() {
        expectThrows(IllegalArgumentException.class, () -> MatMul.scalar(
                Matrix.zeros(2, 3), Matrix.zeros(2, 3)));
        expectThrows(IllegalStateException.class,
                () -> LinearRegression.builder().build().predict(Matrix.zeros(1, 1)));
        expectThrows(IllegalArgumentException.class,
                () -> KNearestNeighbors.builder().k(0).build());
        expectThrows(IllegalArgumentException.class,
                () -> Conv2D.builder().activation(Activation.SOFTMAX).build());
        expectThrows(IllegalArgumentException.class,
                () -> Optimizer.ADAM.withLearningRate(-1));
        expectThrows(IllegalArgumentException.class,
                () -> MatMul.scalar(Matrix.zeros(50_000, 1), Matrix.zeros(1, 50_000)));
        expectThrows(IllegalArgumentException.class,
                () -> Loss.CATEGORICAL_CROSSENTROPY.evaluate(
                        Tensor.ofBC(1, 2, 1.2, -0.2), Tensor.ofBC(1, 2, 1, 0)));

        Dense dense = Dense.builder().build();
        dense.forward(Tensor.ofBC(1, 1, 1));
        expectThrows(IllegalStateException.class,
                () -> dense.update(Optimizer.ADAM.withLearningRate(0.001)));
        Conv2D convolution = Conv2D.builder().build();
        convolution.forward(Tensor.ofBCHW(1, 1, 3, 3, new double[9]));
        expectThrows(IllegalStateException.class,
                () -> convolution.update(Optimizer.ADAM.withLearningRate(0.001)));
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
        check(expected.length == actual.length, message + ": length mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertClose(expected[i], actual[i], tolerance, message + " at " + i);
        }
    }

    private static void assertClose(double expected, double actual, double tolerance, String message) {
        check(Double.isFinite(actual) && Math.abs(expected - actual) <= tolerance,
                message + ": expected " + expected + " but got " + actual
                        + " (tolerance " + tolerance + ")");
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
}
