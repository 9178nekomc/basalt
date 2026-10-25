package basalt.tests;

import basalt.nativebridge.CommandQueueBuilder;
import basalt.nativebridge.NativeContext;
import basalt.nativebridge.NativeLibrary;
import basalt.nativebridge.NativeMLP;
import basalt.nativebridge.NativeTensorHandle;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Numerical and lifecycle tests for the Rust tensor engine.
 *
 * <p>The suite is skipped with a clear message when the shared library is absent, so a pure-Java
 * checkout still builds. {@code verify.sh} runs it whenever {@code build/native} contains the
 * library, and {@code native-build.sh} builds the library before verification.</p>
 */
public final class NativeTestSuite {
    private static int passed;

    private NativeTestSuite() {
    }

    public static void main(String[] args) {
        if (!NativeLibrary.available()) {
            System.out.println("SKIP: native tensor runtime unavailable: " + NativeLibrary.loadError());
            return;
        }
        run("library ABI", NativeTestSuite::testLibrary);
        run("matmul correctness and tails", NativeTestSuite::testMatMul);
        run("broadcast add, relu and queue batching", NativeTestSuite::testQueue);
        run("matmul/add/mse analytic gradients", NativeTestSuite::testAnalyticGradients);
        run("mlp finite-difference gradient check", NativeTestSuite::testFiniteDifference);
        run("sgd and adam update direction", NativeTestSuite::testOptimizers);
        run("tensor release is idempotent", NativeTestSuite::testDoubleRelease);
        run("cleaner releases abandoned handles", NativeTestSuite::testCleaner);
        run("context close invalidates handles safely", NativeTestSuite::testContextClose);
        run("invalid shapes fail fast", NativeTestSuite::testValidation);
        run("queue growth beyond initial capacity", NativeTestSuite::testQueueGrowth);
        run("xor convergence", NativeTestSuite::testXor);
        run("linear regression convergence", NativeTestSuite::testLinearRegression);
        run("concurrent contexts", NativeTestSuite::testConcurrentContexts);
        System.out.println("PASS: " + passed + " native test groups");
    }

    private static void testLibrary() {
        check(NativeLibrary.version() == 2, "expected tensor ABI version 2");
        check(NativeLibrary.libraryPath() != null && !NativeLibrary.libraryPath().isBlank(),
                "library path must be reported");
    }

    private static void testMatMul() {
        int[][] shapes = {{1, 1, 1}, {2, 3, 4}, {7, 9, 11}, {16, 17, 13}, {33, 31, 9}};
        Random random = new Random(11);
        try (NativeContext context = NativeContext.create(4)) {
            for (int[] shape : shapes) {
                int m = shape[0];
                int k = shape[1];
                int n = shape[2];
                float[] left = randomValues(random, m * k);
                float[] right = randomValues(random, k * n);
                NativeTensorHandle a = context.of(left, false, m, k);
                NativeTensorHandle b = context.of(right, false, k, n);
                NativeTensorHandle c = context.zeros(m, n);
                try {
                    new CommandQueueBuilder(context).matMul(a, b, c).execute();
                    float[] expected = referenceMatMul(left, right, m, k, n);
                    assertArray(expected, c.data(), 2e-4f,
                            "matmul " + m + "x" + k + "x" + n);
                } finally {
                    a.close();
                    b.close();
                    c.close();
                }
            }
        }
    }

    private static void testQueue() {
        try (NativeContext context = NativeContext.create(2)) {
            NativeTensorHandle a = context.of(new float[]{1, 2, 3, 4, 5, 6}, false, 2, 3);
            NativeTensorHandle bias = context.of(new float[]{10, 20, 30}, false, 3);
            NativeTensorHandle sum = context.zeros(2, 3);
            NativeTensorHandle activated = context.zeros(2, 3);
            NativeTensorHandle scaled = context.zeros(2, 3);
            try {
                new CommandQueueBuilder(context)
                        .add(a, bias, sum)
                        .relu(sum, activated)
                        .mul(activated, bias, scaled)
                        .execute();
                assertArray(new float[]{110, 440, 990, 140, 500, 1080}, scaled.data(), 1e-4f,
                        "queued add/relu/mul");
                check(context.graphSize() == 3, "three graph nodes expected");
            } finally {
                a.close();
                bias.close();
                sum.close();
                activated.close();
                scaled.close();
            }
        }
    }

    private static void testAnalyticGradients() {
        // pred = X W + b; loss = mean((pred - Y)^2)
        float[] xValues = {1, 2, 3, 4, 5, 6, 7, 8};
        float[] yValues = {1, -1, 2, 0, 3, 1, 4, 2};
        try (NativeContext context = NativeContext.create(4)) {
            NativeTensorHandle x = context.of(xValues, false, 4, 2);
            NativeTensorHandle y = context.of(yValues, false, 4, 2);
            NativeTensorHandle w = context.of(new float[]{0.5f, -0.25f, 0.75f, 0.1f}, true, 2, 2);
            NativeTensorHandle b = context.of(new float[]{0.2f, -0.3f}, true, 2);
            NativeTensorHandle product = context.zeros(4, 2);
            NativeTensorHandle prediction = context.zeros(4, 2);
            NativeTensorHandle loss = context.zeros(1);
            try {
                new CommandQueueBuilder(context)
                        .matMul(x, w, product)
                        .add(product, b, prediction)
                        .mse(prediction, y, loss)
                        .execute();
                context.backward(loss);
                float[] predictionValues = prediction.data();
                float[] expectedW = new float[4];
                float[] expectedB = new float[2];
                float scale = 2.0f / predictionValues.length;
                for (int row = 0; row < 4; row++) {
                    for (int column = 0; column < 2; column++) {
                        int index = row * 2 + column;
                        float gradient = (predictionValues[index] - yValues[index]) * scale;
                        expectedB[column] += gradient;
                        for (int inner = 0; inner < 2; inner++) {
                            expectedW[inner * 2 + column] += xValues[row * 2 + inner] * gradient;
                        }
                    }
                }
                assertArray(expectedW, w.grad(), 2e-5f, "dL/dW");
                assertArray(expectedB, b.grad(), 2e-5f, "dL/db");
            } finally {
                closeAll(x, y, w, b, product, prediction, loss);
            }
        }
    }

    private static void testFiniteDifference() {
        float[] xValues = {0.2f, -0.4f, 0.7f, 0.1f, -0.6f, 0.3f, 0.9f, -0.2f, 0.5f, 0.8f};
        float[] yValues = {0.3f, -0.7f, 0.2f, 0.6f, -0.1f};
        try (NativeContext context = NativeContext.create(2)) {
            NativeTensorHandle x = context.of(xValues, false, 5, 2);
            NativeTensorHandle y = context.of(yValues, false, 5, 1);
            NativeTensorHandle w1 = context.of(
                    new float[]{0.3f, -0.2f, 0.1f, 0.4f, -0.5f, 0.2f}, true, 2, 3);
            NativeTensorHandle b1 = context.of(new float[]{0.1f, -0.1f, 0.05f}, true, 3);
            NativeTensorHandle w2 = context.of(new float[]{0.7f, -0.3f, 0.2f}, true, 3, 1);
            NativeTensorHandle b2 = context.of(new float[]{-0.2f}, true, 1);
            NativeTensorHandle product1 = context.zeros(5, 3);
            NativeTensorHandle biased1 = context.zeros(5, 3);
            NativeTensorHandle hidden = context.zeros(5, 3);
            NativeTensorHandle product2 = context.zeros(5, 1);
            NativeTensorHandle prediction = context.zeros(5, 1);
            NativeTensorHandle loss = context.zeros(1);
            List<NativeTensorHandle> tensors = List.of(x, y, w1, b1, w2, b2,
                    product1, biased1, hidden, product2, prediction, loss);
            try {
                forwardMlp(context, x, w1, b1, w2, b2, product1, biased1, hidden,
                        product2, prediction, y, loss);
                context.backward(loss);
                List<NativeTensorHandle> parameters = List.of(w1, b1, w2, b2);
                List<float[]> analytic = new ArrayList<>();
                for (NativeTensorHandle parameter : parameters) {
                    analytic.add(parameter.grad());
                }
                for (int parameterIndex = 0; parameterIndex < parameters.size(); parameterIndex++) {
                    NativeTensorHandle parameter = parameters.get(parameterIndex);
                    float[] values = parameter.data();
                    float[] numeric = new float[values.length];
                    for (int index = 0; index < values.length; index++) {
                        float original = values[index];
                        values[index] = original + 1e-3f;
                        parameter.setData(values);
                        float plus = runLoss(context, x, w1, b1, w2, b2, product1, biased1,
                                hidden, product2, prediction, y, loss);
                        values[index] = original - 1e-3f;
                        parameter.setData(values);
                        float minus = runLoss(context, x, w1, b1, w2, b2, product1, biased1,
                                hidden, product2, prediction, y, loss);
                        values[index] = original;
                        parameter.setData(values);
                        numeric[index] = (plus - minus) / 2e-3f;
                    }
                    assertArray(numeric, analytic.get(parameterIndex), 2e-3f,
                            "finite-difference parameter " + parameterIndex);
                }
            } finally {
                for (NativeTensorHandle tensor : tensors) {
                    tensor.close();
                }
            }
        }
    }

    private static void forwardMlp(NativeContext context, NativeTensorHandle x,
                                   NativeTensorHandle w1, NativeTensorHandle b1,
                                   NativeTensorHandle w2, NativeTensorHandle b2,
                                   NativeTensorHandle product1, NativeTensorHandle biased1,
                                   NativeTensorHandle hidden, NativeTensorHandle product2,
                                   NativeTensorHandle prediction, NativeTensorHandle y,
                                   NativeTensorHandle loss) {
        context.clearGraph();
        new CommandQueueBuilder(context)
                .matMul(x, w1, product1)
                .add(product1, b1, biased1)
                .tanh(biased1, hidden)
                .matMul(hidden, w2, product2)
                .add(product2, b2, prediction)
                .mse(prediction, y, loss)
                .execute();
    }

    private static float runLoss(NativeContext context, NativeTensorHandle x,
                                 NativeTensorHandle w1, NativeTensorHandle b1,
                                 NativeTensorHandle w2, NativeTensorHandle b2,
                                 NativeTensorHandle product1, NativeTensorHandle biased1,
                                 NativeTensorHandle hidden, NativeTensorHandle product2,
                                 NativeTensorHandle prediction, NativeTensorHandle y,
                                 NativeTensorHandle loss) {
        forwardMlp(context, x, w1, b1, w2, b2, product1, biased1, hidden,
                product2, prediction, y, loss);
        return loss.data()[0];
    }

    private static void testOptimizers() {
        try (NativeContext context = NativeContext.create(1)) {
            NativeTensorHandle target = context.of(new float[]{5.0f}, false, 1);
            NativeTensorHandle parameter = context.of(new float[]{0.0f}, true, 1);
            NativeTensorHandle loss = context.zeros(1);
            try {
                new CommandQueueBuilder(context).mse(parameter, target, loss).execute();
                context.backward(loss);
                assertArray(new float[]{-10.0f}, parameter.grad(), 1e-6f, "quadratic gradient");
                parameter.sgd(0.1f);
                assertArray(new float[]{1.0f}, parameter.data(), 1e-6f, "sgd step");
            } finally {
                target.close();
                parameter.close();
                loss.close();
            }

            NativeTensorHandle target2 = context.of(new float[]{5.0f}, false, 1);
            NativeTensorHandle parameter2 = context.of(new float[]{0.0f}, true, 1);
            NativeTensorHandle loss2 = context.zeros(1);
            try {
                new CommandQueueBuilder(context).mse(parameter2, target2, loss2).execute();
                context.backward(loss2);
                parameter2.adam(0.1f, 0.9f, 0.999f, 1e-8f, 0.0f, 1);
                float updated = parameter2.data()[0];
                check(updated > 0.05f && updated < 0.15f, "adam first step should be ~0.1");
            } finally {
                target2.close();
                parameter2.close();
                loss2.close();
            }
        }
    }

    private static void testDoubleRelease() {
        try (NativeContext context = NativeContext.create(1)) {
            List<NativeTensorHandle> tensors = new ArrayList<>();
            for (int index = 0; index < 128; index++) {
                tensors.add(context.zeros(4, 4));
            }
            for (NativeTensorHandle tensor : tensors) {
                tensor.close();
                tensor.close();
            }
            check(context.stats()[0] == 0, "all tensors must be released");
            check(context.stats()[1] == 0, "live bytes must return to zero");
        }
    }

    private static void testCleaner() {
        try (NativeContext context = NativeContext.create(1)) {
            allocateAbandoned(context, 200);
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
            while (context.stats()[0] > 0 && System.nanoTime() < deadline) {
                System.gc();
                try {
                    Thread.sleep(50);
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
            check(context.stats()[0] == 0,
                    "Cleaner did not release all handles: " + context.stats()[0] + " remain");
        }
    }

    private static void allocateAbandoned(NativeContext context, int count) {
        for (int index = 0; index < count; index++) {
            context.zeros(8, 8);
        }
    }

    private static void testContextClose() {
        NativeContext context = NativeContext.create(1);
        NativeTensorHandle tensor = context.zeros(2, 2);
        context.close();
        check(!context.isOpen(), "context must report closed");
        expectThrows(IllegalStateException.class, tensor::data);
        tensor.close();
        expectThrows(IllegalStateException.class, context::stats);
    }

    private static void testValidation() {
        try (NativeContext context = NativeContext.create(1)) {
            NativeTensorHandle a = context.zeros(2, 3);
            NativeTensorHandle b = context.zeros(4, 2);
            NativeTensorHandle output = context.zeros(2, 2);
            try {
                expectThrows(IllegalStateException.class,
                        () -> new CommandQueueBuilder(context).matMul(a, b, output).execute());
                expectThrows(IllegalStateException.class,
                        () -> new CommandQueueBuilder(context).add(a, b, output).execute());
                expectThrows(IllegalStateException.class,
                        () -> new CommandQueueBuilder(context).relu(a, a).execute());
                expectThrows(IllegalArgumentException.class,
                        () -> context.zeros(2, 0));
                expectThrows(IllegalArgumentException.class,
                        () -> context.of(new float[]{1, 2}, 3, 3));
            } finally {
                a.close();
                b.close();
                output.close();
            }
        }
    }

    private static void testQueueGrowth() {
        try (NativeContext context = NativeContext.create(1)) {
            List<NativeTensorHandle> tensors = new ArrayList<>();
            NativeTensorHandle input = context.of(new float[]{-2, -1, 0, 1, 2}, false, 5);
            tensors.add(input);
            NativeTensorHandle current = input;
            CommandQueueBuilder queue = new CommandQueueBuilder(context);
            int commands = 40;
            for (int index = 0; index < commands; index++) {
                NativeTensorHandle next = context.zeros(5);
                tensors.add(next);
                queue.relu(current, next);
                current = next;
            }
            check(queue.size() == commands, "all commands must be queued");
            check(queue.capacity() >= commands, "buffer must grow to hold every command");
            queue.execute();
            assertArray(new float[]{0, 0, 0, 1, 2}, current.data(), 1e-6f, "relu chain");
            for (NativeTensorHandle tensor : tensors) {
                tensor.close();
            }
        }
    }

    private static void testXor() {
        float[][] x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        float[][] y = {{0}, {1}, {1}, {0}};
        try (NativeContext context = NativeContext.create(4);
             NativeMLP mlp = new NativeMLP(context, new int[]{2, 12, 1},
                     NativeMLP.Activation.TANH, NativeMLP.OutputActivation.SIGMOID, 42)) {
            float loss = mlp.fit(x, y, 3000, 4, 0.05f, 0.0f);
            check(loss < 0.01f, "XOR loss must converge, got " + loss);
            float[] predictions = mlp.predictScalar(x);
            for (int index = 0; index < predictions.length; index++) {
                boolean high = predictions[index] > 0.5f;
                boolean expected = y[index][0] > 0.5f;
                check(high == expected,
                        String.format(Locale.ROOT, "XOR row %d: %.4f", index, predictions[index]));
            }
        }
    }

    private static void testLinearRegression() {
        int rows = 128;
        float[][] x = new float[rows][1];
        float[][] y = new float[rows][1];
        for (int row = 0; row < rows; row++) {
            float value = -2.0f + 4.0f * row / (rows - 1);
            x[row][0] = value;
            y[row][0] = 3.0f * value + 2.0f;
        }
        try (NativeContext context = NativeContext.create(2);
             NativeMLP mlp = new NativeMLP(context, new int[]{1, 1},
                     NativeMLP.Activation.RELU, NativeMLP.OutputActivation.NONE, 7)) {
            float loss = mlp.fit(x, y, 400, 64, 0.05f, 0.0f);
            check(loss < 0.05f, "linear regression MSE must converge, got " + loss);
            float[] prediction = mlp.predictScalar(new float[][]{{4.0f}});
            check(Math.abs(prediction[0] - 14.0f) < 0.5f,
                    "prediction at x=4 should be ~14, got " + prediction[0]);
        }
    }

    private static void testConcurrentContexts() {
        int workers = 4;
        ExecutorService executor = Executors.newFixedThreadPool(workers);
        CountDownLatch start = new CountDownLatch(1);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        List<java.util.concurrent.Future<?>> futures = new ArrayList<>();
        for (int worker = 0; worker < workers; worker++) {
            final int id = worker;
            futures.add(executor.submit(() -> {
                try {
                    start.await();
                    try (NativeContext context = NativeContext.create(2)) {
                        float[] left = new float[64 * 64];
                        float[] right = new float[64 * 64];
                        Arrays.fill(left, id + 1.0f);
                        Arrays.fill(right, 1.0f);
                        NativeTensorHandle a = context.of(left, false, 64, 64);
                        NativeTensorHandle b = context.of(right, false, 64, 64);
                        NativeTensorHandle c = context.zeros(64, 64);
                        new CommandQueueBuilder(context).matMul(a, b, c).execute();
                        float[] values = c.data();
                        check(Math.abs(values[0] - 64.0f * (id + 1)) < 1e-3f,
                                "concurrent matmul value");
                        a.close();
                        b.close();
                        c.close();
                        check(context.stats()[0] == 0, "concurrent release");
                    }
                } catch (Throwable throwable) {
                    failure.compareAndSet(null, throwable);
                }
            }));
        }
        start.countDown();
        executor.shutdown();
        try {
            check(executor.awaitTermination(60, TimeUnit.SECONDS), "concurrent workers timed out");
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new AssertionError(interrupted);
        }
        if (failure.get() != null) {
            throw new AssertionError("concurrent context failure", failure.get());
        }
        check(futures.size() == workers, "worker count");
    }

    private static float[] referenceMatMul(float[] a, float[] b, int m, int k, int n) {
        float[] c = new float[m * n];
        for (int row = 0; row < m; row++) {
            for (int column = 0; column < n; column++) {
                float sum = 0.0f;
                for (int inner = 0; inner < k; inner++) {
                    sum += a[row * k + inner] * b[inner * n + column];
                }
                c[row * n + column] = sum;
            }
        }
        return c;
    }

    private static float[] randomValues(Random random, int count) {
        float[] values = new float[count];
        for (int index = 0; index < count; index++) {
            values[index] = random.nextFloat() * 2.0f - 1.0f;
        }
        return values;
    }

    private static void closeAll(NativeTensorHandle... tensors) {
        for (NativeTensorHandle tensor : tensors) {
            tensor.close();
        }
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

    private static void assertArray(float[] expected, float[] actual, float tolerance,
                                    String message) {
        check(expected.length == actual.length, message + ": length mismatch");
        for (int index = 0; index < expected.length; index++) {
            float difference = Math.abs(expected[index] - actual[index]);
            check(Float.isFinite(actual[index]) && difference <= tolerance,
                    message + " at " + index + ": expected " + expected[index]
                            + " but got " + actual[index] + " (tolerance " + tolerance + ")");
        }
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
