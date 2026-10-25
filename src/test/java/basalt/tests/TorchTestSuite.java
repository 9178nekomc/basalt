package basalt.tests;

import basalt.core.BC;
import basalt.core.BCHW;
import basalt.core.BSF;
import basalt.nativebridge.NativeContext;
import basalt.torch.Activation;
import basalt.torch.AdamW;
import basalt.torch.BatchNorm2d;
import basalt.torch.Conv2d;
import basalt.torch.CrossEntropyLoss;
import basalt.torch.Flatten;
import basalt.torch.LayerNorm;
import basalt.torch.Linear;
import basalt.torch.MaxPool2d;
import basalt.torch.Module;
import basalt.torch.MultiheadAttention;
import basalt.torch.Sequential;
import basalt.torch.Tensor;
import basalt.torch.TransformerBlock;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;

/**
 * Numerical gradient checks and end-to-end training tests for the {@code basalt.torch} eager API.
 * Every new native operator is checked against central finite differences.
 */
public final class TorchTestSuite {
    private static int passed;

    private TorchTestSuite() {
    }

    public static void main(String[] args) {
        run("softmax + cross-entropy gradients", TorchTestSuite::testSoftmaxCrossEntropy);
        run("layer-norm gradients", TorchTestSuite::testLayerNorm);
        run("batch-norm gradients", TorchTestSuite::testBatchNorm);
        run("conv2d input/weight gradients", TorchTestSuite::testConv2d);
        run("max-pool gradient", TorchTestSuite::testMaxPool);
        run("embedding scatter-add gradient", TorchTestSuite::testEmbedding);
        run("batched-matmul gradients", TorchTestSuite::testBatchedMatMul);
        run("dropout mask consistency", TorchTestSuite::testDropout);
        run("permute and reshape gradients", TorchTestSuite::testPermuteReshape);
        run("cnn end-to-end training", TorchTestSuite::testCnnTraining);
        run("transformer block training", TorchTestSuite::testTransformerTraining);
        run("multi-head attention training", TorchTestSuite::testMultiheadAttention);
        run("typed sequential pipeline", TorchTestSuite::testTypedPipeline);
        run("state dict round-trip", TorchTestSuite::testStateDict);
        System.out.println("PASS: " + passed + " torch test groups");
    }

    private static void testSoftmaxCrossEntropy() {
        try (NativeContext context = NativeContext.create(2)) {
            Tensor logits = Tensor.of(context, new float[]{2.0f, 1.0f, 0.0f, 0.0f, 3.0f, 1.0f},
                    true, 2, 3);
            Tensor targets = Tensor.of(context, new float[]{0.0f, 1.0f}, 2);
            checkGradient(context, logits, () -> logits.crossEntropy(targets), 1e-2f, 2e-2f,
                    "cross-entropy logits");
            logits.close();
            targets.close();
        }
    }

    private static void testLayerNorm() {
        try (NativeContext context = NativeContext.create(2)) {
            Tensor input = Tensor.of(context, new float[]{1, 2, 3, 4, 2, 4, 6, 8}, true, 2, 4);
            Tensor gamma = Tensor.of(context, new float[]{1, 1, 1, 1}, true, 4);
            Tensor beta = Tensor.of(context, new float[]{0, 0, 0, 0}, true, 4);
            Tensor target = Tensor.of(context, new float[]{0.5f, -0.5f, 1.0f, -1.0f,
                    0.2f, -0.2f, 0.4f, -0.4f}, 2, 4);
            checkGradient(context, input, () -> input.layerNorm(gamma, beta, 1e-5f).mse(target),
                    1e-2f, 3e-2f, "layer-norm input");
            checkGradient(context, gamma, () -> input.layerNorm(gamma, beta, 1e-5f).mse(target),
                    1e-2f, 3e-2f, "layer-norm gamma");
            checkGradient(context, beta, () -> input.layerNorm(gamma, beta, 1e-5f).mse(target),
                    1e-2f, 3e-2f, "layer-norm beta");
            input.close();
            gamma.close();
            beta.close();
            target.close();
        }
    }

    private static void testBatchNorm() {
        try (NativeContext context = NativeContext.create(2)) {
            Tensor input = Tensor.of(context,
                    new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                    true, 2, 2, 2, 2);
            Tensor gamma = Tensor.of(context, new float[]{1, 1}, true, 2);
            Tensor beta = Tensor.of(context, new float[]{0, 0}, true, 2);
            Tensor runningMean = Tensor.zeros(context, 2);
            Tensor runningVar = Tensor.zeros(context, 2);
            runningVar.fill(1.0f);
            Tensor target = Tensor.zeros(context, 2, 2, 2, 2);
            Supplier<Tensor> factory = () -> {
                runningMean.fill(0.0f);
                runningVar.fill(1.0f);
                return input.batchNorm2d(gamma, beta, runningMean, runningVar, 1e-5f, 0.1f,
                        true).mse(target);
            };
            checkGradient(context, input, factory, 1e-2f, 3e-2f, "batch-norm input");
            checkGradient(context, gamma, factory, 1e-2f, 3e-2f, "batch-norm gamma");
            checkGradient(context, beta, factory, 1e-2f, 3e-2f, "batch-norm beta");
            input.close();
            gamma.close();
            beta.close();
            runningMean.close();
            runningVar.close();
            target.close();
        }
    }

    private static void testConv2d() {
        try (NativeContext context = NativeContext.create(2)) {
            Tensor input = Tensor.of(context,
                    new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, true, 1, 1, 3, 3);
            Tensor weight = Tensor.of(context,
                    new float[]{1, 0, 0, 1, 0.5f, -0.5f, 0.25f, 0.75f}, true, 2, 1, 2, 2);
            Tensor target = Tensor.zeros(context, 1, 2, 2, 2);
            Supplier<Tensor> factory = () -> input.conv2d(weight, 2, 2, 1, 1, 0, 0).mse(target);
            checkGradient(context, input, factory, 1e-2f, 3e-2f, "conv2d input");
            checkGradient(context, weight, factory, 1e-2f, 3e-2f, "conv2d weight");
            input.close();
            weight.close();
            target.close();
        }
    }

    private static void testMaxPool() {
        try (NativeContext context = NativeContext.create(2)) {
            Tensor input = Tensor.of(context,
                    new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                    true, 1, 1, 4, 4);
            Tensor target = Tensor.zeros(context, 1, 1, 2, 2);
            checkGradient(context, input,
                    () -> input.maxPool2d(2, 2, 2, 2).mse(target), 1e-2f, 3e-2f, "max-pool input");
            input.close();
            target.close();
        }
    }

    private static void testEmbedding() {
        try (NativeContext context = NativeContext.create(2)) {
            Tensor indices = Tensor.of(context, new float[]{0, 2, 1, 0}, 4);
            Tensor weight = Tensor.of(context,
                    new float[]{10, 11, 20, 21, 30, 31}, true, 3, 2);
            Tensor target = Tensor.zeros(context, 4, 2);
            checkGradient(context, weight, () -> indices.embedding(weight).mse(target),
                    1e-2f, 2e-2f, "embedding weight");
            indices.close();
            weight.close();
            target.close();
        }
    }

    private static void testBatchedMatMul() {
        try (NativeContext context = NativeContext.create(2)) {
            Tensor left = Tensor.of(context, new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                    true, 2, 2, 3);
            Tensor right = Tensor.of(context,
                    new float[]{1, 0, 2, 0, 1, 0, 1, 1, 0, 0, 1, 2}, true, 2, 3, 2);
            Tensor target = Tensor.zeros(context, 2, 2, 2);
            Supplier<Tensor> factory = () -> left.batchedMatMul(right).mse(target);
            checkGradient(context, left, factory, 1e-2f, 3e-2f, "batched-matmul left");
            checkGradient(context, right, factory, 1e-2f, 3e-2f, "batched-matmul right");
            left.close();
            right.close();
            target.close();
        }
    }

    private static void testDropout() {
        try (NativeContext context = NativeContext.create(1)) {
            Tensor input = Tensor.of(context, new float[]{1, 2, 3, 4, 5, 6, 7, 8}, true, 8);
            Tensor target = Tensor.zeros(context, 8);
            long seed = 12345L;
            checkGradient(context, input,
                    () -> input.dropout(0.5f, true, seed).mse(target), 1e-2f, 2e-2f,
                    "dropout input");
            Tensor evaluated = input.dropout(0.5f, false, seed);
            assertArray(input.data(), evaluated.data(), 0.0f, "dropout eval identity");
            input.close();
            target.close();
            evaluated.close();
        }
    }

    private static void testPermuteReshape() {
        try (NativeContext context = NativeContext.create(2)) {
            Tensor input = Tensor.of(context,
                    new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, true, 2, 2, 3);
            Tensor target = Tensor.zeros(context, 2, 3, 2);
            checkGradient(context, input,
                    () -> input.permute(0, 2, 1).mse(target), 1e-2f, 2e-2f, "permute input");
            checkGradient(context, input,
                    () -> input.reshape(3, 4).mse(Tensor.zeros(context, 3, 4)), 1e-2f, 2e-2f,
                    "reshape input");
            input.close();
            target.close();
        }
    }

    private static void testCnnTraining() {
        int samples = 64;
        float[] images = new float[samples * 64];
        float[] labels = new float[samples];
        Random random = new Random(7);
        for (int sample = 0; sample < samples; sample++) {
            int label = sample % 2;
            labels[sample] = label;
            for (int row = 0; row < 8; row++) {
                for (int column = 0; column < 8; column++) {
                    float value = random.nextFloat() * 0.2f;
                    if (label == 0 ? (column >= 3 && column <= 4) : (row >= 3 && row <= 4)) {
                        value += 1.0f;
                    }
                    images[sample * 64 + row * 8 + column] = value;
                }
            }
        }
        try (NativeContext context = NativeContext.create(4)) {
            Tensor input = Tensor.of(context, images, samples, 1, 8, 8);
            Tensor target = Tensor.of(context, labels, samples);
            Conv2d conv = new Conv2d(context, 1, 4, 3, 3, 1, 1, 1, 1, true, 11);
            Activation<BCHW> relu = new Activation<>(Activation.Kind.RELU);
            MaxPool2d pool = new MaxPool2d(2);
            Flatten flatten = new Flatten();
            Linear head = new Linear(context, 4 * 4 * 4, 2, true, 13);
            Sequential<BCHW, BC> network = Sequential.<BCHW>start()
                    .add(conv).add(relu).add(pool).add(flatten).add(head);
            CrossEntropyLoss lossFunction = new CrossEntropyLoss();
            AdamW optimizer = new AdamW(network.parameters(), 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f);
            float loss = 0.0f;
            for (int step = 0; step < 120; step++) {
                Tensor logits = network.forward(input);
                Tensor value = lossFunction.loss(logits, target);
                loss = value.data()[0];
                value.backward();
                optimizer.step();
                optimizer.zeroGrad();
                context.clearGraph();
            }
            Tensor logits = network.forward(input);
            float[] probabilities = logits.softmax(1).data();
            int correct = 0;
            for (int sample = 0; sample < samples; sample++) {
                int predicted = probabilities[sample * 2] > probabilities[sample * 2 + 1] ? 0 : 1;
                if (predicted == (int) labels[sample]) {
                    correct++;
                }
            }
            check(loss < 0.02f, "CNN loss must converge, got " + loss);
            check(correct >= samples * 0.95, "CNN accuracy must exceed 95%, got " + correct);
            context.clearGraph();
            network.close();
            input.close();
            target.close();
        }
    }

    private static void testTransformerTraining() {
        int batch = 2;
        int time = 4;
        int dimension = 8;
        float[] values = new float[batch * time * dimension];
        Random random = new Random(3);
        for (int index = 0; index < values.length; index++) {
            values[index] = random.nextFloat() * 2.0f - 1.0f;
        }
        try (NativeContext context = NativeContext.create(4)) {
            Tensor input = Tensor.of(context, values, batch, time, dimension);
            Tensor target = Tensor.zeros(context, batch, time, dimension);
            TransformerBlock block = new TransformerBlock(context, dimension, 16, true, 0.0f, 5);
            AdamW optimizer = new AdamW(block.parameters(), 0.01f);
            float first = 0.0f;
            float last = 0.0f;
            for (int step = 0; step < 40; step++) {
                Tensor output = block.forward(input);
                Tensor loss = output.mse(target);
                if (step == 0) {
                    first = loss.data()[0];
                }
                last = loss.data()[0];
                loss.backward();
                optimizer.step();
                optimizer.zeroGrad();
                context.clearGraph();
            }
            check(last < first * 0.5f,
                    "transformer loss must at least halve: " + first + " -> " + last);
            block.close();
            input.close();
            target.close();
        }
    }

    private static void testMultiheadAttention() {
        int batch = 2;
        int time = 4;
        int dimension = 8;
        float[] values = new float[batch * time * dimension];
        Random random = new Random(17);
        for (int index = 0; index < values.length; index++) {
            values[index] = random.nextFloat() * 2.0f - 1.0f;
        }
        try (NativeContext context = NativeContext.create(4)) {
            Tensor input = Tensor.of(context, values, batch, time, dimension);
            Tensor target = Tensor.zeros(context, batch, time, dimension);
            MultiheadAttention attention = new MultiheadAttention(context, dimension, 2, true,
                    0.0f, 21);
            AdamW optimizer = new AdamW(attention.parameters(), 0.01f);
            float first = 0.0f;
            float last = 0.0f;
            for (int step = 0; step < 40; step++) {
                Tensor output = attention.forward(input);
                if (step == 0) {
                    check(Arrays.equals(output.shape(), new int[]{batch, time, dimension}),
                            "multi-head output shape");
                }
                Tensor loss = output.mse(target);
                if (step == 0) {
                    first = loss.data()[0];
                }
                last = loss.data()[0];
                loss.backward();
                optimizer.step();
                optimizer.zeroGrad();
                context.clearGraph();
            }
            check(last < first * 0.5f,
                    "multi-head loss must at least halve: " + first + " -> " + last);
            attention.close();
            input.close();
            target.close();
        }
    }

    private static void testTypedPipeline() {
        try (NativeContext context = NativeContext.create(1)) {
            Sequential<BCHW, BC> network = Sequential.<BCHW>start()
                    .add(new Conv2d(context, 1, 2, 3, 3, 1, 1, 1, 1, true, 1))
                    .add(new Activation<BCHW>(Activation.Kind.RELU))
                    .add(new MaxPool2d(2))
                    .add(new Flatten())
                    .add(new Linear(context, 2 * 2 * 2, 3));
            Tensor input = Tensor.zeros(context, 2, 1, 4, 4);
            Tensor output = network.forward(input);
            assertArray(new int[]{2, 3}, output.shape(), "typed pipeline output shape");
            check(network.depth() == 5, "typed pipeline depth");
            network.close();
            input.close();
        }
    }

    private static void testStateDict() {
        try (NativeContext context = NativeContext.create(1)) {
            Linear original = new Linear(context, 3, 2, true, 9);
            Map<String, float[]> state = original.stateDict();
            Linear restored = new Linear(context, 3, 2, true, 10);
            restored.loadStateDict(state);
            Tensor input = Tensor.of(context, new float[]{0.5f, -0.2f, 0.1f}, 1, 3);
            assertArray(original.forward(input).data(), restored.forward(input).data(), 0.0f,
                    "state dict round-trip");
            original.close();
            restored.close();
            input.close();
        }
    }

    private static void checkGradient(NativeContext context, Tensor parameter,
                                      Supplier<Tensor> lossFactory, float epsilon, float tolerance,
                                      String name) {
        context.clearGraph();
        parameter.zeroGrad();
        Tensor loss = lossFactory.get();
        loss.backward();
        float[] analytic = parameter.grad();
        float[] values = parameter.data();
        float[] numeric = new float[values.length];
        for (int index = 0; index < values.length; index++) {
            float original = values[index];
            values[index] = original + epsilon;
            parameter.setData(values);
            float plus = evaluate(context, lossFactory);
            values[index] = original - epsilon;
            parameter.setData(values);
            float minus = evaluate(context, lossFactory);
            values[index] = original;
            parameter.setData(values);
            numeric[index] = (plus - minus) / (2.0f * epsilon);
        }
        assertArray(numeric, analytic, tolerance, name);
    }

    private static float evaluate(NativeContext context, Supplier<Tensor> lossFactory) {
        context.clearGraph();
        return lossFactory.get().data()[0];
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
            check(Float.isFinite(actual[index])
                            && difference <= tolerance * (1.0f + Math.abs(expected[index])),
                    String.format(Locale.ROOT,
                            "%s at %d: expected %.6f but got %.6f (tolerance %.4f)",
                            message, index, expected[index], actual[index], tolerance));
        }
    }

    private static void assertArray(int[] expected, int[] actual, String message) {
        check(Arrays.equals(expected, actual),
                message + ": expected " + Arrays.toString(expected)
                        + " but got " + Arrays.toString(actual));
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
}
