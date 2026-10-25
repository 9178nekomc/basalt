package jsl.tests;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import jsl.JSL;
import jsl.core.BC;
import jsl.core.BCHW;
import jsl.nn.Activation;
import jsl.nativebridge.CommandQueueBuilder;
import jsl.nativebridge.NativeContext;
import jsl.nativebridge.NativeLibrary;
import jsl.nativebridge.NativeMLP;
import jsl.nativebridge.NativeTensorHandle;
import jsl.torch.CrossEntropyLoss;
import jsl.torch.Flatten;
import jsl.torch.Linear;
import jsl.torch.MaxPool2d;
import jsl.torch.Sequential;
import jsl.torch.Tensor;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** JUnit 5 performance smoke tests; CSV benchmarks remain available via benchmark.sh. */
@Tag("performance")
class PerformanceTest {
    @Test
    void nativeGemmThroughput() {
        Assumptions.assumeTrue(NativeLibrary.available(), NativeLibrary.loadError());
        int size = 256;
        Random random = new Random(11);
        float[] left = new float[size * size];
        float[] right = new float[size * size];
        for (int index = 0; index < left.length; index++) {
            left[index] = random.nextFloat() * 2.0f - 1.0f;
            right[index] = random.nextFloat() * 2.0f - 1.0f;
        }
        try (NativeContext context = NativeContext.create(8)) {
            NativeTensorHandle a = context.of(left, false, size, size);
            NativeTensorHandle b = context.of(right, false, size, size);
            NativeTensorHandle c = context.zeros(size, size);
            CommandQueueBuilder queue = new CommandQueueBuilder(context);
            context.clearGraph();
            queue.matMul(a, b, c).execute();
            queue.clear();
            double[] times = new double[5];
            for (int repetition = 0; repetition < times.length; repetition++) {
                context.clearGraph();
                queue.matMul(a, b, c);
                long start = System.nanoTime();
                queue.execute();
                times[repetition] = (System.nanoTime() - start) / 1e6;
                queue.clear();
            }
            Arrays.sort(times);
            double gflops = 2.0 * size * size * size / (times[2] * 1e6);
            assertTrue(times[2] > 0.0 && times[2] < 5000.0, "GEMM time " + times[2]);
            assertTrue(gflops > 1.0, "GEMM throughput " + gflops + " GF");
            assertTrue(Float.isFinite(c.data()[0]));
            a.close();
            b.close();
            c.close();
        }
    }

    @Test
    void mlpXorConvergence() {
        Assumptions.assumeTrue(NativeLibrary.available(), NativeLibrary.loadError());
        float[][] x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        float[][] y = {{0}, {1}, {1}, {0}};
        try (NativeContext context = NativeContext.create(4);
             NativeMLP mlp = new NativeMLP(context, new int[]{2, 12, 1},
                     NativeMLP.Activation.TANH, NativeMLP.OutputActivation.SIGMOID, 42)) {
            float loss = mlp.fit(x, y, 2000, 4, 0.05f, 0.0f);
            assertTrue(loss < 0.05f, "XOR loss " + loss);
        }
    }

    @Test
    void cnnConvergence() {
        int samples = 48;
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
        try (JSL jsl = JSL.create(4)) {
            Tensor input = jsl.tensor(images, samples, 1, 8, 8);
            Tensor target = jsl.tensor(labels, samples);
            Sequential<BCHW, BC> network = jsl.<BCHW>sequential()
                    .add(jsl.conv2d(1, 4, 3, 1, 1))
                    .add(jsl.activation(Activation.RELU))
                    .add(new MaxPool2d(2))
                    .add(new Flatten())
                    .add(jsl.linear(4 * 4 * 4, 2));
            CrossEntropyLoss lossFunction = jsl.crossEntropy();
            var optimizer = jsl.adamw(network.parameters(), 0.01f);
            float loss = 0.0f;
            for (int step = 0; step < 80; step++) {
                Tensor logits = network.forward(input);
                Tensor value = lossFunction.loss(logits, target);
                loss = value.data()[0];
                value.backward();
                optimizer.step();
                optimizer.zeroGrad();
                jsl.clearGraph();
            }
            assertTrue(loss < 0.1f, "CNN loss " + loss);
            Tensor logits = network.forward(input);
            float[] probabilities = logits.softmax(1).data();
            int correct = 0;
            for (int sample = 0; sample < samples; sample++) {
                int predicted = probabilities[sample * 2] > probabilities[sample * 2 + 1] ? 0 : 1;
                if (predicted == (int) labels[sample]) {
                    correct++;
                }
            }
            assertEquals(samples, correct);
            network.close();
            input.close();
            target.close();
        }
    }
}
