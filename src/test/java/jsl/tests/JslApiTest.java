package jsl.tests;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import jsl.JSL;
import jsl.JSLRuntimeException;
import jsl.core.BC;
import jsl.core.BCHW;
import jsl.core.Matrix;
import jsl.ml.Estimator;
import jsl.nn.Activation;
import jsl.nativebridge.CommandQueueBuilder;
import jsl.nativebridge.NativeContext;
import jsl.nativebridge.NativeLibrary;
import jsl.nativebridge.NativeTensorHandle;
import jsl.torch.Conv2d;
import jsl.torch.CrossEntropyLoss;
import jsl.torch.Flatten;
import jsl.torch.Graph;
import jsl.torch.Linear;
import jsl.torch.MaxPool2d;
import jsl.torch.Sequential;
import jsl.torch.Tensor;
import org.junit.jupiter.api.Test;

/** JUnit 5 contract tests for the public Maven SDK. */
class JslApiTest {
    @Test
    void nativeRuntimeIsPackagedAndLoadable() {
        assertTrue(NativeLibrary.available(), NativeLibrary.loadError());
        assertEquals(2, NativeLibrary.version());
    }

    @Test
    void quickStartForwardAndBackward() {
        try (JSL jsl = JSL.create()) {
            Tensor x = jsl.randn(8, 4);
            Tensor y = jsl.randn(8, 1);
            Linear hidden = jsl.linear(4, 8);
            Linear output = jsl.linear(8, 1);
            Tensor loss = output.forward(hidden.forward(x).relu()).mse(y);
            loss.backward();
            assertTrue(Float.isFinite(loss.data()[0]));
            assertTrue(hidden.weight().gradNorm() > 0.0);
            assertTrue(output.weight().gradNorm() > 0.0);
        }
    }

    @Test
    void graphWrapperControlsBackwardAndClear() {
        try (JSL jsl = JSL.create()) {
            Tensor x = jsl.randn(4, 3);
            Tensor y = jsl.randn(4, 1);
            Linear linear = jsl.linear(3, 1);
            Tensor loss = linear.forward(x).mse(y);
            Graph graph = jsl.graph();
            graph.backward(loss);
            assertTrue(graph.size() > 0);
            assertTrue(linear.weight().gradNorm() > 0.0);
            graph.clear();
            assertEquals(0, graph.size());
        }
    }

    @Test
    void fluentTensorChain() {
        try (JSL jsl = JSL.create()) {
            Tensor a = jsl.tensor(new float[]{1, 2, 3, 4}, 2, 2);
            Tensor b = jsl.tensor(new float[]{1, 0, 0, 1}, 2, 2);
            Tensor bias = jsl.tensor(new float[]{1, 1}, 1, 2);
            Tensor c = a.matMul(b).add(bias).relu();
            assertArrayEquals(new int[]{2, 2}, c.shape());
            assertArrayEquals(new float[]{2, 3, 4, 5}, c.data(), 1e-6f);
        }
    }

    @Test
    void handWrittenCnnForwardAndBackward() {
        try (JSL jsl = JSL.create(2)) {
            Tensor input = jsl.randn(2, 1, 4, 4);
            Tensor labels = jsl.tensor(new float[]{0, 1}, 2);
            Sequential<BCHW, BC> network = jsl.<BCHW>sequential()
                    .add(jsl.conv2d(1, 2, 3))
                    .add(jsl.activation(Activation.RELU))
                    .add(new MaxPool2d(2))
                    .add(new Flatten())
                    .add(jsl.linear(2, 3));
            CrossEntropyLoss lossFunction = jsl.crossEntropy();
            Tensor logits = network.forward(input);
            Tensor loss = lossFunction.loss(logits, labels);
            loss.backward();
            assertTrue(Float.isFinite(loss.data()[0]));
            assertTrue(network.parameters().get(0).gradNorm() > 0.0);
            assertEquals(5, network.depth());
        }
    }

    @Test
    void nativeShapeErrorIsTranslated() {
        try (NativeContext context = NativeContext.create(1)) {
            NativeTensorHandle left = context.zeros(2, 3);
            NativeTensorHandle right = context.zeros(4, 2);
            NativeTensorHandle output = context.zeros(2, 2);
            JSLRuntimeException failure = assertThrows(JSLRuntimeException.class,
                    () -> new CommandQueueBuilder(context).matMul(left, right, output).execute());
            assertTrue(failure.getMessage().contains("MatMul")
                            || failure.getMessage().contains("shape")
                            || failure.getMessage().contains("dimension"),
                    failure.getMessage());
            left.close();
            right.close();
            output.close();
        }
    }

    @Test
    void classicalFacadeKnnAndForest() {
        Matrix features = Matrix.fromRows(new double[][]{
                {-2, -1}, {-1, -1}, {-1, -2}, {1, 1}, {2, 1}, {1, 2}});
        double[] labels = {0, 0, 0, 1, 1, 1};
        Estimator knn = JSL.fit(jsl.ml.JSL.Algorithm.KNN, features, labels);
        double[] prediction = JSL.predict(knn, features);
        for (int index = 0; index < labels.length; index++) {
            assertEquals(labels[index], prediction[index], 1e-9);
        }
        Estimator forest = JSL.fit(jsl.ml.JSL.Algorithm.RANDOM_FOREST, features, labels);
        assertEquals(labels.length, JSL.predict(forest, features).length);
    }

    @Test
    void typedModuleStateDictRoundTrip() {
        try (JSL jsl = JSL.create()) {
            Linear first = jsl.linear(3, 4);
            Linear second = jsl.linear(3, 4);
            second.loadStateDict(first.stateDict());
            Tensor input = jsl.tensor(new float[]{0.1f, -0.2f, 0.3f}, 1, 3);
            assertArrayEquals(first.forward(input).data(), second.forward(input).data(), 0.0f);
        }
    }

    @Test
    void conv2dShapeValidation() {
        try (JSL jsl = JSL.create()) {
            Conv2d conv = jsl.conv2d(1, 2, 3);
            Tensor wrong = jsl.zeros(1, 2, 4, 4);
            assertThrows(IllegalArgumentException.class, () -> conv.forward(wrong));
        }
    }
}
