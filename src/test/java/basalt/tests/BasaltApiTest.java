package basalt.tests;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import basalt.Basalt;
import basalt.BasaltRuntimeException;
import basalt.core.BC;
import basalt.core.BCHW;
import basalt.core.Matrix;
import basalt.ml.Estimator;
import basalt.nn.Activation;
import basalt.nativebridge.CommandQueueBuilder;
import basalt.nativebridge.NativeContext;
import basalt.nativebridge.NativeLibrary;
import basalt.nativebridge.NativeTensorHandle;
import basalt.torch.Conv2d;
import basalt.torch.CrossEntropyLoss;
import basalt.torch.Flatten;
import basalt.torch.Graph;
import basalt.torch.Linear;
import basalt.torch.MaxPool2d;
import basalt.torch.Sequential;
import basalt.torch.Tensor;
import org.junit.jupiter.api.Test;

/** JUnit 5 contract tests for the public Maven SDK. */
class BasaltApiTest {
    @Test
    void nativeRuntimeIsPackagedAndLoadable() {
        assertTrue(NativeLibrary.available(), NativeLibrary.loadError());
        assertEquals(2, NativeLibrary.version());
    }

    @Test
    void quickStartForwardAndBackward() {
        try (Basalt engine = Basalt.create()) {
            Tensor x = engine.randn(8, 4);
            Tensor y = engine.randn(8, 1);
            Linear hidden = engine.linear(4, 8);
            Linear output = engine.linear(8, 1);
            Tensor loss = output.forward(hidden.forward(x).relu()).mse(y);
            loss.backward();
            assertTrue(Float.isFinite(loss.data()[0]));
            assertTrue(hidden.weight().gradNorm() > 0.0);
            assertTrue(output.weight().gradNorm() > 0.0);
        }
    }

    @Test
    void graphWrapperControlsBackwardAndClear() {
        try (Basalt engine = Basalt.create()) {
            Tensor x = engine.randn(4, 3);
            Tensor y = engine.randn(4, 1);
            Linear linear = engine.linear(3, 1);
            Tensor loss = linear.forward(x).mse(y);
            Graph graph = engine.graph();
            graph.backward(loss);
            assertTrue(graph.size() > 0);
            assertTrue(linear.weight().gradNorm() > 0.0);
            graph.clear();
            assertEquals(0, graph.size());
        }
    }

    @Test
    void fluentTensorChain() {
        try (Basalt engine = Basalt.create()) {
            Tensor a = engine.tensor(new float[]{1, 2, 3, 4}, 2, 2);
            Tensor b = engine.tensor(new float[]{1, 0, 0, 1}, 2, 2);
            Tensor bias = engine.tensor(new float[]{1, 1}, 1, 2);
            Tensor c = a.matMul(b).add(bias).relu();
            assertArrayEquals(new int[]{2, 2}, c.shape());
            assertArrayEquals(new float[]{2, 3, 4, 5}, c.data(), 1e-6f);
        }
    }

    @Test
    void handWrittenCnnForwardAndBackward() {
        try (Basalt engine = Basalt.create(2)) {
            Tensor input = engine.randn(2, 1, 4, 4);
            Tensor labels = engine.tensor(new float[]{0, 1}, 2);
            Sequential<BCHW, BC> network = engine.<BCHW>sequential()
                    .add(engine.conv2d(1, 2, 3))
                    .add(engine.activation(Activation.RELU))
                    .add(new MaxPool2d(2))
                    .add(new Flatten())
                    .add(engine.linear(2, 3));
            CrossEntropyLoss lossFunction = engine.crossEntropy();
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
            BasaltRuntimeException failure = assertThrows(BasaltRuntimeException.class,
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
        Estimator knn = Basalt.fit(basalt.ml.Estimators.Algorithm.KNN, features, labels);
        double[] prediction = Basalt.predict(knn, features);
        for (int index = 0; index < labels.length; index++) {
            assertEquals(labels[index], prediction[index], 1e-9);
        }
        Estimator forest = Basalt.fit(basalt.ml.Estimators.Algorithm.RANDOM_FOREST, features, labels);
        assertEquals(labels.length, Basalt.predict(forest, features).length);
    }

    @Test
    void typedModuleStateDictRoundTrip() {
        try (Basalt engine = Basalt.create()) {
            Linear first = engine.linear(3, 4);
            Linear second = engine.linear(3, 4);
            second.loadStateDict(first.stateDict());
            Tensor input = engine.tensor(new float[]{0.1f, -0.2f, 0.3f}, 1, 3);
            assertArrayEquals(first.forward(input).data(), second.forward(input).data(), 0.0f);
        }
    }

    @Test
    void conv2dShapeValidation() {
        try (Basalt engine = Basalt.create()) {
            Conv2d conv = engine.conv2d(1, 2, 3);
            Tensor wrong = engine.zeros(1, 2, 4, 4);
            assertThrows(IllegalArgumentException.class, () -> conv.forward(wrong));
        }
    }
}
