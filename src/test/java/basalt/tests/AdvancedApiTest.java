package basalt.tests;

import basalt.Basalt;
import basalt.auto.Auto;
import basalt.auto.AutoModel;
import basalt.auto.Candidate;
import basalt.core.BC;
import basalt.core.Matrix;
import basalt.ml.Estimator;
import basalt.ml.Estimators;
import basalt.ml.Registry;
import basalt.torch.Module;
import basalt.torch.Tensor;
import basalt.zoo.History;
import basalt.zoo.Trainer;
import basalt.zoo.Zoo;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Covers the three tiers: one-call Auto, composed Zoo + Trainer, and hand-written modules. */
class AdvancedApiTest {

    private static final int ROWS = 120;
    private static final int FEATURES = 3;

    private static Matrix classificationFeatures() {
        Random random = new Random(7L);
        double[][] rows = new double[ROWS][FEATURES];
        for (int row = 0; row < ROWS; row++) {
            double signal = random.nextGaussian();
            for (int column = 0; column < FEATURES; column++) {
                rows[row][column] = signal + 0.3 * random.nextGaussian();
            }
        }
        return Matrix.fromRows(rows);
    }

    private static double[] classificationTargets(Matrix features) {
        double[] labels = new double[features.rows()];
        for (int row = 0; row < features.rows(); row++) {
            labels[row] = features.get(row, 0) > 0 ? 1.0 : 0.0;
        }
        return labels;
    }

    @Test
    void autoScreensAndFitsClassification() {
        Matrix features = classificationFeatures();
        double[] labels = classificationTargets(features);

        AutoModel model = Auto.builder(features, labels)
                .candidates("logistic_regression", "decision_tree", "knn")
                .fit();

        assertEquals(Estimators.Task.CLASSIFICATION, model.task());
        assertEquals(Estimators.Metric.MACRO_F1, model.metric());
        assertEquals(3, model.candidates().size());
        assertTrue(model.score() > 0.8, "held-out F1 should be high, got " + model.score());

        List<Candidate> table = model.candidates();
        assertTrue(table.get(0).selected(), "the winner must be marked");
        for (int index = 1; index < table.size(); index++) {
            assertTrue(table.get(index - 1).score() >= table.get(index).score(),
                    "candidates must be sorted by score");
        }

        double[] prediction = model.predict(features);
        assertEquals(features.rows(), prediction.length);
        assertTrue(model.summary().contains(model.algorithm()));
    }

    @Test
    void autoDetectsRegressionFromContinuousTargets() {
        Random random = new Random(11L);
        double[][] rows = new double[ROWS][2];
        double[] targets = new double[ROWS];
        for (int row = 0; row < ROWS; row++) {
            double x = random.nextDouble() * 4 - 2;
            rows[row][0] = x;
            rows[row][1] = random.nextGaussian();
            targets[row] = 3.0 * x + 1.0 + 0.01 * random.nextGaussian();
        }

        AutoModel model = Auto.builder(Matrix.fromRows(rows), targets)
                .candidates("linear_regression", "ridge")
                .fit();

        assertEquals(Estimators.Task.REGRESSION, model.task());
        assertEquals(Estimators.Metric.R2, model.metric());
        assertTrue(model.score() > 0.95, "R2 should be near 1, got " + model.score());
    }

    @Test
    void autoRejectsUnusableTargets() {
        Matrix features = classificationFeatures();
        double[] constant = new double[features.rows()];
        assertThrows(IllegalArgumentException.class, () -> Auto.quickstart(features, constant));

        double[] mismatched = new double[features.rows() - 1];
        assertThrows(IllegalArgumentException.class, () -> Auto.quickstart(features, mismatched));

        double[] withNaN = classificationTargets(features);
        withNaN[3] = Double.NaN;
        assertThrows(IllegalArgumentException.class, () -> Auto.quickstart(features, withNaN));
    }

    @Test
    void autoRejectsCandidatesFromAnotherTask() {
        Matrix features = classificationFeatures();
        double[] labels = classificationTargets(features);
        assertThrows(IllegalArgumentException.class,
                () -> Auto.builder(features, labels).candidates("linear_regression").fit());
        assertThrows(IllegalArgumentException.class,
                () -> Auto.builder(features, labels).candidates("does_not_exist").fit());
    }

    @Test
    void registryRejectsDuplicateNames() {
        assertThrows(IllegalArgumentException.class,
                () -> Registry.register("knn", Estimators.Task.CLASSIFICATION, () -> null));
        assertThrows(IllegalArgumentException.class,
                () -> Registry.register("", Estimators.Task.CLASSIFICATION, () -> null));
        assertTrue(Registry.contains("KNN"), "lookup is case-insensitive");
        assertTrue(Registry.byTask(Estimators.Task.CLASSIFICATION).size() >= 6);
        assertTrue(Registry.size() >= 23);
    }

    @Test
    void userRegisteredAlgorithmTakesPartInScreening() {
        // a deliberately trivial estimator: predict the mean of the training targets
        Registry.register("test_constant_regressor", Estimators.Task.REGRESSION, MeanRegressor::new);

        Random random = new Random(3L);
        double[][] rows = new double[60][2];
        double[] targets = new double[60];
        for (int row = 0; row < 60; row++) {
            rows[row][0] = random.nextGaussian();
            rows[row][1] = random.nextGaussian();
            targets[row] = 5.0 + random.nextGaussian();
        }
        AutoModel model = Auto.builder(Matrix.fromRows(rows), targets)
                .candidates("test_constant_regressor")
                .fit();

        assertEquals("test_constant_regressor", model.algorithm());
        assertNotNull(model.estimator());
        double[] prediction = model.predict(Matrix.fromRows(rows));
        assertEquals(60, prediction.length);
        assertTrue(Math.abs(prediction[0] - 5.0) < 1.0, "mean prediction should sit near 5");
    }

    @Test
    void zooMlpTrainsWithTrainer() {
        try (Basalt engine = Basalt.create()) {
            Random random = new Random(5L);
            int rows = 96;
            float[] data = new float[rows * 4];
            float[] labels = new float[rows];
            for (int row = 0; row < rows; row++) {
                double a = random.nextGaussian();
                for (int column = 0; column < 4; column++) {
                    data[row * 4 + column] = (float) (a + 0.1 * random.nextGaussian());
                }
                labels[row] = a > 0 ? 1f : 0f;
            }
            Tensor features = engine.tensor(data, rows, 4);
            Tensor targets = engine.tensor(labels, rows);

            var network = Zoo.mlp(engine, 4, new int[]{16, 8}, 2);
            History history = Trainer.of(engine.context(), network,
                            engine.adamw(network.parameters(), 1e-2f), engine.crossEntropy())
                    .epochs(40)
                    .batchSize(16)
                    .fit(features, targets);

            assertEquals(40, history.epochs());
            assertTrue(history.bestLoss() < 0.2,
                    "cross-entropy should drop below 0.2, got " + history.bestLoss());
            assertTrue(history.bestEpoch() >= 0, "best epoch must be recorded");
            network.close();
        }
    }

    @Test
    void trainerRejectsShapeMismatch() {
        try (Basalt engine = Basalt.create()) {
            var network = Zoo.mlp(engine, 3, new int[]{4}, 2);
            Tensor features = engine.tensor(new float[30], 10, 3);
            Tensor targets = engine.tensor(new float[9], 9);
            Trainer trainer = Trainer.of(engine.context(), network,
                    engine.adamw(network.parameters(), 1e-2f), engine.crossEntropy());
            assertThrows(IllegalArgumentException.class, () -> trainer.fit(features, targets));
            assertThrows(IllegalArgumentException.class,
                    () -> Trainer.of(engine.context(), network,
                            engine.adamw(network.parameters(), 1e-2f), engine.crossEntropy())
                            .epochs(0));
            network.close();
        }
    }

    @Test
    void handWrittenModuleTrainsThroughTheSameLoop() {
        try (Basalt engine = Basalt.create()) {
            Random random = new Random(9L);
            int rows = 64;
            int inputs = 3;
            float[] data = new float[rows * inputs];
            float[] labels = new float[rows];
            for (int row = 0; row < rows; row++) {
                double sum = 0.0;
                for (int column = 0; column < inputs; column++) {
                    float value = (float) random.nextGaussian();
                    data[row * inputs + column] = value;
                    sum += value;
                }
                labels[row] = (float) sum;
            }
            Tensor features = engine.tensor(data, rows, inputs);
            // MSE compares shapes element-wise, so the target is [rows, 1]
            Tensor targets = engine.tensor(labels, rows, 1);

            // a layer written from scratch: y = x W, no bias, no library layer involved
            ScratchLinear layer = new ScratchLinear(engine.context(), inputs, 1);
            History history = Trainer.of(engine.context(), layer,
                            engine.adamw(layer.parameters(), 5e-2f), engine.mse())
                    .epochs(120)
                    .batchSize(rows)
                    .shuffle(false)
                    .fit(features, targets);

            assertTrue(history.lastLoss() < 0.01,
                    "a hand-written layer must train through the shared loop, got "
                            + history.lastLoss());
            layer.close();
        }
    }

    /** Minimal custom layer used to prove that {@link Module} is the only contract needed. */
    private static final class ScratchLinear extends Module<BC, BC> {
        private final Tensor weight;

        ScratchLinear(basalt.nativebridge.NativeContext context, int inputs, int outputs) {
            this.weight = Tensor.parameter(context, inputs, outputs);
            this.weight.fill(0.0f);
        }

        @Override
        public Tensor forward(Tensor input) {
            return input.matMul(weight);
        }

        @Override
        public List<Tensor> parameters() {
            return List.of(weight);
        }
    }

    /** Trivial estimator used to prove the registry extension point. */
    private static final class MeanRegressor implements Estimator {
        private double mean;

        @Override
        public Estimator fit(Matrix features, double[] targets) {
            double total = 0.0;
            for (double value : targets) {
                total += value;
            }
            mean = targets.length == 0 ? 0.0 : total / targets.length;
            return this;
        }

        @Override
        public double[] predict(Matrix features) {
            double[] prediction = new double[features.rows()];
            java.util.Arrays.fill(prediction, mean);
            return prediction;
        }
    }
}
