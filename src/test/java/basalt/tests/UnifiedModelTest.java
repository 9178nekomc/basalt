package basalt.tests;

import basalt.auto.Auto;
import basalt.auto.AutoModel;
import basalt.core.Matrix;
import basalt.ml.Estimator;
import basalt.ml.Estimators;
import basalt.ml.Registry;
import basalt.model.ClassicalSolver;
import basalt.model.Dataset;
import basalt.model.Solver;
import basalt.model.Solvers;
import basalt.model.Task;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The unified layer: one screening loop over classical estimators and neural models. */
class UnifiedModelTest {

    private static final String CORPUS = "问：你好 答：滚木，你好，我听不懂请说人话";

    private static int[] encode(String text) {
        StringBuilder builder = new StringBuilder();
        text.chars().distinct().sorted().forEach(builder::appendCodePoint);
        String alphabet = builder.toString();
        int[] ids = new int[text.length()];
        for (int index = 0; index < ids.length; index++) {
            ids[index] = alphabet.indexOf(text.charAt(index));
        }
        return ids;
    }

    @Test
    void sequenceScreeningTrainsAndGenerates() {
        int[] ids = encode(CORPUS);
        assertEquals(21, ids.length);

        AutoModel model = Auto.sequence(ids);
        assertEquals(Task.SEQUENCE_LM, model.task());
        assertEquals("char_lm", model.algorithm());
        assertEquals(1, model.candidates().size(), "only one sequence solver ships today");

        int window = 6;
        int[] seed = new int[window];
        System.arraycopy(ids, 0, seed, 0, window);
        int[] generated = model.generate(seed, ids.length - window);

        int matched = 0;
        for (int index = 0; index < generated.length; index++) {
            if (generated[index] == ids[window + index]) {
                matched++;
            }
        }
        assertTrue(matched >= generated.length - 2,
                "should reproduce the corpus, matched " + matched + " of " + generated.length);
        model.close();
    }

    @Test
    void tabularScreeningIncludesNeuralCandidates() {
        Random random = new Random(4L);
        int rows = 80;
        double[][] data = new double[rows][3];
        double[] labels = new double[rows];
        for (int row = 0; row < rows; row++) {
            double signal = random.nextGaussian();
            for (int column = 0; column < 3; column++) {
                data[row][column] = signal + 0.2 * random.nextGaussian();
            }
            labels[row] = signal > 0 ? 1 : 0;
        }

        AutoModel model = Auto.quickstart(Matrix.fromRows(data), labels);
        List<String> names = new ArrayList<>();
        for (var candidate : model.candidates()) {
            names.add(candidate.algorithm());
        }
        assertTrue(names.contains("neural_mlp"),
                "the dense candidate must compete with the estimators, got " + names);
        assertTrue(names.contains("logistic_regression") && names.contains("random_forest"),
                "classical classifiers must still be screened, got " + names);
        assertTrue(names.size() >= 8,
                "eight classifiers plus the network should compete, got " + names);
        assertTrue(model.score() > 0.7, "winner should be usable, got " + model.score());
        model.close();
    }

    @Test
    void registeringAnEstimatorMakesItScreenable() {
        Registry.register("test_threshold_voter", Estimators.Task.CLASSIFICATION,
                ThresholdVoter::new);

        assertTrue(Solvers.contains("test_threshold_voter"),
                "Solvers must derive classical candidates from the estimator registry");
        List<String> names = new ArrayList<>();
        for (Solver solver : Solvers.byTask(Task.TABULAR_CLASSIFICATION)) {
            names.add(solver.name());
        }
        assertTrue(names.contains("test_threshold_voter"), "got " + names);

        Solver solver = Solvers.get("test_threshold_voter");
        assertTrue(solver instanceof ClassicalSolver);
        assertEquals(Task.TABULAR_CLASSIFICATION, solver.task());
    }

    @Test
    void capabilitiesAreTaskSpecific() {
        int[] ids = encode(CORPUS);
        AutoModel sequence = Auto.builder(Dataset.sequence(ids))
                .candidates("char_lm")
                .fit();
        assertEquals(Task.SEQUENCE_LM, sequence.task());
        assertThrows(UnsupportedOperationException.class,
                () -> sequence.predict(Matrix.fromRows(new double[][]{{0, 1}})));

        Random random = new Random(8L);
        double[][] rows = new double[40][2];
        double[] targets = new double[40];
        for (int row = 0; row < 40; row++) {
            rows[row][0] = random.nextGaussian();
            rows[row][1] = random.nextGaussian();
            targets[row] = 2 * rows[row][0] + random.nextGaussian() * 0.01;
        }
        AutoModel tabular = Auto.builder(Matrix.fromRows(rows), targets)
                .candidates("linear_regression")
                .fit();
        assertThrows(UnsupportedOperationException.class, () -> tabular.generate(new int[]{0, 1}, 3));
        tabular.close();
        sequence.close();
    }

    @Test
    void candidateFromAnotherTaskIsRejected() {
        int[] ids = encode(CORPUS);
        assertThrows(IllegalArgumentException.class,
                () -> Auto.builder(Dataset.sequence(ids)).candidates("linear_regression").fit());
        assertThrows(IllegalArgumentException.class,
                () -> Auto.builder(Dataset.sequence(ids)).candidates("nope").fit());
    }

    @Test
    void datasetRejectsUnusableInput() {
        assertThrows(IllegalArgumentException.class, () -> Dataset.sequence(new int[]{1, 2, 3}));
        assertThrows(IllegalArgumentException.class,
                () -> Dataset.sequence(new int[]{1, 2, 3, 4, 5, 6, 7, -1}));

        Matrix tiny = Matrix.fromRows(new double[][]{{0}, {1}});
        assertThrows(IllegalArgumentException.class, () -> Dataset.tabular(tiny, new double[]{0, 1}));

        Matrix features = Matrix.fromRows(new double[][]{{0}, {1}, {2}, {3}});
        assertThrows(IllegalArgumentException.class,
                () -> Dataset.tabular(features, new double[]{1, 1, 1, 1}));
    }

    @Test
    void taskInferenceCanBeOverridden() {
        Matrix features = Matrix.fromRows(new double[][]{{0}, {1}, {2}, {3}, {4}, {5}});
        double[] integralTargets = {1, 2, 3, 4, 5, 6};

        assertEquals(Task.TABULAR_REGRESSION, Dataset.tabular(features, integralTargets).task(),
                "six distinct whole numbers over six rows is treated as a quantity, not six classes");
        assertEquals(Task.TABULAR_CLASSIFICATION,
                Dataset.tabular(features, integralTargets, Task.TABULAR_CLASSIFICATION).task());
    }

    @Test
    void tokenIdsRoundTripThroughGeneration() {
        int[] ids = encode("abcdefgh");
        assertEquals(8, ids.length);
        Dataset dataset = Dataset.sequence(ids);
        assertEquals(8, dataset.rows());
        assertEquals(8, dataset.vocabulary());
        assertArrayEquals(ids, dataset.tokens());
        assertFalse(dataset.task().isTabular());
    }

    /** Predicts the majority class: enough to prove the registry link, useless as a model. */
    private static final class ThresholdVoter implements Estimator {
        private double threshold;

        @Override
        public Estimator fit(Matrix features, double[] targets) {
            double total = 0.0;
            for (double value : targets) {
                total += value;
            }
            threshold = targets.length == 0 ? 0.0 : total / targets.length;
            return this;
        }

        @Override
        public double[] predict(Matrix features) {
            double[] prediction = new double[features.rows()];
            for (int row = 0; row < prediction.length; row++) {
                prediction[row] = features.get(row, 0) >= threshold ? 1 : 0;
            }
            return prediction;
        }
    }
}
