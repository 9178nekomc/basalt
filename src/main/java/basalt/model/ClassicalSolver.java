package basalt.model;

import basalt.core.Matrix;
import basalt.ml.DataSplit;
import basalt.ml.Estimator;
import basalt.ml.Estimators;
import basalt.ml.Registry;
import java.util.List;
import java.util.Locale;

/**
 * Adapts a registered {@link Estimator} to the {@link Solver} contract.
 *
 * <p>Scoring uses k-fold cross-validation — these models are cheap enough that a full sweep costs
 * nothing next to a single epoch of a network, and cross-validation is far less noisy than one
 * holdout split when the dataset is small.</p>
 */
public final class ClassicalSolver implements Solver {

    private final String algorithm;
    private final Task task;

    /**
     * @param algorithm registry name, case-insensitive
     * @throws IllegalArgumentException if the name is unknown, or names an unsupervised algorithm,
     *                                  which has no score to compare
     */
    public ClassicalSolver(String algorithm) {
        if (algorithm == null || algorithm.isBlank()) {
            throw new IllegalArgumentException("algorithm must not be blank");
        }
        if (!Registry.contains(algorithm)) {
            throw new IllegalArgumentException("unknown algorithm '" + algorithm + "'");
        }
        String key = algorithm.toLowerCase(Locale.ROOT);
        Estimators.Task legacy = null;
        for (Registry.Entry entry : Registry.all()) {
            if (entry.name().equals(key)) {
                legacy = entry.task();
            }
        }
        if (legacy == null || legacy == Estimators.Task.UNSUPERVISED) {
            throw new IllegalArgumentException("algorithm '" + key
                    + "' is unsupervised; it has no score to screen");
        }
        this.algorithm = key;
        this.task = legacy == Estimators.Task.CLASSIFICATION
                ? Task.TABULAR_CLASSIFICATION
                : Task.TABULAR_REGRESSION;
    }

    @Override
    public String name() {
        return algorithm;
    }

    @Override
    public Task task() {
        return task;
    }

    @Override
    public double score(Dataset data, int folds, long seed) {
        requireTask(data);
        Matrix features = data.features();
        double[] targets = data.targets();
        int depth = effectiveFolds(folds, features.rows(), targets, task);
        List<DataSplit.Fold> splits = task == Task.TABULAR_CLASSIFICATION
                ? DataSplit.stratifiedKFold(features, targets, depth, true, seed)
                : DataSplit.kFold(features, targets, depth, seed);
        double total = 0.0;
        for (DataSplit.Fold fold : splits) {
            Estimator estimator = Registry.create(algorithm)
                    .fit(fold.trainFeatures(), fold.trainTargets());
            total += task.scoreOf(task.defaultMetric(), fold.testTargets(),
                    estimator.predict(fold.testFeatures()));
        }
        return total / splits.size();
    }

    @Override
    public Fitted fit(Dataset data, long seed) {
        requireTask(data);
        Estimator estimator = Registry.create(algorithm).fit(data.features(), data.targets());
        return new ClassicalFitted(estimator, task);
    }

    private void requireTask(Dataset data) {
        if (data.task() != task) {
            throw new IllegalArgumentException("solver '" + algorithm + "' solves " + task
                    + " but the dataset poses " + data.task());
        }
    }

    private static int effectiveFolds(int requested, int rows, double[] targets,
                                     Task task) {
        int folds = Math.max(2, Math.min(requested, rows));
        if (task == Task.TABULAR_CLASSIFICATION) {
            int smallest = Integer.MAX_VALUE;
            java.util.Map<Double, Integer> counts = new java.util.LinkedHashMap<>();
            for (double value : targets) {
                counts.merge(value, 1, Integer::sum);
            }
            for (int count : counts.values()) {
                smallest = Math.min(smallest, count);
            }
            folds = Math.min(folds, smallest);
        }
        return Math.max(2, folds);
    }

    @Override
    public String toString() {
        return "ClassicalSolver[" + algorithm + ", " + task + "]";
    }

    /** Wraps a fitted estimator; generation is not part of its vocabulary. */
    private static final class ClassicalFitted implements Fitted {
        private final Estimator estimator;
        private final Task task;

        ClassicalFitted(Estimator estimator, Task task) {
            this.estimator = estimator;
            this.task = task;
        }

        @Override
        public double[] predict(Matrix features) {
            return estimator.predict(features);
        }

        @Override
        public Task task() {
            return task;
        }
    }
}
