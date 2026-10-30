package basalt.model;

import basalt.core.Matrix;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * One dataset, in the shape its {@link Task} expects.
 *
 * <p>Two forms exist: a feature matrix with one target per row, and a token stream. Keeping both in
 * one type is what lets {@link basalt.auto.Auto} screen classical estimators and sequence models
 * with the same call — the dataset knows its task, and only solvers that declare the same task are
 * considered.</p>
 *
 * <p>Instances are immutable views: the tabular factories keep the caller's {@link Matrix} by
 * reference (copying a large matrix on every screening run would be wasteful), the sequence factory
 * copies the id array because it is small and often built inline.</p>
 */
public final class Dataset {

    private static final int MAX_CLASSES = 20;

    private final Task task;
    private final Matrix features;
    private final double[] targets;
    private final int[] tokens;

    private Dataset(Task task, Matrix features, double[] targets, int[] tokens) {
        this.task = task;
        this.features = features;
        this.targets = targets;
        this.tokens = tokens;
    }

    /**
     * Tabular dataset whose task is inferred from the targets.
     *
     * <p>Targets that are all finite, all whole numbers, and take at most {@value #MAX_CLASSES}
     * distinct values (and no more than one value per five rows) are treated as classes; anything
     * else is treated as a continuous quantity. Use {@link #tabular(Matrix, double[], Task)} when
     * the inference is wrong — an integral regression target, for instance.</p>
     *
     * @throws IllegalArgumentException on null or mismatched input, non-finite values, fewer than
     *                                  four rows, or a single distinct target value
     */
    public static Dataset tabular(Matrix features, double[] targets) {
        return tabular(features, targets, null);
    }

    /**
     * Tabular dataset with an explicit task.
     *
     * @param forced {@code TABULAR_CLASSIFICATION} or {@code TABULAR_REGRESSION}; {@code null} to
     *               infer
     * @throws IllegalArgumentException if {@code forced} is a non-tabular task
     */
    public static Dataset tabular(Matrix features, double[] targets, Task forced) {
        if (features == null) {
            throw new IllegalArgumentException("features must not be null");
        }
        if (targets == null) {
            throw new IllegalArgumentException("targets must not be null");
        }
        if (features.rows() != targets.length) {
            throw new IllegalArgumentException("features has " + features.rows()
                    + " rows but targets has " + targets.length + " values");
        }
        if (features.rows() < 4) {
            throw new IllegalArgumentException("at least 4 rows are required, got "
                    + features.rows());
        }
        if (features.columns() < 1) {
            throw new IllegalArgumentException("features must have at least one column");
        }
        for (int row = 0; row < features.rows(); row++) {
            for (int column = 0; column < features.columns(); column++) {
                if (!Double.isFinite(features.get(row, column))) {
                    throw new IllegalArgumentException("features[" + row + "][" + column
                            + "] is not finite");
                }
            }
        }
        for (int row = 0; row < targets.length; row++) {
            if (!Double.isFinite(targets[row])) {
                throw new IllegalArgumentException("targets[" + row + "] is not finite");
            }
        }
        Task task = forced != null ? forced : infer(targets);
        if (!task.isTabular()) {
            throw new IllegalArgumentException("tabular data cannot carry task " + task);
        }
        return new Dataset(task, features, targets, null);
    }

    /**
     * Token stream for next-token prediction.
     *
     * <p>Ids must be non-negative; the vocabulary size is taken as {@code max(id) + 1} by the
     * solvers, so gaps in the id space cost parameters but never break anything.</p>
     *
     * @throws IllegalArgumentException on null, fewer than eight tokens, or a negative id
     */
    public static Dataset sequence(int[] tokens) {
        if (tokens == null) {
            throw new IllegalArgumentException("tokens must not be null");
        }
        if (tokens.length < 8) {
            throw new IllegalArgumentException("at least 8 tokens are required, got "
                    + tokens.length);
        }
        for (int index = 0; index < tokens.length; index++) {
            if (tokens[index] < 0) {
                throw new IllegalArgumentException("tokens[" + index + "] is negative: "
                        + tokens[index]);
            }
        }
        return new Dataset(Task.SEQUENCE_LM, null, null, tokens.clone());
    }

    /** Task this dataset poses. */
    public Task task() {
        return task;
    }

    /** Feature matrix of a tabular dataset. */
    public Matrix features() {
        requireTabular();
        return features;
    }

    /** Targets of a tabular dataset. */
    public double[] targets() {
        requireTabular();
        return targets;
    }

    /** Token stream of a sequence dataset. */
    public int[] tokens() {
        if (tokens == null) {
            throw new IllegalStateException("this dataset is tabular, not a token stream");
        }
        return tokens.clone();
    }

    /** Number of rows for tabular data, number of tokens for a stream. */
    public int rows() {
        return task.isTabular() ? features.rows() : tokens.length;
    }

    /** Vocabulary implied by a token stream: {@code max(id) + 1}. */
    public int vocabulary() {
        int largest = -1;
        for (int token : tokens) {
            largest = Math.max(largest, token);
        }
        return largest + 1;
    }

    private void requireTabular() {
        if (!task.isTabular()) {
            throw new IllegalStateException("this dataset is a token stream, not tabular");
        }
    }

    private static Task infer(double[] targets) {
        Set<Double> distinct = new LinkedHashSet<>();
        boolean integral = true;
        for (double value : targets) {
            if (value != Math.rint(value)) {
                integral = false;
            }
            if (distinct.size() <= MAX_CLASSES) {
                distinct.add(value);
            }
        }
        if (distinct.size() == 1) {
            throw new IllegalArgumentException("targets contain a single distinct value; "
                    + "there is nothing to learn");
        }
        int ceiling = Math.min(MAX_CLASSES, Math.max(2, targets.length / 5));
        if (integral && distinct.size() <= ceiling) {
            return Task.TABULAR_CLASSIFICATION;
        }
        return Task.TABULAR_REGRESSION;
    }

    @Override
    public String toString() {
        return task.isTabular()
                ? "Dataset[" + task + ", " + features.rows() + "x" + features.columns() + "]"
                : "Dataset[" + task + ", " + tokens.length + " tokens, vocab " + vocabulary() + "]";
    }
}
