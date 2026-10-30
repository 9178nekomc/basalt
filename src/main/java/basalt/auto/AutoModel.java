package basalt.auto;

import basalt.core.Matrix;
import basalt.ml.Estimators;
import basalt.model.Fitted;
import basalt.model.Task;
import java.util.List;

/**
 * Result of an {@link Auto} run: the winning model, how it was chosen, and how it scored.
 *
 * <p>One type covers every family. {@link #predict} answers tabular models, {@link #generate}
 * answers sequence models, and asking the wrong one throws with a message that names the task —
 * rather than returning an empty array that fails somewhere else.</p>
 *
 * <p>{@link #score} is whatever the winning solver reported: cross-validated for classical models,
 * a single holdout for neural ones. That difference is stated in the screening table rather than
 * averaged away, because a cross-validated 0.91 and a holdout 0.91 are not the same evidence.</p>
 */
public final class AutoModel {

    private final Fitted fitted;
    private final String algorithm;
    private final Task task;
    private final Estimators.Metric metric;
    private final double score;
    private final int rows;
    private final int columns;
    private final List<Candidate> candidates;

    AutoModel(Fitted fitted, String algorithm, Task task, Estimators.Metric metric, double score,
              int rows, int columns, List<Candidate> candidates) {
        this.fitted = fitted;
        this.algorithm = algorithm;
        this.task = task;
        this.metric = metric;
        this.score = score;
        this.rows = rows;
        this.columns = columns;
        this.candidates = List.copyOf(candidates);
    }

    /** One prediction per row. Tabular models only. */
    public double[] predict(Matrix features) {
        return fitted.predict(features);
    }

    /** Convenience overload that wraps the rows into a {@link Matrix}. */
    public double[] predict(double[][] features) {
        return fitted.predict(Matrix.fromRows(features));
    }

    /** Continues {@code seed} by {@code steps} tokens. Sequence models only. */
    public int[] generate(int[] seed, int steps) {
        return fitted.generate(seed, steps);
    }

    /** Name of the winning solver. */
    public String algorithm() {
        return algorithm;
    }

    /** Task the run was performed for. */
    public Task task() {
        return task;
    }

    /** Metric the winner was scored with. */
    public Estimators.Metric metric() {
        return metric;
    }

    /** Score of the winner; larger is better for every metric. */
    public double score() {
        return score;
    }

    /** Rows of tabular data, or tokens of a stream. */
    public int rows() {
        return rows;
    }

    /** Feature columns, or vocabulary size for a token stream. */
    public int columns() {
        return columns;
    }

    /** Screening table, ordered by score descending. */
    public List<Candidate> candidates() {
        return candidates;
    }

    /** The trained model, for callers that need to keep it past the report. */
    public Fitted fitted() {
        return fitted;
    }

    /** Releases the native resources held by the model. */
    public void close() {
        fitted.close();
    }

    /** Human-readable report: task, winner, score and the screening table. */
    public String summary() {
        StringBuilder out = new StringBuilder();
        out.append("task     : ").append(task).append('\n');
        out.append("algorithm: ").append(algorithm).append('\n');
        out.append("metric   : ").append(metric).append(" = ")
                .append(String.format("%.4f", score)).append('\n');
        out.append("data     : ").append(rows)
                .append(task.isTabular() ? " rows x " : " tokens, vocab ")
                .append(columns).append(task.isTabular() ? " features" : "").append('\n');
        out.append("screened :\n");
        for (Candidate candidate : candidates) {
            out.append("  ").append(candidate).append('\n');
        }
        return out.toString();
    }

    @Override
    public String toString() {
        return "AutoModel[" + algorithm + ", " + task + ", " + metric + "="
                + String.format("%.4f", score) + "]";
    }
}
