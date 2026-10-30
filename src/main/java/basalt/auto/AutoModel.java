package basalt.auto;

import basalt.core.Matrix;
import basalt.ml.Estimator;
import basalt.ml.Estimators;
import java.util.List;

/**
 * Result of a {@link Auto} run: the winning estimator, how it was chosen, and how it scored.
 *
 * <p>The model is already fitted on every row it was given, so {@link #predict} can be called
 * directly. {@link #score} is measured on the held-out slice that took no part in fitting or
 * screening, which is the number worth trusting; the per-candidate scores in {@link #candidates}
 * come from cross-validation on the training slice only.</p>
 */
public final class AutoModel {

    private final Estimator estimator;
    private final String algorithm;
    private final Estimators.Task task;
    private final Estimators.Metric metric;
    private final double score;
    private final int rows;
    private final int columns;
    private final List<Candidate> candidates;

    AutoModel(Estimator estimator, String algorithm, Estimators.Task task,
              Estimators.Metric metric, double score, int rows, int columns,
              List<Candidate> candidates) {
        this.estimator = estimator;
        this.algorithm = algorithm;
        this.task = task;
        this.metric = metric;
        this.score = score;
        this.rows = rows;
        this.columns = columns;
        this.candidates = List.copyOf(candidates);
    }

    /** Predicts one value per row. */
    public double[] predict(Matrix features) {
        return estimator.predict(features);
    }

    /** Convenience overload that wraps the rows into a {@link Matrix}. */
    public double[] predict(double[][] features) {
        return estimator.predict(Matrix.fromRows(features));
    }

    /** Registry name of the winning algorithm. */
    public String algorithm() {
        return algorithm;
    }

    /** Whether the run treated the targets as classes or as a continuous quantity. */
    public Estimators.Task task() {
        return task;
    }

    /** Metric the winner was scored with. */
    public Estimators.Metric metric() {
        return metric;
    }

    /** Held-out score of the winner; higher is better for every metric. */
    public double score() {
        return score;
    }

    /** Training rows the model was fitted on. */
    public int rows() {
        return rows;
    }

    /** Feature columns the model expects. */
    public int columns() {
        return columns;
    }

    /** Screening table, ordered by score descending. */
    public List<Candidate> candidates() {
        return candidates;
    }

    /** The fitted estimator, for callers that need the concrete type. */
    public Estimator estimator() {
        return estimator;
    }

    /** Human-readable report: task, winner, held-out score and the screening table. */
    public String summary() {
        StringBuilder out = new StringBuilder();
        out.append("task     : ").append(task).append('\n');
        out.append("algorithm: ").append(algorithm).append('\n');
        out.append("metric   : ").append(metric).append(" = ")
                .append(String.format("%.4f", score)).append(" (held out)\n");
        out.append("data     : ").append(rows).append(" rows x ").append(columns)
                .append(" features\n");
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
