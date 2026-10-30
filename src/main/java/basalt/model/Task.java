package basalt.model;

import basalt.ml.Estimators;

/**
 * What kind of problem a dataset describes.
 *
 * <p>This is the axis the whole model layer is organised around: a {@link Solver} declares the task
 * it can solve, a {@link Dataset} declares the task it poses, and screening only pairs the two when
 * they match. Adding a new family means adding a constant here and a solver that reports it — no
 * existing solver has to change.</p>
 */
public enum Task {

    /** Rows of features, one class label per row. */
    TABULAR_CLASSIFICATION,

    /** Rows of features, one continuous value per row. */
    TABULAR_REGRESSION,

    /** A stream of token ids; the model predicts the next one. */
    SEQUENCE_LM;

    /** Metric used to rank candidates of this task when the caller does not pick one. */
    public Estimators.Metric defaultMetric() {
        return switch (this) {
            case TABULAR_CLASSIFICATION -> Estimators.Metric.MACRO_F1;
            case TABULAR_REGRESSION -> Estimators.Metric.R2;
            case SEQUENCE_LM -> Estimators.Metric.CROSS_ENTROPY;
        };
    }

    /**
     * Whether a larger score is better.
     *
     * <p>Classification and regression metrics are all reported so that bigger is better; cross
     * entropy is a loss and is negated before comparison, so every stored score follows the same
     * rule and sorting never has to special-case anything.</p>
     */
    public boolean higherIsBetter() {
        return this != SEQUENCE_LM;
    }

    /** Whether this task operates on a feature matrix rather than a token stream. */
    public boolean isTabular() {
        return this == TABULAR_CLASSIFICATION || this == TABULAR_REGRESSION;
    }

    /**
     * Scores predictions against expected values, normalised so that larger is always better.
     *
     * <p>RMSE is negated here rather than at every call site, so a screening table can be sorted in
     * one direction regardless of which metric produced the numbers.</p>
     *
     * @throws IllegalArgumentException for {@code CROSS_ENTROPY}, which needs a distribution rather
     *                                  than a hard prediction and is computed by the sequence
     *                                  solvers directly
     */
    public double scoreOf(basalt.ml.Estimators.Metric metric, double[] expected,
                          double[] predicted) {
        return switch (metric) {
            case ACCURACY -> basalt.ml.Metrics.accuracy(expected, predicted);
            case MACRO_F1 -> basalt.ml.Metrics.macroF1(expected, predicted);
            case R2 -> basalt.ml.Metrics.r2(expected, predicted);
            case RMSE -> -basalt.ml.Metrics.rmse(expected, predicted);
            case CROSS_ENTROPY -> throw new IllegalArgumentException(
                    "cross entropy is computed from logits, not from hard predictions");
        };
    }
}
