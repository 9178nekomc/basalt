package basalt.auto;

import basalt.core.Matrix;
import basalt.ml.DataSplit;
import basalt.ml.Estimator;
import basalt.ml.Estimators;
import basalt.ml.Metrics;
import basalt.ml.Registry;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Picks a model for you.
 *
 * <p>This is the one-call tier: hand it features and targets and it decides whether the problem is
 * classification or regression, screens every registered algorithm of that task with
 * cross-validation, refits the winner on all rows, and reports how it did on a held-out slice.</p>
 *
 * <pre>{@code
 * AutoModel model = Auto.quickstart(features, labels);
 * double[] prediction = model.predict(unseen);
 * System.out.println(model.summary());
 * }</pre>
 *
 * <p>Anything you might want to control is on {@link Builder}: the seed, the number of folds, the
 * held-out fraction, the metric, and which candidates take part. Candidates come from
 * {@link Registry}, so an algorithm registered by your own code is screened alongside the built-ins
 * without any change here.</p>
 *
 * <p>The task is inferred from the targets: every value must be finite, and if all of them are
 * whole numbers with at most {@value #MAX_CLASSES} distinct values, the run is treated as
 * classification. Pass {@code task(...)} when the inference is wrong — a regression target that
 * happens to be integral, for example.</p>
 */
public final class Auto {

    /** Upper bound on distinct values for the classification inference. */
    public static final int MAX_CLASSES = 20;

    private Auto() {
    }

    /** Screens the catalogue and fits the winner, using default settings. */
    public static AutoModel quickstart(double[][] features, double[] targets) {
        return quickstart(Matrix.fromRows(features), targets);
    }

    /** Screens the catalogue and fits the winner, using default settings. */
    public static AutoModel quickstart(Matrix features, double[] targets) {
        return builder(features, targets).fit();
    }

    /** Starts a configured run. */
    public static Builder builder(Matrix features, double[] targets) {
        return new Builder(features, targets);
    }

    /** Fluent configuration for one screening run. */
    public static final class Builder {
        private final Matrix features;
        private final double[] targets;
        private long seed = 42L;
        private int folds = 5;
        private double testFraction = 0.25;
        private Estimators.Metric metric;
        private Estimators.Task task;
        private List<String> candidates;

        private Builder(Matrix features, double[] targets) {
            this.features = features;
            this.targets = targets;
        }

        /** Seed for the split, the folds and any estimator that randomises. */
        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        /** Requested number of cross-validation folds; clamped to what the data allows. */
        public Builder folds(int folds) {
            this.folds = folds;
            return this;
        }

        /** Fraction held out for the final score, in {@code (0, 1)}. */
        public Builder testFraction(double testFraction) {
            this.testFraction = testFraction;
            return this;
        }

        /** Scoring metric; defaults to {@code MACRO_F1} for classification and {@code R2}. */
        public Builder metric(Estimators.Metric metric) {
            this.metric = metric;
            return this;
        }

        /** Forces the task instead of inferring it from the targets. */
        public Builder task(Estimators.Task task) {
            this.task = task;
            return this;
        }

        /** Restricts screening to these registry names; defaults to the whole task. */
        public Builder candidates(String... names) {
            if (names == null || names.length == 0) {
                throw new IllegalArgumentException("candidates must not be empty");
            }
            this.candidates = Arrays.asList(names);
            return this;
        }

        /**
         * Runs screening and fits the winner.
         *
         * @throws IllegalArgumentException if the data is unusable, the task cannot be inferred,
         *                                  a named candidate is unknown or belongs to another task,
         *                                  or every candidate fails
         */
        public AutoModel fit() {
            validate();
            Estimators.Task resolvedTask = task != null ? task : detectTask(targets);
            if (resolvedTask == Estimators.Task.UNSUPERVISED) {
                throw new IllegalArgumentException(
                        "Auto screens supervised algorithms only; pass CLASSIFICATION or REGRESSION");
            }
            Estimators.Metric resolvedMetric = metric != null ? metric : defaultMetric(resolvedTask);

            DataSplit.Holdout split = DataSplit.holdout(features, targets,
                    clampFraction(), seed);
            List<String> names = resolveCandidates(resolvedTask);
            List<DataSplit.Fold> foldList = folds(split.trainFeatures(), split.trainTargets(),
                    resolvedTask);

            List<Candidate> scored = new ArrayList<>(names.size());
            for (String name : names) {
                double total = 0.0;
                int used = 0;
                for (DataSplit.Fold fold : foldList) {
                    Estimator estimator = Registry.create(name);
                    estimator.fit(fold.trainFeatures(), fold.trainTargets());
                    double[] prediction = estimator.predict(fold.testFeatures());
                    total += scoreOf(resolvedMetric, fold.testTargets(), prediction);
                    used++;
                }
                if (used > 0) {
                    scored.add(new Candidate(name, total / used, false));
                }
            }
            if (scored.isEmpty()) {
                throw new IllegalArgumentException("no candidate could be fitted on this data");
            }
            scored.sort(Comparator.comparingDouble(Candidate::score).reversed());

            String winner = scored.get(0).algorithm();
            Estimator fitted = Registry.create(winner).fit(features, targets);

            double heldOut = scored.get(0).score();
            if (split.testFeatures().rows() > 0) {
                heldOut = scoreOf(resolvedMetric, split.testTargets(),
                        fitted.predict(split.testFeatures()));
            }

            List<Candidate> table = new ArrayList<>(scored.size());
            for (int index = 0; index < scored.size(); index++) {
                Candidate candidate = scored.get(index);
                table.add(new Candidate(candidate.algorithm(), candidate.score(), index == 0));
            }
            return new AutoModel(fitted, winner, resolvedTask, resolvedMetric, heldOut,
                    features.rows(), features.columns(), table);
        }

        private void validate() {
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
            if (!(testFraction > 0.0) || testFraction >= 1.0) {
                throw new IllegalArgumentException("testFraction must be in (0, 1)");
            }
            if (folds < 2) {
                throw new IllegalArgumentException("folds must be at least 2");
            }
        }

        private double clampFraction() {
            // keep at least one row on each side of the split
            int rows = features.rows();
            double minimum = 1.0 / rows;
            return Math.max(testFraction, minimum);
        }

        private List<String> resolveCandidates(Estimators.Task resolvedTask) {
            if (candidates == null) {
                List<String> all = new ArrayList<>();
                for (Registry.Entry entry : Registry.byTask(resolvedTask)) {
                    all.add(entry.name());
                }
                if (all.isEmpty()) {
                    throw new IllegalArgumentException(
                            "no registered algorithm solves " + resolvedTask);
                }
                return all;
            }
            List<String> resolved = new ArrayList<>(candidates.size());
            for (String name : candidates) {
                if (!Registry.contains(name)) {
                    throw new IllegalArgumentException("unknown candidate '" + name + "'");
                }
                Estimators.Task candidateTask = null;
                for (Registry.Entry entry : Registry.byTask(resolvedTask)) {
                    if (entry.name().equals(name.toLowerCase(Locale.ROOT))) {
                        candidateTask = entry.task();
                    }
                }
                if (candidateTask != resolvedTask) {
                    throw new IllegalArgumentException("candidate '" + name + "' does not solve "
                            + resolvedTask);
                }
                resolved.add(name.toLowerCase(Locale.ROOT));
            }
            return resolved;
        }

        private List<DataSplit.Fold> folds(Matrix trainFeatures, double[] trainTargets,
                                           Estimators.Task resolvedTask) {
            int rows = trainFeatures.rows();
            int effective = Math.min(folds, rows);
            if (resolvedTask == Estimators.Task.CLASSIFICATION) {
                effective = Math.min(effective, smallestClassCount(trainTargets));
                if (effective < 2) {
                    // every class appears once; stratification cannot work, fall back to plain folds
                    effective = Math.min(folds, rows);
                }
            }
            effective = Math.max(2, effective);
            if (resolvedTask == Estimators.Task.CLASSIFICATION) {
                return DataSplit.stratifiedKFold(trainFeatures, trainTargets, effective, true, seed);
            }
            return DataSplit.kFold(trainFeatures, trainTargets, effective, seed);
        }
    }

    static Estimators.Task detectTask(double[] targets) {
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
            return Estimators.Task.CLASSIFICATION;
        }
        return Estimators.Task.REGRESSION;
    }

    static Estimators.Metric defaultMetric(Estimators.Task task) {
        return task == Estimators.Task.CLASSIFICATION
                ? Estimators.Metric.MACRO_F1
                : Estimators.Metric.R2;
    }

    static double scoreOf(Estimators.Metric metric, double[] expected, double[] predicted) {
        return switch (metric) {
            case ACCURACY -> Metrics.accuracy(expected, predicted);
            case MACRO_F1 -> Metrics.macroF1(expected, predicted);
            case R2 -> Metrics.r2(expected, predicted);
            // negated so that "higher is better" holds for every metric
            case RMSE -> -Metrics.rmse(expected, predicted);
        };
    }

    private static int smallestClassCount(double[] targets) {
        java.util.Map<Double, Integer> counts = new java.util.LinkedHashMap<>();
        for (double value : targets) {
            counts.merge(value, 1, Integer::sum);
        }
        int smallest = Integer.MAX_VALUE;
        for (int count : counts.values()) {
            smallest = Math.min(smallest, count);
        }
        return smallest == Integer.MAX_VALUE ? 0 : smallest;
    }
}
