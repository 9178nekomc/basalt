package basalt.auto;

import basalt.core.Matrix;
import basalt.ml.Estimators;
import basalt.model.Dataset;
import basalt.model.Fitted;
import basalt.model.Solver;
import basalt.model.Solvers;
import basalt.model.Task;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

/**
 * Picks a model for you — for tabular data and for token streams alike.
 *
 * <p>Screening walks {@link Solvers} for the dataset's task, so classical estimators and neural
 * candidates compete in the same table. A decision forest, a dense network and a windowed
 * character-level model are ranked by one loop; adding a solver to the catalogue is all it takes for
 * a new family to join.</p>
 *
 * <pre>{@code
 * AutoModel tabular = Auto.quickstart(features, labels);
 * AutoModel language = Auto.sequence(tokenIds);
 * }</pre>
 *
 * <p>The task comes from the dataset, not from a guess made here: {@link Dataset#tabular} infers it
 * from the targets and {@link Dataset#sequence} declares it. Pass an explicit task to
 * {@code Dataset.tabular} when the inference is wrong.</p>
 */
public final class Auto {

    private Auto() {
    }

    /** Screens the catalogue for tabular data, with default settings. */
    public static AutoModel quickstart(double[][] features, double[] targets) {
        return solve(Dataset.tabular(Matrix.fromRows(features), targets));
    }

    /** Screens the catalogue for tabular data, with default settings. */
    public static AutoModel quickstart(Matrix features, double[] targets) {
        return solve(Dataset.tabular(features, targets));
    }

    /** Screens the catalogue for a token stream. */
    public static AutoModel sequence(int[] tokens) {
        return solve(Dataset.sequence(tokens));
    }

    /** Screens every solver registered for the dataset's task and fits the winner. */
    public static AutoModel solve(Dataset data) {
        return builder(data).fit();
    }

    /** Starts a configured run. */
    public static Builder builder(Matrix features, double[] targets) {
        return new Builder(Dataset.tabular(features, targets));
    }

    /** Starts a configured run. */
    public static Builder builder(Dataset data) {
        return new Builder(data);
    }

    /** Fluent configuration for one screening run. */
    public static final class Builder {
        private final Dataset data;
        private long seed = 42L;
        private int folds = 5;
        private Estimators.Metric metric;
        private List<String> candidates;

        private Builder(Dataset data) {
            this.data = data;
        }

        /** Seed for splits, folds and any randomised candidate. */
        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        /** Requested cross-validation depth; classical solvers honour it, neural ones do not. */
        public Builder folds(int folds) {
            this.folds = folds;
            return this;
        }

        /** Scoring metric; defaults to the one the task declares. */
        public Builder metric(Estimators.Metric metric) {
            this.metric = metric;
            return this;
        }

        /** Restricts screening to these solver names; defaults to the whole task. */
        public Builder candidates(String... names) {
            if (names == null || names.length == 0) {
                throw new IllegalArgumentException("candidates must not be empty");
            }
            this.candidates = Arrays.asList(names);
            return this;
        }

        /**
         * Scores every candidate, refits the winner on all data and returns it.
         *
         * @throws IllegalArgumentException if a named candidate is unknown or solves a different
         *                                  task, or if no candidate could be scored at all
         */
        public AutoModel fit() {
            if (folds < 2) {
                throw new IllegalArgumentException("folds must be at least 2");
            }
            Task task = data.task();
            Estimators.Metric resolvedMetric = metric != null ? metric : task.defaultMetric();
            List<Solver> chosen = resolve(task);

            List<Candidate> scored = new ArrayList<>(chosen.size());
            for (Solver solver : chosen) {
                try {
                    scored.add(new Candidate(solver.name(), solver.score(data, folds, seed), false));
                } catch (RuntimeException unusable) {
                    // a candidate that cannot handle this dataset simply does not compete
                }
            }
            if (scored.isEmpty()) {
                throw new IllegalArgumentException("no candidate could be scored on this dataset");
            }
            scored.sort(Comparator.comparingDouble(Candidate::score).reversed());

            String winner = scored.get(0).algorithm();
            Fitted fitted = Solvers.get(winner).fit(data, seed);

            List<Candidate> table = new ArrayList<>(scored.size());
            for (int index = 0; index < scored.size(); index++) {
                Candidate candidate = scored.get(index);
                table.add(new Candidate(candidate.algorithm(), candidate.score(), index == 0));
            }
            int columns = task.isTabular() ? data.features().columns() : data.vocabulary();
            return new AutoModel(fitted, winner, task, resolvedMetric, scored.get(0).score(),
                    data.rows(), columns, table);
        }

        private List<Solver> resolve(Task task) {
            if (candidates == null) {
                List<Solver> available = Solvers.byTask(task);
                if (available.isEmpty()) {
                    throw new IllegalArgumentException("no solver registered for " + task);
                }
                return available;
            }
            List<Solver> resolved = new ArrayList<>(candidates.size());
            for (String name : candidates) {
                if (!Solvers.contains(name)) {
                    throw new IllegalArgumentException("unknown candidate '" + name + "'");
                }
                Solver solver = Solvers.get(name.toLowerCase(Locale.ROOT));
                if (solver.task() != task) {
                    throw new IllegalArgumentException("candidate '" + name + "' solves "
                            + solver.task() + " but the dataset poses " + task);
                }
                resolved.add(solver);
            }
            return resolved;
        }
    }
}
