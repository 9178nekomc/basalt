package basalt.ml;

import basalt.core.Matrix;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.function.Function;

/**
 * Randomized search over hyperparameter distributions or discrete candidate lists.
 */
public final class RandomizedSearchCV implements Estimator {
    @FunctionalInterface
    public interface ParameterSampler {
        Object sample(Random random);
    }

    private final Function<Map<String, Object>, Estimator> estimatorFactory;
    private final Map<String, ParameterSampler> distributions;
    private final int nIter;
    private final Estimators.Metric scoring;
    private final int cvFolds;
    private final boolean stratified;
    private final long seed;

    private Estimator bestEstimator;
    private Map<String, Object> bestParams;
    private double bestScore = Double.NEGATIVE_INFINITY;
    private final List<GridSearchCV.CandidateResult> cvResults = new ArrayList<>();

    private RandomizedSearchCV(Builder builder) {
        this.estimatorFactory = builder.estimatorFactory;
        this.distributions = new LinkedHashMap<>(builder.distributions);
        this.nIter = builder.nIter;
        this.scoring = builder.scoring;
        this.cvFolds = builder.cvFolds;
        this.stratified = builder.stratified;
        this.seed = builder.seed;
    }

    public static Builder builder(Function<Map<String, Object>, Estimator> estimatorFactory) {
        return new Builder(estimatorFactory);
    }

    @Override
    public RandomizedSearchCV fit(Matrix features, double[] targets) {
        MlChecks.trainingData(features, targets);
        List<DataSplit.Fold> folds = stratified
                ? DataSplit.stratifiedKFold(features, targets, cvFolds, true, seed)
                : DataSplit.kFold(features, targets, cvFolds, true, seed);

        Random random = new Random(seed);
        cvResults.clear();
        bestScore = Double.NEGATIVE_INFINITY;
        bestParams = null;

        for (int iter = 0; iter < nIter; iter++) {
            Map<String, Object> params = new LinkedHashMap<>();
            for (Map.Entry<String, ParameterSampler> entry : distributions.entrySet()) {
                params.put(entry.getKey(), entry.getValue().sample(random));
            }
            double[] scores = new double[folds.size()];
            double sum = 0.0;
            for (int f = 0; f < folds.size(); f++) {
                DataSplit.Fold fold = folds.get(f);
                Estimator model = estimatorFactory.apply(params);
                model.fit(fold.trainFeatures(), fold.trainTargets());
                double[] preds = model.predict(fold.testFeatures());
                scores[f] = GridSearchCV.evaluateScore(scoring, fold.testTargets(), preds);
                sum += scores[f];
            }
            double mean = sum / folds.size();
            double var = 0.0;
            for (double s : scores) {
                double d = s - mean;
                var += d * d;
            }
            double std = Math.sqrt(var / folds.size());
            Map<String, Object> immutableParams = Collections.unmodifiableMap(params);
            cvResults.add(new GridSearchCV.CandidateResult(immutableParams, mean, std, scores));
            if (mean > bestScore || bestParams == null) {
                bestScore = mean;
                bestParams = immutableParams;
            }
        }
        bestEstimator = estimatorFactory.apply(bestParams);
        bestEstimator.fit(features, targets);
        return this;
    }

    @Override
    public double[] predict(Matrix features) {
        if (bestEstimator == null) {
            throw new IllegalStateException("RandomizedSearchCV must be fitted first");
        }
        return bestEstimator.predict(features);
    }

    public Estimator bestEstimator() {
        if (bestEstimator == null) throw new IllegalStateException("RandomizedSearchCV is not fitted");
        return bestEstimator;
    }

    public Map<String, Object> bestParams() {
        if (bestParams == null) throw new IllegalStateException("RandomizedSearchCV is not fitted");
        return bestParams;
    }

    public double bestScore() {
        if (bestParams == null) throw new IllegalStateException("RandomizedSearchCV is not fitted");
        return bestScore;
    }

    public List<GridSearchCV.CandidateResult> cvResults() {
        return Collections.unmodifiableList(cvResults);
    }

    public static final class Builder {
        private final Function<Map<String, Object>, Estimator> estimatorFactory;
        private final Map<String, ParameterSampler> distributions = new LinkedHashMap<>();
        private int nIter = 10;
        private Estimators.Metric scoring = Estimators.Metric.ACCURACY;
        private int cvFolds = 5;
        private boolean stratified = false;
        private long seed = 42L;

        private Builder(Function<Map<String, Object>, Estimator> estimatorFactory) {
            this.estimatorFactory = Objects.requireNonNull(estimatorFactory, "estimatorFactory");
        }

        public Builder paramChoices(String name, Object... choices) {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(choices, "choices");
            if (choices.length == 0) {
                throw new IllegalArgumentException("choices must not be empty");
            }
            Object[] copy = choices.clone();
            distributions.put(name, rng -> copy[rng.nextInt(copy.length)]);
            return this;
        }

        public Builder paramUniform(String name, double low, double high) {
            if (!(high > low)) throw new IllegalArgumentException("high must be > low");
            distributions.put(name, rng -> low + rng.nextDouble() * (high - low));
            return this;
        }

        public Builder paramLogUniform(String name, double low, double high) {
            if (low <= 0.0 || !(high > low)) {
                throw new IllegalArgumentException("log-uniform requires 0 < low < high");
            }
            double logLow = Math.log(low);
            double logHigh = Math.log(high);
            distributions.put(name, rng -> Math.exp(logLow + rng.nextDouble() * (logHigh - logLow)));
            return this;
        }

        public Builder paramIntRange(String name, int lowInclusive, int highInclusive) {
            if (highInclusive < lowInclusive) {
                throw new IllegalArgumentException("highInclusive must be >= lowInclusive");
            }
            distributions.put(name, rng -> lowInclusive + rng.nextInt(highInclusive - lowInclusive + 1));
            return this;
        }

        public Builder nIter(int nIter) {
            if (nIter <= 0) throw new IllegalArgumentException("nIter must be positive");
            this.nIter = nIter;
            return this;
        }

        public Builder scoring(Estimators.Metric scoring) {
            this.scoring = Objects.requireNonNull(scoring, "scoring");
            return this;
        }

        public Builder cv(int cvFolds) {
            if (cvFolds < 2) throw new IllegalArgumentException("cvFolds must be >= 2");
            this.cvFolds = cvFolds;
            return this;
        }

        public Builder stratified(boolean stratified) {
            this.stratified = stratified;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public RandomizedSearchCV build() {
            if (distributions.isEmpty()) {
                throw new IllegalArgumentException("RandomizedSearchCV requires at least one parameter distribution");
            }
            return new RandomizedSearchCV(this);
        }
    }
}
