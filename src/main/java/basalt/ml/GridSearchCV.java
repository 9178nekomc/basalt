package basalt.ml;

import basalt.core.Matrix;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/**
 * Exhaustive search over specified hyperparameter values for an estimator using K-Fold or
 * Stratified K-Fold cross-validation, matching {@code sklearn.model_selection.GridSearchCV}.
 */
public final class GridSearchCV implements Estimator {
    public record CandidateResult(Map<String, Object> params, double meanScore, double stdScore,
                                  double[] foldScores) {
    }

    private final Function<Map<String, Object>, Estimator> estimatorFactory;
    private final Map<String, List<Object>> paramGrid;
    private final Estimators.Metric scoring;
    private final int cvFolds;
    private final boolean stratified;
    private final long seed;

    private Estimator bestEstimator;
    private Map<String, Object> bestParams;
    private double bestScore = Double.NEGATIVE_INFINITY;
    private final List<CandidateResult> cvResults = new ArrayList<>();

    private GridSearchCV(Builder builder) {
        this.estimatorFactory = builder.estimatorFactory;
        this.paramGrid = new LinkedHashMap<>(builder.paramGrid);
        this.scoring = builder.scoring;
        this.cvFolds = builder.cvFolds;
        this.stratified = builder.stratified;
        this.seed = builder.seed;
    }

    public static Builder builder(Function<Map<String, Object>, Estimator> estimatorFactory) {
        return new Builder(estimatorFactory);
    }

    @Override
    public GridSearchCV fit(Matrix features, double[] targets) {
        MlChecks.trainingData(features, targets);
        List<DataSplit.Fold> folds = stratified
                ? DataSplit.stratifiedKFold(features, targets, cvFolds, true, seed)
                : DataSplit.kFold(features, targets, cvFolds, true, seed);

        List<Map<String, Object>> combinations = cartesianProduct(paramGrid);
        cvResults.clear();
        bestScore = Double.NEGATIVE_INFINITY;
        bestParams = null;

        for (Map<String, Object> params : combinations) {
            double[] scores = new double[folds.size()];
            double sum = 0.0;
            for (int f = 0; f < folds.size(); f++) {
                DataSplit.Fold fold = folds.get(f);
                Estimator model = estimatorFactory.apply(params);
                model.fit(fold.trainFeatures(), fold.trainTargets());
                double[] preds = model.predict(fold.testFeatures());
                scores[f] = evaluateScore(scoring, fold.testTargets(), preds);
                sum += scores[f];
            }
            double mean = sum / folds.size();
            double var = 0.0;
            for (double s : scores) {
                double d = s - mean;
                var += d * d;
            }
            double std = Math.sqrt(var / folds.size());
            Map<String, Object> immutableParams = Collections.unmodifiableMap(new LinkedHashMap<>(params));
            cvResults.add(new CandidateResult(immutableParams, mean, std, scores));
            if (mean > bestScore || bestParams == null) {
                bestScore = mean;
                bestParams = immutableParams;
            }
        }
        bestEstimator = estimatorFactory.apply(bestParams);
        bestEstimator.fit(features, targets);
        return this;
    }

    static double evaluateScore(Estimators.Metric metric, double[] expected, double[] predicted) {
        return switch (metric) {
            case ACCURACY -> Metrics.accuracy(expected, predicted);
            case MACRO_F1 -> Metrics.macroF1(expected, predicted);
            case R2 -> Metrics.r2(expected, predicted);
            case RMSE -> -Metrics.rmse(expected, predicted);
        };
    }

    private static List<Map<String, Object>> cartesianProduct(Map<String, List<Object>> grid) {
        List<Map<String, Object>> result = new ArrayList<>();
        result.add(new LinkedHashMap<>());
        for (Map.Entry<String, List<Object>> entry : grid.entrySet()) {
            String key = entry.getKey();
            List<Object> values = entry.getValue();
            List<Map<String, Object>> next = new ArrayList<>(result.size() * values.size());
            for (Map<String, Object> current : result) {
                for (Object val : values) {
                    Map<String, Object> copy = new LinkedHashMap<>(current);
                    copy.put(key, val);
                    next.add(copy);
                }
            }
            result = next;
        }
        return result;
    }

    @Override
    public double[] predict(Matrix features) {
        if (bestEstimator == null) {
            throw new IllegalStateException("GridSearchCV must be fitted first");
        }
        return bestEstimator.predict(features);
    }

    public Estimator bestEstimator() {
        if (bestEstimator == null) throw new IllegalStateException("GridSearchCV is not fitted");
        return bestEstimator;
    }

    public Map<String, Object> bestParams() {
        if (bestParams == null) throw new IllegalStateException("GridSearchCV is not fitted");
        return bestParams;
    }

    public double bestScore() {
        if (bestParams == null) throw new IllegalStateException("GridSearchCV is not fitted");
        return bestScore;
    }

    public List<CandidateResult> cvResults() {
        return Collections.unmodifiableList(cvResults);
    }

    public static final class Builder {
        private final Function<Map<String, Object>, Estimator> estimatorFactory;
        private final Map<String, List<Object>> paramGrid = new LinkedHashMap<>();
        private Estimators.Metric scoring = Estimators.Metric.ACCURACY;
        private int cvFolds = 5;
        private boolean stratified = false;
        private long seed = 42L;

        private Builder(Function<Map<String, Object>, Estimator> estimatorFactory) {
            this.estimatorFactory = Objects.requireNonNull(estimatorFactory, "estimatorFactory");
        }

        public Builder param(String name, Object... values) {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(values, "values");
            if (values.length == 0) {
                throw new IllegalArgumentException("Parameter values for " + name + " must not be empty");
            }
            List<Object> list = new ArrayList<>(values.length);
            for (Object v : values) {
                list.add(v);
            }
            paramGrid.put(name, list);
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

        public GridSearchCV build() {
            if (paramGrid.isEmpty()) {
                throw new IllegalArgumentException("GridSearchCV requires at least one parameter in paramGrid");
            }
            return new GridSearchCV(this);
        }
    }
}
