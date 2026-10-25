package jsl.ml;

import jsl.core.Matrix;
import java.util.List;

/**
 * One-call facade over the estimator catalogue.
 *
 * <pre>{@code
 * Estimator model = JSL.fit(JSL.Algorithm.RANDOM_FOREST, x, y);
 * double[] predictions = JSL.predict(model, testX);
 * double accuracy = JSL.crossValidate(JSL.Algorithm.RIDGE, x, y, 5, JSL.Metric.R2);
 * }</pre>
 *
 * <p>Defaults are intentionally conservative; use the concrete builders when tuning. The facade
 * exists so a model can be selected by name/enum without a dependency-injection framework.</p>
 */
public final class JSL {
    private JSL() {
    }

    public enum Task {
        CLASSIFICATION,
        REGRESSION,
        UNSUPERVISED
    }

    public enum Algorithm {
        LINEAR_REGRESSION(Task.REGRESSION),
        LOGISTIC_REGRESSION(Task.CLASSIFICATION),
        RIDGE(Task.REGRESSION),
        LASSO(Task.REGRESSION),
        KNN(Task.CLASSIFICATION),
        KNN_REGRESSOR(Task.REGRESSION),
        DECISION_TREE(Task.CLASSIFICATION),
        RANDOM_FOREST(Task.CLASSIFICATION),
        RANDOM_FOREST_REGRESSOR(Task.REGRESSION),
        GRADIENT_BOOSTING(Task.CLASSIFICATION),
        GRADIENT_BOOSTING_REGRESSOR(Task.REGRESSION),
        GAUSSIAN_NAIVE_BAYES(Task.CLASSIFICATION),
        LINEAR_SVM(Task.CLASSIFICATION),
        KMEANS(Task.UNSUPERVISED),
        PCA(Task.UNSUPERVISED),
        DBSCAN(Task.UNSUPERVISED);

        private final Task task;

        Algorithm(Task task) {
            this.task = task;
        }

        public Task task() {
            return task;
        }
    }

    public enum Metric {
        ACCURACY,
        MACRO_F1,
        R2,
        RMSE
    }

    public static Estimator fit(Algorithm algorithm, Matrix features, double[] targets) {
        if (algorithm.task() != Task.UNSUPERVISED && targets == null) {
            throw new IllegalArgumentException(algorithm + " requires targets");
        }
        return switch (algorithm) {
            case LINEAR_REGRESSION -> LinearRegression.builder().build().fit(features, targets);
            case LOGISTIC_REGRESSION -> LogisticRegression.builder().build().fit(features, targets);
            case RIDGE -> Ridge.builder().build().fit(features, targets);
            case LASSO -> Lasso.builder().build().fit(features, targets);
            case KNN -> KNearestNeighbors.builder().build().fit(features, targets);
            case KNN_REGRESSOR -> KNeighborsRegressor.builder().build().fit(features, targets);
            case DECISION_TREE -> DecisionTree.builder().build().fit(features, targets);
            case RANDOM_FOREST -> RandomForest.builder().build().fit(features, targets);
            case RANDOM_FOREST_REGRESSOR -> RandomForest.builder().task(TreeTask.REGRESSION)
                    .build().fit(features, targets);
            case GRADIENT_BOOSTING -> HistogramGradientBoosting.builder()
                    .objective(BoostingObjective.LOGISTIC).build().fit(features, targets);
            case GRADIENT_BOOSTING_REGRESSOR -> HistogramGradientBoosting.builder()
                    .objective(BoostingObjective.SQUARED_ERROR).build().fit(features, targets);
            case GAUSSIAN_NAIVE_BAYES -> GaussianNaiveBayes.builder().build().fit(features, targets);
            case LINEAR_SVM -> LinearSVM.builder().build().fit(features, targets);
            case KMEANS -> KMeans.builder().build().fit(features);
            case PCA -> PCA.builder().build().fit(features);
            case DBSCAN -> DBSCAN.builder().build().fit(features, null);
        };
    }

    public static double[] predict(Estimator model, Matrix features) {
        return model.predict(features);
    }

    /** Mean cross-validated score. Unsupervised algorithms are rejected. */
    public static double crossValidate(Algorithm algorithm, Matrix features, double[] targets,
                                       int folds, Metric metric) {
        if (algorithm.task() == Task.UNSUPERVISED) {
            throw new IllegalArgumentException("Cross-validation requires a supervised algorithm");
        }
        List<DataSplit.Fold> splits = DataSplit.kFold(features, targets, folds, 42);
        double total = 0.0;
        for (DataSplit.Fold fold : splits) {
            Estimator model = fit(algorithm, fold.trainFeatures(), fold.trainTargets());
            double[] prediction = model.predict(fold.testFeatures());
            total += switch (metric) {
                case ACCURACY -> Metrics.accuracy(fold.testTargets(), prediction);
                case MACRO_F1 -> Metrics.macroF1(fold.testTargets(), prediction);
                case R2 -> Metrics.r2(fold.testTargets(), prediction);
                case RMSE -> -Metrics.rmse(fold.testTargets(), prediction);
            };
        }
        return total / splits.size();
    }
}
