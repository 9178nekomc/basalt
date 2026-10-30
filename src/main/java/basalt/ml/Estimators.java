package basalt.ml;

import basalt.core.Matrix;
import java.util.List;

/**
 * One-call facade over the estimator catalogue.
 */
public final class Estimators {
    private Estimators() {
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
        SVC(Task.CLASSIFICATION),
        KERNEL_RIDGE(Task.REGRESSION),
        KMEANS(Task.UNSUPERVISED),
        PCA(Task.UNSUPERVISED),
        KERNEL_PCA(Task.UNSUPERVISED),
        DBSCAN(Task.UNSUPERVISED),
        GAUSSIAN_MIXTURE(Task.UNSUPERVISED),
        AGGLOMERATIVE_CLUSTERING(Task.UNSUPERVISED),
        ISOLATION_FOREST(Task.UNSUPERVISED),
        KERNEL_DENSITY(Task.UNSUPERVISED);

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
        RMSE,
        /**
         * Mean cross entropy over the predicted class distribution.
         *
         * <p>Unlike the other four this is a loss, so it is smaller-is-better. {@link
         * basalt.auto.Auto} and the model layer negate it before storing a score, which keeps
         * "higher is better" true everywhere a score is compared.</p>
         */
        CROSS_ENTROPY
    }

    /**
     * Builds a fresh, unfitted estimator for {@code algorithm}.
     *
     * <p>{@link #fit} is {@code create(algorithm).fit(features, targets)}; exposing the
     * construction step separately lets callers configure a model before fitting it, and lets
     * {@link Registry} hand out new instances without duplicating the catalogue.</p>
     */
    public static Estimator create(Algorithm algorithm) {
        if (algorithm == null) {
            throw new IllegalArgumentException("algorithm must not be null");
        }
        return switch (algorithm) {
            case LINEAR_REGRESSION -> LinearRegression.builder().build();
            case LOGISTIC_REGRESSION -> LogisticRegression.builder().build();
            case RIDGE -> Ridge.builder().build();
            case LASSO -> Lasso.builder().build();
            case KNN -> KNearestNeighbors.builder().build();
            case KNN_REGRESSOR -> KNeighborsRegressor.builder().build();
            case DECISION_TREE -> DecisionTree.builder().build();
            case RANDOM_FOREST -> RandomForest.builder().build();
            case RANDOM_FOREST_REGRESSOR -> RandomForest.builder().task(TreeTask.REGRESSION).build();
            case GRADIENT_BOOSTING -> HistogramGradientBoosting.builder()
                    .objective(BoostingObjective.LOGISTIC).build();
            case GRADIENT_BOOSTING_REGRESSOR -> HistogramGradientBoosting.builder()
                    .objective(BoostingObjective.SQUARED_ERROR).build();
            case GAUSSIAN_NAIVE_BAYES -> GaussianNaiveBayes.builder().build();
            case LINEAR_SVM -> LinearSVM.builder().build();
            case SVC -> SVC.builder().build();
            case KERNEL_RIDGE -> KernelRidge.builder().build();
            case KMEANS -> KMeans.builder().build();
            case PCA -> PCA.builder().build();
            case KERNEL_PCA -> KernelPCA.builder().build();
            case DBSCAN -> DBSCAN.builder().build();
            case GAUSSIAN_MIXTURE -> GaussianMixture.builder().build();
            case AGGLOMERATIVE_CLUSTERING -> AgglomerativeClustering.builder().build();
            case ISOLATION_FOREST -> IsolationForest.builder().build();
            case KERNEL_DENSITY -> KernelDensity.builder().build();
        };
    }

    public static Estimator fit(Algorithm algorithm, Matrix features, double[] targets) {
        if (algorithm.task() != Task.UNSUPERVISED && targets == null) {
            throw new IllegalArgumentException(algorithm + " requires targets");
        }
        return create(algorithm).fit(features, targets);
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
                case CROSS_ENTROPY -> throw new IllegalArgumentException(
                        "cross entropy needs logits; classical estimators only produce labels");
            };
        }
        return total / splits.size();
    }
}
