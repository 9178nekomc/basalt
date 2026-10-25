package basalt.ml;

/** Node impurity used to maximize information gain in decision trees. */
public enum ImpurityCriterion {
    GINI(TreeTask.CLASSIFICATION) {
        @Override
        double impurity(int[] counts, int total) {
            double sumSquares = 0.0;
            for (int count : counts) {
                double probability = (double) count / total;
                sumSquares += probability * probability;
            }
            return 1.0 - sumSquares;
        }
    },
    ENTROPY(TreeTask.CLASSIFICATION) {
        @Override
        double impurity(int[] counts, int total) {
            double entropy = 0.0;
            for (int count : counts) {
                if (count > 0) {
                    double probability = (double) count / total;
                    entropy -= probability * (Math.log(probability) / Math.log(2.0));
                }
            }
            return entropy;
        }
    },
    /** Variance impurity; the split search uses the algebraically equivalent sum/sum-of-squares form. */
    MSE(TreeTask.REGRESSION) {
        @Override
        double impurity(int[] counts, int total) {
            throw new IllegalStateException("MSE impurity operates on continuous targets");
        }
    };

    private final TreeTask task;

    ImpurityCriterion(TreeTask task) {
        this.task = task;
    }

    public TreeTask task() {
        return task;
    }

    abstract double impurity(int[] counts, int total);
}
