package jsl.core;

/** Stable wire identifiers for every persistable model and transformer family. */
public enum ModelKind {
    CNN(1),
    HISTOGRAM_BOOSTING(2),
    RANDOM_FOREST(3),
    KMEANS(4),
    PCA(5),
    GAUSSIAN_NAIVE_BAYES(6),
    LSTM(7),
    GRU(8),
    VANILLA_RNN(9),
    SELF_ATTENTION(10),
    SEQUENCE_NETWORK(11),
    STANDARD_SCALER(12),
    MIN_MAX_SCALER(13),
    ONE_HOT_ENCODER(14),
    LABEL_ENCODER(15),
    SIMPLE_IMPUTER(16),
    PIPELINE(17),
    SVC(18),
    KERNEL_RIDGE(19),
    KERNEL_PCA(20),
    GAUSSIAN_MIXTURE(21),
    ISOLATION_FOREST(22),
    KERNEL_DENSITY(23),
    SELECT_K_BEST(24),
    POLYNOMIAL_FEATURES(25),
    KBINS_DISCRETIZER(26),
    LINEAR_REGRESSION(27),
    LOGISTIC_REGRESSION(28),
    RIDGE(29),
    LASSO(30),
    LINEAR_SVM(31),
    DECISION_TREE(32);

    private final int code;

    ModelKind(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    public static ModelKind fromCode(int code) {
        for (ModelKind kind : values()) {
            if (kind.code == code) {
                return kind;
            }
        }
        throw new ModelFormatException("Unknown JSL model kind code " + code);
    }
}
