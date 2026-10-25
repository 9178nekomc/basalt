package jsl.core;

/** Stable wire identifiers for every persistable model family. */
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
    SEQUENCE_NETWORK(11);

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
