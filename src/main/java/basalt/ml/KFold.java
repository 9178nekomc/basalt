package basalt.ml;

import basalt.core.Matrix;
import java.util.List;

/**
 * K-Fold cross-validator.
 */
public final class KFold {
    private final int folds;
    private final boolean shuffle;
    private final long seed;

    public KFold(int folds) {
        this(folds, true, 42L);
    }

    public KFold(int folds, boolean shuffle, long seed) {
        if (folds < 2) {
            throw new IllegalArgumentException("folds must be >= 2");
        }
        this.folds = folds;
        this.shuffle = shuffle;
        this.seed = seed;
    }

    public int folds() {
        return folds;
    }

    public boolean shuffle() {
        return shuffle;
    }

    public List<DataSplit.Fold> split(Matrix features, double[] targets) {
        return DataSplit.kFold(features, targets, folds, shuffle, seed);
    }
}
