package basalt.ml;

import basalt.core.Matrix;
import java.util.List;

/**
 * Stratified K-Fold cross-validator that preserves class frequencies in each fold,
 * matching {@code sklearn.model_selection.StratifiedKFold}.
 */
public final class StratifiedKFold {
    private final int folds;
    private final boolean shuffle;
    private final long seed;

    public StratifiedKFold(int folds) {
        this(folds, true, 42L);
    }

    public StratifiedKFold(int folds, boolean shuffle, long seed) {
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
        return DataSplit.stratifiedKFold(features, targets, folds, shuffle, seed);
    }
}
