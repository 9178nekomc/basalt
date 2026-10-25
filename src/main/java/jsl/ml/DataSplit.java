package jsl.ml;

import jsl.core.Matrix;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Deterministic holdout and k-fold splits over rows. */
public final class DataSplit {
    private DataSplit() {
    }

    public record Holdout(Matrix trainFeatures, double[] trainTargets,
                          Matrix testFeatures, double[] testTargets) {
    }

    public record Fold(Matrix trainFeatures, double[] trainTargets,
                       Matrix testFeatures, double[] testTargets) {
    }

    public static Holdout holdout(Matrix features, double[] targets, double testFraction, long seed) {
        MlChecks.trainingData(features, targets);
        if (!(testFraction > 0.0) || testFraction >= 1.0) {
            throw new IllegalArgumentException("testFraction must be in (0, 1)");
        }
        int[] order = shuffledRows(features.rows(), seed);
        int testSize = Math.max(1, Math.min(features.rows() - 1,
                (int) Math.round(features.rows() * testFraction)));
        return new Holdout(
                select(features, order, testSize, features.rows()),
                selectTargets(targets, order, testSize, targets.length),
                select(features, order, 0, testSize),
                selectTargets(targets, order, 0, testSize));
    }

    public static List<Fold> kFold(Matrix features, double[] targets, int folds, long seed) {
        MlChecks.trainingData(features, targets);
        if (folds < 2 || folds > features.rows()) {
            throw new IllegalArgumentException("folds must be in [2, rows]");
        }
        int[] order = shuffledRows(features.rows(), seed);
        List<Fold> result = new ArrayList<>(folds);
        for (int fold = 0; fold < folds; fold++) {
            int testFrom = fold * features.rows() / folds;
            int testTo = (fold + 1) * features.rows() / folds;
            Matrix train = concatRows(features, order, 0, testFrom, testTo, features.rows());
            double[] trainTargets = concatTargets(targets, order, 0, testFrom, testTo, targets.length);
            Matrix test = select(features, order, testFrom, testTo);
            double[] testTargets = selectTargets(targets, order, testFrom, testTo);
            result.add(new Fold(train, trainTargets, test, testTargets));
        }
        return result;
    }

    private static int[] shuffledRows(int rows, long seed) {
        int[] order = new int[rows];
        for (int i = 0; i < rows; i++) {
            order[i] = i;
        }
        Random random = new Random(seed);
        for (int i = rows - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temporary = order[i];
            order[i] = order[j];
            order[j] = temporary;
        }
        return order;
    }

    private static Matrix select(Matrix features, int[] order, int from, int to) {
        int columns = features.columns();
        double[] values = new double[(to - from) * columns];
        for (int row = from; row < to; row++) {
            System.arraycopy(features.data(), order[row] * columns,
                    values, (row - from) * columns, columns);
        }
        return new Matrix(to - from, columns, values);
    }

    private static double[] selectTargets(double[] targets, int[] order, int from, int to) {
        double[] values = new double[to - from];
        for (int row = from; row < to; row++) {
            values[row - from] = targets[order[row]];
        }
        return values;
    }

    private static Matrix concatRows(Matrix features, int[] order, int firstFrom, int firstTo,
                                     int secondFrom, int secondTo) {
        int columns = features.columns();
        int rows = (firstTo - firstFrom) + (secondTo - secondFrom);
        double[] values = new double[rows * columns];
        int position = 0;
        for (int row = firstFrom; row < firstTo; row++) {
            System.arraycopy(features.data(), order[row] * columns, values, position * columns, columns);
            position++;
        }
        for (int row = secondFrom; row < secondTo; row++) {
            System.arraycopy(features.data(), order[row] * columns, values, position * columns, columns);
            position++;
        }
        return new Matrix(rows, columns, values);
    }

    private static double[] concatTargets(double[] targets, int[] order, int firstFrom, int firstTo,
                                          int secondFrom, int secondTo) {
        double[] values = new double[(firstTo - firstFrom) + (secondTo - secondFrom)];
        int position = 0;
        for (int row = firstFrom; row < firstTo; row++) {
            values[position++] = targets[order[row]];
        }
        for (int row = secondFrom; row < secondTo; row++) {
            values[position++] = targets[order[row]];
        }
        return values;
    }
}
