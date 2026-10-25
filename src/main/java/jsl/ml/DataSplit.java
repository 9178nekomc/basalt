package jsl.ml;

import jsl.core.Matrix;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Deterministic holdout ({@code trainTestSplit}), k-fold, and stratified k-fold splits over rows.
 */
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
        return trainTestSplit(features, targets, testFraction, true, false, seed);
    }

    public static Holdout trainTestSplit(Matrix features, double[] targets, double testFraction,
                                         long seed) {
        return trainTestSplit(features, targets, testFraction, true, false, seed);
    }

    public static Holdout trainTestSplit(Matrix features, double[] targets, double testFraction,
                                         boolean shuffle, boolean stratify, long seed) {
        MlChecks.trainingData(features, targets);
        if (!(testFraction > 0.0) || testFraction >= 1.0) {
            throw new IllegalArgumentException("testFraction must be in (0, 1)");
        }
        int rows = features.rows();
        if (rows < 2) {
            throw new IllegalArgumentException("trainTestSplit requires at least 2 samples");
        }
        if (!stratify) {
            int[] order = shuffle ? shuffledRows(rows, seed) : identityRows(rows);
            int testSize = Math.max(1, Math.min(rows - 1, (int) Math.round(rows * testFraction)));
            return new Holdout(
                    select(features, order, testSize, rows),
                    selectTargets(targets, order, testSize, rows),
                    select(features, order, 0, testSize),
                    selectTargets(targets, order, 0, testSize));
        }

        Map<Double, List<Integer>> byClass = new LinkedHashMap<>();
        for (int r = 0; r < rows; r++) {
            byClass.computeIfAbsent(targets[r], k -> new ArrayList<>()).add(r);
        }
        Random random = new Random(seed);
        List<Integer> trainIdx = new ArrayList<>();
        List<Integer> testIdx = new ArrayList<>();
        for (List<Integer> indices : byClass.values()) {
            if (shuffle) {
                for (int i = indices.size() - 1; i > 0; i--) {
                    int j = random.nextInt(i + 1);
                    int tmp = indices.get(i);
                    indices.set(i, indices.get(j));
                    indices.set(j, tmp);
                }
            }
            int clsTest = Math.max(1, Math.min(indices.size() - 1,
                    (int) Math.round(indices.size() * testFraction)));
            if (indices.size() == 1) {
                trainIdx.add(indices.get(0));
            } else {
                for (int i = 0; i < clsTest; i++) {
                    testIdx.add(indices.get(i));
                }
                for (int i = clsTest; i < indices.size(); i++) {
                    trainIdx.add(indices.get(i));
                }
            }
        }
        int[] trainOrder = trainIdx.stream().mapToInt(Integer::intValue).toArray();
        int[] testOrder = testIdx.stream().mapToInt(Integer::intValue).toArray();
        return new Holdout(
                select(features, trainOrder, 0, trainOrder.length),
                selectTargets(targets, trainOrder, 0, trainOrder.length),
                select(features, testOrder, 0, testOrder.length),
                selectTargets(targets, testOrder, 0, testOrder.length));
    }

    public static List<Fold> kFold(Matrix features, double[] targets, int folds, long seed) {
        return kFold(features, targets, folds, true, seed);
    }

    public static List<Fold> kFold(Matrix features, double[] targets, int folds, boolean shuffle,
                                   long seed) {
        MlChecks.trainingData(features, targets);
        if (folds < 2 || folds > features.rows()) {
            throw new IllegalArgumentException("folds must be in [2, rows]");
        }
        int[] order = shuffle ? shuffledRows(features.rows(), seed) : identityRows(features.rows());
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

    public static List<Fold> stratifiedKFold(Matrix features, double[] targets, int folds,
                                             boolean shuffle, long seed) {
        MlChecks.trainingData(features, targets);
        int rows = features.rows();
        if (folds < 2 || folds > rows) {
            throw new IllegalArgumentException("folds must be in [2, rows]");
        }
        Map<Double, List<Integer>> byClass = new LinkedHashMap<>();
        for (int r = 0; r < rows; r++) {
            byClass.computeIfAbsent(targets[r], k -> new ArrayList<>()).add(r);
        }
        Random random = new Random(seed);
        List<List<Integer>> foldBuckets = new ArrayList<>(folds);
        for (int f = 0; f < folds; f++) {
            foldBuckets.add(new ArrayList<>());
        }
        int rr = 0;
        for (List<Integer> indices : byClass.values()) {
            if (indices.size() < folds) {
                throw new IllegalArgumentException(
                        "Class has fewer samples (" + indices.size() + ") than folds (" + folds + ")");
            }
            if (shuffle) {
                for (int i = indices.size() - 1; i > 0; i--) {
                    int j = random.nextInt(i + 1);
                    int tmp = indices.get(i);
                    indices.set(i, indices.get(j));
                    indices.set(j, tmp);
                }
            }
            for (int idx : indices) {
                foldBuckets.get(rr % folds).add(idx);
                rr++;
            }
        }
        List<Fold> result = new ArrayList<>(folds);
        for (int f = 0; f < folds; f++) {
            List<Integer> testList = foldBuckets.get(f);
            List<Integer> trainList = new ArrayList<>(rows - testList.size());
            for (int other = 0; other < folds; other++) {
                if (other != f) {
                    trainList.addAll(foldBuckets.get(other));
                }
            }
            int[] trainOrder = trainList.stream().mapToInt(Integer::intValue).toArray();
            int[] testOrder = testList.stream().mapToInt(Integer::intValue).toArray();
            result.add(new Fold(
                    select(features, trainOrder, 0, trainOrder.length),
                    selectTargets(targets, trainOrder, 0, trainOrder.length),
                    select(features, testOrder, 0, testOrder.length),
                    selectTargets(targets, testOrder, 0, testOrder.length)));
        }
        return result;
    }

    private static int[] identityRows(int rows) {
        int[] order = new int[rows];
        for (int i = 0; i < rows; i++) order[i] = i;
        return order;
    }

    private static int[] shuffledRows(int rows, long seed) {
        int[] order = identityRows(rows);
        Random random = new Random(seed);
        for (int i = rows - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temporary = order[i];
            order[i] = order[j];
            order[j] = temporary;
        }
        return order;
    }

    static Matrix select(Matrix features, int[] order, int from, int to) {
        int columns = features.columns();
        double[] values = new double[(to - from) * columns];
        for (int row = from; row < to; row++) {
            System.arraycopy(features.data(), order[row] * columns,
                    values, (row - from) * columns, columns);
        }
        return new Matrix(to - from, columns, values);
    }

    static double[] selectTargets(double[] targets, int[] order, int from, int to) {
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
