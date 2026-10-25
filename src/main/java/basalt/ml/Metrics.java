package basalt.ml;

import java.util.Arrays;

/** Classification and regression metrics with no external dependencies. */
public final class Metrics {
    private Metrics() {
    }

    public static double accuracy(double[] expected, double[] predicted) {
        requireSameLength(expected, predicted);
        int correct = 0;
        for (int i = 0; i < expected.length; i++) {
            if (expected[i] == predicted[i]) {
                correct++;
            }
        }
        return (double) correct / expected.length;
    }

    public static double precision(double[] expected, double[] predicted, double positive) {
        requireSameLength(expected, predicted);
        int truePositive = 0;
        int falsePositive = 0;
        for (int i = 0; i < expected.length; i++) {
            if (predicted[i] == positive) {
                if (expected[i] == positive) {
                    truePositive++;
                } else {
                    falsePositive++;
                }
            }
        }
        int denominator = truePositive + falsePositive;
        return denominator == 0 ? 0.0 : (double) truePositive / denominator;
    }

    public static double recall(double[] expected, double[] predicted, double positive) {
        requireSameLength(expected, predicted);
        int truePositive = 0;
        int falseNegative = 0;
        for (int i = 0; i < expected.length; i++) {
            if (expected[i] == positive) {
                if (predicted[i] == positive) {
                    truePositive++;
                } else {
                    falseNegative++;
                }
            }
        }
        int denominator = truePositive + falseNegative;
        return denominator == 0 ? 0.0 : (double) truePositive / denominator;
    }

    public static double f1(double[] expected, double[] predicted, double positive) {
        double precision = precision(expected, predicted, positive);
        double recall = recall(expected, predicted, positive);
        return precision + recall == 0.0 ? 0.0 : 2.0 * precision * recall / (precision + recall);
    }

    public static double macroF1(double[] expected, double[] predicted) {
        double[] classes = unique(expected);
        double sum = 0.0;
        for (double label : classes) {
            sum += f1(expected, predicted, label);
        }
        return sum / classes.length;
    }

    public static int[][] confusionMatrix(double[] expected, double[] predicted) {
        requireSameLength(expected, predicted);
        double[] classes = unique(expected);
        int[][] matrix = new int[classes.length][classes.length];
        for (int i = 0; i < expected.length; i++) {
            int row = Arrays.binarySearch(classes, expected[i]);
            int column = Arrays.binarySearch(classes, predicted[i]);
            if (column < 0) {
                throw new IllegalArgumentException("Prediction contains an unknown class");
            }
            matrix[row][column]++;
        }
        return matrix;
    }

    /** Area under the ROC curve for binary labels 0/1 and real-valued scores. */
    public static double rocAuc(double[] labels, double[] scores) {
        requireSameLength(labels, scores);
        int positives = 0;
        for (double label : labels) {
            if (label != 0.0 && label != 1.0) {
                throw new IllegalArgumentException("ROC labels must be 0 or 1");
            }
            positives += label == 1.0 ? 1 : 0;
        }
        int negatives = labels.length - positives;
        if (positives == 0 || negatives == 0) {
            throw new IllegalArgumentException("ROC requires both classes");
        }
        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (left, right) -> Double.compare(scores[left], scores[right]));
        double[] ranks = new double[scores.length];
        int index = 0;
        while (index < order.length) {
            int end = index;
            while (end + 1 < order.length
                    && Double.compare(scores[order[end + 1]], scores[order[index]]) == 0) {
                end++;
            }
            double averageRank = (index + end + 2.0) / 2.0;
            for (int position = index; position <= end; position++) {
                ranks[order[position]] = averageRank;
            }
            index = end + 1;
        }
        double positiveRankSum = 0.0;
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] == 1.0) {
                positiveRankSum += ranks[i];
            }
        }
        return (positiveRankSum - (double) positives * (positives + 1) / 2.0)
                / ((double) positives * negatives);
    }

    public static double mse(double[] expected, double[] predicted) {
        requireSameLength(expected, predicted);
        double sum = 0.0;
        for (int i = 0; i < expected.length; i++) {
            double error = expected[i] - predicted[i];
            sum += error * error;
        }
        return sum / expected.length;
    }

    public static double rmse(double[] expected, double[] predicted) {
        return Math.sqrt(mse(expected, predicted));
    }

    public static double mae(double[] expected, double[] predicted) {
        requireSameLength(expected, predicted);
        double sum = 0.0;
        for (int i = 0; i < expected.length; i++) {
            sum += Math.abs(expected[i] - predicted[i]);
        }
        return sum / expected.length;
    }

    public static double r2(double[] expected, double[] predicted) {
        requireSameLength(expected, predicted);
        double mean = 0.0;
        for (double value : expected) {
            mean += value;
        }
        mean /= expected.length;
        double residual = 0.0;
        double total = 0.0;
        for (int i = 0; i < expected.length; i++) {
            residual += (expected[i] - predicted[i]) * (expected[i] - predicted[i]);
            total += (expected[i] - mean) * (expected[i] - mean);
        }
        return total == 0.0 ? (residual == 0.0 ? 1.0 : 0.0) : 1.0 - residual / total;
    }

    static double[] unique(double[] values) {
        double[] copy = values.clone();
        Arrays.sort(copy);
        int count = 0;
        for (int i = 0; i < copy.length; i++) {
            if (i == 0 || copy[i] != copy[i - 1]) {
                copy[count++] = copy[i];
            }
        }
        return Arrays.copyOf(copy, count);
    }

    private static void requireSameLength(double[] expected, double[] predicted) {
        if (expected.length != predicted.length) {
            throw new IllegalArgumentException("Expected and predicted lengths differ");
        }
        if (expected.length == 0) {
            throw new IllegalArgumentException("Metrics require at least one sample");
        }
    }
}
