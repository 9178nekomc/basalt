package jsl.ml;

import jsl.core.Matrix;
import jsl.core.NativeKernels;
import java.util.Arrays;

/**
 * Quantile binning shared by the histogram boosting core and the random-forest trees.
 *
 * <p>Each feature is discretized into at most {@code maxBins} buckets. A bucket boundary is stored
 * as the midpoint of two adjacent order statistics and is interpreted as
 * {@code value <= threshold -> left}. Midpoints are computed as {@code a/2 + b/2} so that two huge
 * finite values cannot overflow to infinity, and the fallback keeps the boundary strictly inside
 * {@code [a, b)} even for adjacent IEEE-754 doubles.</p>
 */
final class BinnedFeatures {
    static final int MAX_BINS = 255;

    private final int rows;
    private final int features;
    private final byte[] bins;
    private final double[][] thresholds;

    private BinnedFeatures(int rows, int features, byte[] bins, double[][] thresholds) {
        this.rows = rows;
        this.features = features;
        this.bins = bins;
        this.thresholds = thresholds;
    }

    static BinnedFeatures build(Matrix features, int maxBins) {
        if (maxBins < 2 || maxBins > MAX_BINS) {
            throw new IllegalArgumentException("maxBins must be in [2, " + MAX_BINS + "]");
        }
        int rows = features.rows();
        int columns = features.columns();
        byte[] binned = new byte[rows * columns];
        if (NativeKernels.available() && rows >= 512 && columns >= 2) {
            double[] flat = new double[columns * (maxBins - 1)];
            int[] counts = new int[columns];
            if (NativeKernels.binFeaturesChecked(features.data(), rows, columns, maxBins,
                    binned, flat, counts)) {
                double[][] nativeThresholds = new double[columns][];
                int offset = 0;
                for (int feature = 0; feature < columns; feature++) {
                    nativeThresholds[feature] = Arrays.copyOfRange(flat, offset,
                            offset + counts[feature]);
                    offset += maxBins - 1;
                }
                return new BinnedFeatures(rows, columns, binned, nativeThresholds);
            }
        }
        double[][] thresholds = new double[columns][];
        double[] column = new double[rows];
        double[] data = features.data();
        for (int feature = 0; feature < columns; feature++) {
            for (int row = 0; row < rows; row++) {
                column[row] = data[row * columns + feature];
            }
            Arrays.sort(column);
            double[] candidates = new double[Math.min(maxBins - 1, rows - 1)];
            int count = 0;
            for (int bin = 1; bin <= maxBins - 1 && count < candidates.length; bin++) {
                int position = (int) ((long) rows * bin / maxBins) - 1;
                if (position < 0) {
                    position = 0;
                }
                if (position > rows - 2) {
                    position = rows - 2;
                }
                double left = column[position];
                double right = column[position + 1];
                if (left == right) {
                    continue;
                }
                double midpoint = left / 2.0 + right / 2.0;
                double threshold = midpoint >= left && midpoint < right ? midpoint : left;
                if (count == 0 || threshold != candidates[count - 1]) {
                    candidates[count++] = threshold;
                }
            }
            double[] featureThresholds = Arrays.copyOf(candidates, count);
            thresholds[feature] = featureThresholds;
            for (int row = 0; row < rows; row++) {
                int insertion = Arrays.binarySearch(
                        featureThresholds, data[row * columns + feature]);
                int bin = insertion >= 0 ? insertion : -insertion - 1;
                binned[row * columns + feature] = (byte) bin;
            }
        }
        return new BinnedFeatures(rows, columns, binned, thresholds);
    }

    int rows() {
        return rows;
    }

    int features() {
        return features;
    }

    byte[] bins() {
        return bins;
    }

    int bin(int row, int feature) {
        return Byte.toUnsignedInt(bins[row * features + feature]);
    }

    /** Number of populated buckets for a feature, i.e. {@code thresholds.length + 1}. */
    int binCount(int feature) {
        return thresholds[feature].length + 1;
    }

    /** Left/right boundary used by a split that places bucket {@code bin} on the left. */
    double threshold(int feature, int bin) {
        return thresholds[feature][bin];
    }

    double[][] thresholds() {
        return thresholds;
    }

    /** Largest number of buckets any feature actually uses. */
    int maxBinCount() {
        int maximum = 1;
        for (double[] featureThresholds : thresholds) {
            maximum = Math.max(maximum, featureThresholds.length + 1);
        }
        return maximum;
    }
}
