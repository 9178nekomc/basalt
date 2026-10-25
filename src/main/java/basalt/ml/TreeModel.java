package basalt.ml;

import basalt.core.ModelFormatException;
import basalt.core.ModelReader;
import basalt.core.ModelWriter;
import java.util.Arrays;

/**
 * Flat, array-backed regression/classification tree used for inference.
 *
 * <p>Nodes live in parallel arrays; a leaf is marked by {@code feature == -1}. Prediction is an
 * iterative pointer walk over primitive arrays, which removes the virtual dispatch and pointer
 * chasing of an object tree and keeps the hot path cache friendly. The same representation is
 * used by random forests and gradient boosting, so both share one predictor.</p>
 */
final class TreeModel {
    private int[] feature;
    private double[] threshold;
    private int[] left;
    private int[] right;
    private double[] value;
    private int size;

    TreeModel() {
        this(16);
    }

    TreeModel(int capacity) {
        int initial = Math.max(1, capacity);
        feature = new int[initial];
        threshold = new double[initial];
        left = new int[initial];
        right = new int[initial];
        value = new double[initial];
    }

    int addNode() {
        if (size == feature.length) {
            grow();
        }
        feature[size] = -1;
        threshold[size] = 0.0;
        left[size] = -1;
        right[size] = -1;
        value[size] = 0.0;
        return size++;
    }

    void setLeaf(int node, double leafValue) {
        feature[node] = -1;
        value[node] = leafValue;
        left[node] = -1;
        right[node] = -1;
    }

    void setSplit(int node, int splitFeature, double splitThreshold, int leftChild, int rightChild) {
        if (splitFeature < 0 || leftChild < 0 || rightChild < 0) {
            throw new IllegalArgumentException("Invalid split");
        }
        feature[node] = splitFeature;
        threshold[node] = splitThreshold;
        left[node] = leftChild;
        right[node] = rightChild;
    }

    int size() {
        return size;
    }

    int depth() {
        return depth(0);
    }

    private int depth(int node) {
        if (feature[node] < 0) {
            return 0;
        }
        return 1 + Math.max(depth(left[node]), depth(right[node]));
    }

    boolean isLeaf(int node) {
        return feature[node] < 0;
    }

    int feature(int node) {
        return feature[node];
    }

    int left(int node) {
        return left[node];
    }

    int right(int node) {
        return right[node];
    }

    TreeModel compact() {
        if (size == feature.length) {
            return this;
        }
        TreeModel result = new TreeModel(size);
        System.arraycopy(feature, 0, result.feature, 0, size);
        System.arraycopy(threshold, 0, result.threshold, 0, size);
        System.arraycopy(left, 0, result.left, 0, size);
        System.arraycopy(right, 0, result.right, 0, size);
        System.arraycopy(value, 0, result.value, 0, size);
        result.size = size;
        return result;
    }

    double predict(double[] sample, int offset) {
        int node = 0;
        while (true) {
            int splitFeature = feature[node];
            if (splitFeature < 0) {
                return value[node];
            }
            node = sample[offset + splitFeature] <= threshold[node] ? left[node] : right[node];
        }
    }

    /** Adds {@code scale * tree(x)} into {@code target} without allocating per-row predictions. */
    void addInto(double[] sample, int rows, int columns, double[] target, double scale) {
        for (int row = 0; row < rows; row++) {
            int offset = row * columns;
            int node = 0;
            while (true) {
                int splitFeature = feature[node];
                if (splitFeature < 0) {
                    target[row] += scale * value[node];
                    break;
                }
                node = sample[offset + splitFeature] <= threshold[node] ? left[node] : right[node];
            }
        }
    }

    /** Writes tree predictions into {@code target}, replacing previous values. */
    void predictInto(double[] sample, int rows, int columns, double[] target) {
        for (int row = 0; row < rows; row++) {
            int offset = row * columns;
            int node = 0;
            while (true) {
                int splitFeature = feature[node];
                if (splitFeature < 0) {
                    target[row] = value[node];
                    break;
                }
                node = sample[offset + splitFeature] <= threshold[node] ? left[node] : right[node];
            }
        }
    }

    void copyInto(int offset, int[] targetFeature, double[] targetThreshold,
                  int[] targetLeft, int[] targetRight, double[] targetValue) {
        System.arraycopy(feature, 0, targetFeature, offset, size);
        System.arraycopy(threshold, 0, targetThreshold, offset, size);
        System.arraycopy(value, 0, targetValue, offset, size);
        for (int node = 0; node < size; node++) {
            targetLeft[offset + node] = left[node] < 0 ? -1 : left[node] + offset;
            targetRight[offset + node] = right[node] < 0 ? -1 : right[node] + offset;
        }
    }

    void write(ModelWriter writer) {
        writer.putInt(size);
        writer.putInts(Arrays.copyOf(feature, size));
        writer.putDoubles(Arrays.copyOf(threshold, size));
        writer.putInts(Arrays.copyOf(left, size));
        writer.putInts(Arrays.copyOf(right, size));
        writer.putDoubles(Arrays.copyOf(value, size));
    }

    static TreeModel read(ModelReader reader) {
        int size = reader.getInt();
        if (size <= 0 || size > 10_000_000) {
            throw new ModelFormatException("Invalid tree node count " + size);
        }
        int[] feature = reader.getInts(size);
        double[] threshold = reader.getDoubles(size);
        int[] left = reader.getInts(size);
        int[] right = reader.getInts(size);
        double[] value = reader.getDoubles(size);
        for (int node = 0; node < size; node++) {
            if (feature[node] < -1) {
                throw new ModelFormatException("Invalid tree feature at node " + node);
            }
            if (feature[node] >= 0) {
                if (left[node] < 0 || left[node] >= size || right[node] < 0 || right[node] >= size) {
                    throw new ModelFormatException("Invalid tree child at node " + node);
                }
            }
        }
        TreeModel model = new TreeModel(size);
        model.feature = feature;
        model.threshold = threshold;
        model.left = left;
        model.right = right;
        model.value = value;
        model.size = size;
        return model;
    }

    private void grow() {
        int capacity = feature.length * 2;
        feature = Arrays.copyOf(feature, capacity);
        threshold = Arrays.copyOf(threshold, capacity);
        left = Arrays.copyOf(left, capacity);
        right = Arrays.copyOf(right, capacity);
        value = Arrays.copyOf(value, capacity);
    }
}
