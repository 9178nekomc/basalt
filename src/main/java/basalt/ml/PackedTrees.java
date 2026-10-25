package basalt.ml;

import java.util.List;

/** Concatenated tree arrays for the native ensemble predictor. */
final class PackedTrees {
    final int[] feature;
    final double[] threshold;
    final int[] left;
    final int[] right;
    final double[] value;
    final int[] offsets;

    private PackedTrees(int[] feature, double[] threshold, int[] left, int[] right,
                        double[] value, int[] offsets) {
        this.feature = feature;
        this.threshold = threshold;
        this.left = left;
        this.right = right;
        this.value = value;
        this.offsets = offsets;
    }

    static PackedTrees pack(List<TreeModel> trees) {
        int total = 0;
        for (TreeModel tree : trees) {
            total += tree.size();
        }
        int[] feature = new int[total];
        double[] threshold = new double[total];
        int[] left = new int[total];
        int[] right = new int[total];
        double[] value = new double[total];
        int[] offsets = new int[trees.size() + 1];
        int offset = 0;
        for (int index = 0; index < trees.size(); index++) {
            offsets[index] = offset;
            TreeModel tree = trees.get(index);
            tree.copyInto(offset, feature, threshold, left, right, value);
            offset += tree.size();
        }
        offsets[trees.size()] = offset;
        return new PackedTrees(feature, threshold, left, right, value, offsets);
    }
}
