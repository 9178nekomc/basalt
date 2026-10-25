package basalt.ml;

import basalt.core.Matrix;
import basalt.core.ModelFormatException;
import basalt.core.ModelKind;
import basalt.core.ModelReader;
import basalt.core.ModelWriter;
import basalt.core.NativeKernels;
import basalt.core.Persistable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.TreeMap;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * Bootstrap-aggregated histogram trees with per-node feature subsampling.
 *
 * <p>Trees are independent, so fitting scales linearly with cores. Each tree pre-bins the sampled
 * matrix once and then grows with histogram splits, which is substantially faster than sorting
 * every feature at every node. Both classification and regression are supported.</p>
 */
public final class RandomForest implements Estimator, Persistable {
    private final int treeCount;
    private final int maxDepth;
    private final int minSamplesLeaf;
    private final int maxFeatures;
    private final int maxBins;
    private final int threads;
    private final long seed;
    private final TreeTask task;
    private final boolean oob;
    private DecisionTree[] trees;
    private PackedTrees packed;
    private double[] classes;
    private int featureCount;
    private double oobScore = Double.NaN;

    private RandomForest(Builder builder) {
        treeCount = builder.treeCount;
        maxDepth = builder.maxDepth;
        minSamplesLeaf = builder.minSamplesLeaf;
        maxFeatures = builder.maxFeatures;
        maxBins = builder.maxBins;
        threads = builder.threads;
        seed = builder.seed;
        task = builder.task;
        oob = builder.oob;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public RandomForest fit(Matrix features, double[] targets) {
        MlChecks.trainingData(features, targets);
        featureCount = features.columns();
        int rows = features.rows();
        if (task == TreeTask.CLASSIFICATION) {
            TreeMap<Double, Integer> classIndex = new TreeMap<>();
            for (double target : targets) {
                classIndex.putIfAbsent(target, 0);
            }
            classes = new double[classIndex.size()];
            int position = 0;
            for (double label : classIndex.keySet()) {
                classes[position++] = label;
            }
            for (int index = 0; index < classes.length; index++) {
                classIndex.put(classes[index], index);
            }
        } else {
            classes = null;
        }
        if (oob && (long) treeCount * rows > 8_000_000L) {
            throw new IllegalArgumentException(
                    "OOB evaluation is limited to 8M tree-row predictions; disable oob()");
        }
        int resolvedFeatures = maxFeatures == 0
                ? Math.max(1, (int) Math.sqrt(featureCount)) : maxFeatures;
        trees = new DecisionTree[treeCount];
        BinnedFeatures binned = BinnedFeatures.build(features, maxBins);
        double[][] oobPredictions = oob && task == TreeTask.REGRESSION ? new double[treeCount][] : null;
        int[][] oobClasses = oob && task == TreeTask.CLASSIFICATION ? new int[treeCount][] : null;
        boolean[][] inBag = oob ? new boolean[treeCount][] : null;
        ForkJoinPool pool = new ForkJoinPool(threads);
        try {
            pool.submit(() -> IntStream.range(0, treeCount).parallel().forEach(treeIndex -> {
                Random random = new Random(seed + treeIndex * 1_000_003L);
                int[] bootstrap = new int[rows];
                boolean[] selectedFlags = oob ? new boolean[rows] : null;
                for (int row = 0; row < rows; row++) {
                    int selected = random.nextInt(rows);
                    bootstrap[row] = selected;
                    if (selectedFlags != null) {
                        selectedFlags[selected] = true;
                    }
                }
                DecisionTree tree = DecisionTree.builder()
                        .task(task)
                        .maxDepth(maxDepth)
                        .minSamplesLeaf(minSamplesLeaf)
                        .maxFeatures(resolvedFeatures)
                        .bins(maxBins)
                        .seed(seed + treeIndex * 31L)
                        .build()
                        .fitBinned(binned, bootstrap, targets);
                trees[treeIndex] = tree;
                if (oob) {
                    double[] prediction = tree.predict(features);
                    if (task == TreeTask.REGRESSION) {
                        oobPredictions[treeIndex] = prediction;
                    } else {
                        int[] encoded = new int[rows];
                        for (int row = 0; row < rows; row++) {
                            int index = Arrays.binarySearch(classes, prediction[row]);
                            if (index < 0) {
                                throw new IllegalStateException("Unknown tree class");
                            }
                            encoded[row] = index;
                        }
                        oobClasses[treeIndex] = encoded;
                    }
                    inBag[treeIndex] = selectedFlags;
                }
            })).join();
        } finally {
            pool.shutdown();
        }
        packed = packTrees();
        if (oob) {
            oobScore = task == TreeTask.REGRESSION
                    ? computeOobRegression(oobPredictions, inBag, targets)
                    : computeOobAccuracy(oobClasses, inBag, targets);
        }
        return this;
    }

    private double computeOobRegression(double[][] predictions, boolean[][] inBag, double[] targets) {
        int rows = targets.length;
        double[] sums = new double[rows];
        int[] counts = new int[rows];
        for (int treeIndex = 0; treeIndex < treeCount; treeIndex++) {
            double[] prediction = predictions[treeIndex];
            boolean[] selected = inBag[treeIndex];
            for (int row = 0; row < rows; row++) {
                if (!selected[row]) {
                    sums[row] += prediction[row];
                    counts[row]++;
                }
            }
        }
        double mean = 0.0;
        for (double target : targets) {
            mean += target;
        }
        mean /= rows;
        double residual = 0.0;
        double total = 0.0;
        int evaluated = 0;
        for (int row = 0; row < rows; row++) {
            if (counts[row] == 0) {
                continue;
            }
            double prediction = sums[row] / counts[row];
            residual += (targets[row] - prediction) * (targets[row] - prediction);
            total += (targets[row] - mean) * (targets[row] - mean);
            evaluated++;
        }
        return evaluated == 0 || total == 0.0 ? Double.NaN : 1.0 - residual / total;
    }

    private double computeOobAccuracy(int[][] encodedPredictions, boolean[][] inBag, double[] targets) {
        int rows = targets.length;
        int[] votes = new int[rows * classes.length];
        int[] counts = new int[rows];
        for (int treeIndex = 0; treeIndex < treeCount; treeIndex++) {
            int[] encoded = encodedPredictions[treeIndex];
            boolean[] selected = inBag[treeIndex];
            for (int row = 0; row < rows; row++) {
                if (!selected[row]) {
                    votes[row * classes.length + encoded[row]]++;
                    counts[row]++;
                }
            }
        }
        int correct = 0;
        int evaluated = 0;
        for (int row = 0; row < rows; row++) {
            if (counts[row] == 0) {
                continue;
            }
            int best = 0;
            for (int column = 1; column < classes.length; column++) {
                if (votes[row * classes.length + column] > votes[row * classes.length + best]) {
                    best = column;
                }
            }
            if (classes[best] == targets[row]) {
                correct++;
            }
            evaluated++;
        }
        return evaluated == 0 ? Double.NaN : (double) correct / evaluated;
    }

    @Override
    public double[] predict(Matrix features) {
        MlChecks.predictionData(features, featureCount, trees != null);
        int rows = features.rows();
        double[] result = new double[rows];
        if (task == TreeTask.REGRESSION) {
            double[] data = features.data();
            int columns = features.columns();
            double scale = 1.0 / trees.length;
            if (NativeKernels.available() && packed != null
                    && (long) rows * trees.length >= 20_000_000L
                    && NativeKernels.forestPredictChecked(data, rows, columns,
                    packed.feature, packed.threshold, packed.left, packed.right,
                    packed.value, packed.offsets, result, scale)) {
                return result;
            }
            for (DecisionTree tree : trees) {
                tree.model().addInto(data, rows, columns, result, scale);
            }
            return result;
        }
        Matrix probabilities = predictProbability(features);
        for (int row = 0; row < rows; row++) {
            int best = 0;
            for (int column = 1; column < classes.length; column++) {
                if (probabilities.get(row, column) > probabilities.get(row, best)) {
                    best = column;
                }
            }
            result[row] = classes[best];
        }
        return result;
    }

    public Matrix predictProbability(Matrix features) {
        MlChecks.predictionData(features, featureCount, trees != null);
        if (task != TreeTask.CLASSIFICATION) {
            throw new IllegalStateException("Probabilities require a classification forest");
        }
        int rows = features.rows();
        Matrix result = Matrix.zeros(rows, classes.length);
        double[] votes = result.data();
        double[] data = features.data();
        int columns = features.columns();
        int classCount = classes.length;
        double weight = 1.0 / trees.length;
        if (NativeKernels.available() && packed != null
                && (long) rows * trees.length >= 20_000_000L
                && NativeKernels.forestVoteChecked(data, rows, columns,
                packed.feature, packed.threshold, packed.left, packed.right,
                packed.value, packed.offsets, classes, votes, weight)) {
            return result;
        }
        double[] buffer = new double[rows];
        for (DecisionTree tree : trees) {
            tree.model().predictInto(data, rows, columns, buffer);
            for (int row = 0; row < rows; row++) {
                int classColumn = Arrays.binarySearch(classes, buffer[row]);
                if (classColumn < 0) {
                    throw new IllegalStateException("Unknown tree class");
                }
                votes[row * classCount + classColumn] += weight;
            }
        }
        return result;
    }

    public double[] classes() {
        return classes == null ? null : classes.clone();
    }

    public TreeTask task() {
        return task;
    }

    public double oobScore() {
        if (!oob) {
            throw new IllegalStateException("OOB evaluation was disabled");
        }
        return oobScore;
    }

    public double[] featureImportances() {
        if (trees == null) {
            throw new IllegalStateException("RandomForest must be fitted first");
        }
        double[] result = new double[featureCount];
        for (DecisionTree tree : trees) {
            double[] importance = tree.featureImportances();
            for (int feature = 0; feature < result.length; feature++) {
                result[feature] += importance[feature];
            }
        }
        for (int feature = 0; feature < result.length; feature++) {
            result[feature] /= trees.length;
        }
        double total = 0.0;
        for (double importance : result) {
            total += importance;
        }
        if (total > 0.0) {
            for (int feature = 0; feature < result.length; feature++) {
                result[feature] /= total;
            }
        }
        return result;
    }

    public int treeCount() {
        return trees == null ? 0 : trees.length;
    }

    private PackedTrees packTrees() {
        List<TreeModel> models = new ArrayList<>(trees.length);
        for (DecisionTree tree : trees) {
            models.add(tree.model());
        }
        return PackedTrees.pack(models);
    }

    public void writeState(ModelWriter writer) {
        if (trees == null) {
            throw new IllegalStateException("RandomForest must be fitted before serialization");
        }        writer.putInt(task.ordinal());
        writer.putInt(featureCount);
        if (classes == null) {
            writer.putInt(0);
        } else {
            writer.putInt(classes.length);
            writer.putDoubles(classes);
        }
        writer.putInt(trees.length);
        for (DecisionTree tree : trees) {
            tree.writeState(writer);
        }
    }

    public static RandomForest readState(ModelReader reader) {
        int taskOrdinal = reader.getInt();
        if (taskOrdinal < 0 || taskOrdinal >= TreeTask.values().length) {
            throw new ModelFormatException("Invalid forest task " + taskOrdinal);
        }
        TreeTask task = TreeTask.values()[taskOrdinal];
        int features = reader.getInt();
        int classCount = reader.getInt();
        double[] classes = classCount == 0 ? null : reader.getDoubles(classCount);
        if (task == TreeTask.CLASSIFICATION && (classes == null || classes.length == 0)) {
            throw new ModelFormatException("Classification forest requires class labels");
        }
        int count = reader.getInt();
        if (count <= 0 || count > 100_000) {
            throw new ModelFormatException("Invalid forest tree count " + count);
        }
        DecisionTree[] trees = new DecisionTree[count];
        for (int i = 0; i < count; i++) {
            trees[i] = DecisionTree.readState(reader);
        }
        RandomForest forest = new RandomForest(new Builder().task(task).oob(false));
        forest.featureCount = features;
        forest.classes = classes;
        forest.trees = trees;
        forest.packed = forest.packTrees();
        return forest;
    }

    @Override
    public ModelKind kind() {
        return ModelKind.RANDOM_FOREST;
    }

    @Override
    public void write(ModelWriter writer) {
        writeState(writer);
    }

    public static final class Builder {
        private int treeCount = 100;
        private int maxDepth = 8;
        private int minSamplesLeaf = 1;
        private int maxFeatures;
        private int maxBins = 64;
        private int threads = Runtime.getRuntime().availableProcessors();
        private long seed = 42;
        private TreeTask task = TreeTask.CLASSIFICATION;
        private boolean oob;

        public Builder trees(int treeCount) {
            this.treeCount = treeCount;
            return this;
        }

        public Builder maxDepth(int maxDepth) {
            this.maxDepth = maxDepth;
            return this;
        }

        public Builder minSamplesLeaf(int minSamplesLeaf) {
            this.minSamplesLeaf = minSamplesLeaf;
            return this;
        }

        public Builder maxFeatures(int maxFeatures) {
            this.maxFeatures = maxFeatures;
            return this;
        }

        public Builder bins(int maxBins) {
            this.maxBins = maxBins;
            return this;
        }

        public Builder threads(int threads) {
            this.threads = threads;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder task(TreeTask task) {
            this.task = Objects.requireNonNull(task, "task");
            return this;
        }

        public Builder oob(boolean oob) {
            this.oob = oob;
            return this;
        }

        public RandomForest build() {
            if (treeCount <= 0 || maxDepth < 0 || minSamplesLeaf <= 0
                    || maxFeatures < 0 || threads <= 0) {
                throw new IllegalArgumentException("Invalid random forest configuration");
            }
            if (maxBins < 2 || maxBins > BinnedFeatures.MAX_BINS) {
                throw new IllegalArgumentException("bins must be in [2, " + BinnedFeatures.MAX_BINS + "]");
            }
            return new RandomForest(this);
        }
    }
}
