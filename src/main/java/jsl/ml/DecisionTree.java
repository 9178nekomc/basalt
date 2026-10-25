package jsl.ml;

import jsl.core.Matrix;
import jsl.core.ModelFormatException;
import jsl.core.ModelReader;
import jsl.core.ModelWriter;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;
import java.util.TreeMap;

/**
 * CART tree over pre-binned features. Split search is histogram based: one pass over the node's
 * samples fills per-feature class counts (or sum/sum-of-squares for regression), then every bucket
 * boundary is evaluated in O(classes) or O(1).
 *
 * <p>The fitted topology is flattened into a {@link TreeModel}; inference never touches objects.
 * Random forests reuse one global {@link BinnedFeatures} and call {@link #fitBinned} with bootstrap
 * row indices, which removes the per-tree binning and matrix copy of the naive implementation.</p>
 */
public final class DecisionTree implements Estimator {
    private final int maxDepth;
    private final int minSamplesSplit;
    private final int minSamplesLeaf;
    private final ImpurityCriterion criterion;
    private final TreeTask task;
    private final int maxFeatures;
    private final int maxBins;
    private final long seed;

    private int featureCount;
    private double[] classes;
    private TreeModel model;
    private double[] importances;

    private DecisionTree(Builder builder) {
        maxDepth = builder.maxDepth;
        minSamplesSplit = builder.minSamplesSplit;
        minSamplesLeaf = builder.minSamplesLeaf;
        criterion = builder.criterion;
        task = builder.task;
        maxFeatures = builder.maxFeatures;
        maxBins = builder.maxBins;
        seed = builder.seed;
    }

    public static Builder builder() {
        return new Builder();
    }

    public DecisionTree fit(Matrix features, double[] targets) {
        MlChecks.trainingData(features, targets);
        if (features.rows() < minSamplesLeaf) {
            throw new IllegalArgumentException("Not enough samples for minSamplesLeaf");
        }
        featureCount = features.columns();
        BinnedFeatures binned = BinnedFeatures.build(features, maxBins);
        return fitBinned(binned, allRows(features.rows()), targets);
    }

    /**
     * Fits on shared bins and explicit row indices. Indices may repeat (bootstrap) and may refer
     * to any row of {@code binned}; labels are indexed by the same row ids.
     */
    DecisionTree fitBinned(BinnedFeatures binned, int[] samples, double[] targets) {
        Objects.requireNonNull(binned, "binned");
        Objects.requireNonNull(samples, "samples");
        Objects.requireNonNull(targets, "targets");
        if (binned.features() != featureCount) {
            featureCount = binned.features();
        }
        if (targets.length != binned.rows()) {
            throw new IllegalArgumentException("Target count must equal binned row count");
        }
        if (samples.length == 0) {
            throw new IllegalArgumentException("Cannot fit a tree on zero samples");
        }
        for (int sample : samples) {
            if (sample < 0 || sample >= binned.rows()) {
                throw new IllegalArgumentException("Sample index outside binned rows: " + sample);
            }
        }
        int[] encoded = null;
        double[] regressionTargets = null;
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
            encoded = new int[targets.length];
            for (int row = 0; row < targets.length; row++) {
                encoded[row] = classIndex.get(targets[row]);
            }
        } else {
            classes = null;
            regressionTargets = targets;
        }

        Workspace workspace = new Workspace(binned, task, classes == null ? 0 : classes.length,
                regressionTargets, encoded);
        importances = new double[featureCount];
        Random random = new Random(seed);
        TreeModel builder = new TreeModel(Math.max(16, samples.length / 16));
        build(samples, 0, workspace, random, builder);
        model = builder.compact();
        double total = 0.0;
        for (double importance : importances) {
            total += importance;
        }
        if (total > 0.0) {
            for (int feature = 0; feature < importances.length; feature++) {
                importances[feature] /= total;
            }
        }
        return this;
    }

    private int[] allRows(int rows) {
        int[] result = new int[rows];
        for (int row = 0; row < rows; row++) {
            result[row] = row;
        }
        return result;
    }

    private int build(int[] samples, int depth, Workspace workspace, Random random,
                      TreeModel builder) {
        return task == TreeTask.CLASSIFICATION
                ? buildClassification(samples, depth, workspace, random, builder)
                : buildRegression(samples, depth, workspace, random, builder);
    }

    private int buildClassification(int[] samples, int depth, Workspace workspace, Random random,
                                    TreeModel builder) {
        int node = builder.addNode();
        int[] totalCounts = workspace.totalCounts;
        Arrays.fill(totalCounts, 0);
        for (int sample : samples) {
            totalCounts[workspace.encoded[sample]]++;
        }
        int majority = 0;
        for (int label = 1; label < totalCounts.length; label++) {
            if (totalCounts[label] > totalCounts[majority]) {
                majority = label;
            }
        }
        double parentImpurity = criterion.impurity(totalCounts, samples.length);
        if (depth >= maxDepth || samples.length < minSamplesSplit
                || samples.length < 2 * minSamplesLeaf || parentImpurity <= 0.0) {
            builder.setLeaf(node, classes[majority]);
            return node;
        }

        int classesCount = totalCounts.length;
        int[] histogram = workspace.classHistogram;
        int[] leftCounts = workspace.leftCounts;
        int[] rightCounts = workspace.rightCounts;
        int stride = workspace.maxBins * classesCount;
        int bestFeature = -1;
        int bestBin = -1;
        double bestGain = 0.0;
        double threshold = 0.0;
        int[] features = candidateFeatures(random);

        for (int feature : features) {
            int bins = workspace.binned.binCount(feature);
            int base = feature * stride;
            Arrays.fill(histogram, base, base + bins * classesCount, 0);
            for (int sample : samples) {
                histogram[base + workspace.binned.bin(sample, feature) * classesCount
                        + workspace.encoded[sample]]++;
            }
            System.arraycopy(totalCounts, 0, rightCounts, 0, classesCount);
            Arrays.fill(leftCounts, 0);
            int leftSize = 0;
            for (int bin = 0; bin < bins - 1; bin++) {
                int offset = base + bin * classesCount;
                for (int label = 0; label < classesCount; label++) {
                    int count = histogram[offset + label];
                    leftCounts[label] += count;
                    rightCounts[label] -= count;
                }
                leftSize += histogramCount(histogram, offset, classesCount);
                int rightSize = samples.length - leftSize;
                if (leftSize < minSamplesLeaf || rightSize < minSamplesLeaf) {
                    continue;
                }
                double gain = parentImpurity
                        - (double) leftSize / samples.length
                        * criterion.impurity(leftCounts, leftSize)
                        - (double) rightSize / samples.length
                        * criterion.impurity(rightCounts, rightSize);
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = feature;
                    bestBin = bin;
                    threshold = workspace.binned.threshold(feature, bin);
                }
            }
        }
        if (bestFeature < 0 || bestGain <= 1e-12 * Math.max(1.0, parentImpurity)) {
            builder.setLeaf(node, classes[majority]);
            return node;
        }
        int leftSize = 0;
        for (int sample : samples) {
            if (workspace.binned.bin(sample, bestFeature) <= bestBin) {
                leftSize++;
            }
        }
        if (leftSize < minSamplesLeaf || samples.length - leftSize < minSamplesLeaf) {
            builder.setLeaf(node, classes[majority]);
            return node;
        }
        int[] left = new int[leftSize];
        int[] right = new int[samples.length - leftSize];
        int leftIndex = 0;
        int rightIndex = 0;
        for (int sample : samples) {
            if (workspace.binned.bin(sample, bestFeature) <= bestBin) {
                left[leftIndex++] = sample;
            } else {
                right[rightIndex++] = sample;
            }
        }
        importances[bestFeature] += bestGain * samples.length;
        int leftNode = build(left, depth + 1, workspace, random, builder);
        int rightNode = build(right, depth + 1, workspace, random, builder);
        builder.setSplit(node, bestFeature, threshold, leftNode, rightNode);
        return node;
    }

    private int buildRegression(int[] samples, int depth, Workspace workspace, Random random,
                                TreeModel builder) {
        int node = builder.addNode();
        double totalSum = 0.0;
        double totalSumSquares = 0.0;
        for (int sample : samples) {
            double target = workspace.regressionTargets[sample];
            totalSum += target;
            totalSumSquares += target * target;
        }
        double mean = totalSum / samples.length;
        double parentImpurity = Math.max(0.0, totalSumSquares / samples.length - mean * mean);
        if (depth >= maxDepth || samples.length < minSamplesSplit
                || samples.length < 2 * minSamplesLeaf
                || parentImpurity <= 1e-12 * Math.max(1e-12, mean * mean)) {
            builder.setLeaf(node, mean);
            return node;
        }

        double[] sum = workspace.regressionSum;
        double[] sumSquares = workspace.regressionSumSquares;
        int[] counts = workspace.regressionCounts;
        int stride = workspace.maxBins;
        int bestFeature = -1;
        int bestBin = -1;
        double bestGain = 0.0;
        double threshold = 0.0;
        int[] features = candidateFeatures(random);

        for (int feature : features) {
            int bins = workspace.binned.binCount(feature);
            int base = feature * stride;
            Arrays.fill(sum, base, base + bins, 0.0);
            Arrays.fill(sumSquares, base, base + bins, 0.0);
            Arrays.fill(counts, base, base + bins, 0);
            for (int sample : samples) {
                int index = base + workspace.binned.bin(sample, feature);
                double target = workspace.regressionTargets[sample];
                sum[index] += target;
                sumSquares[index] += target * target;
                counts[index]++;
            }
            double leftSum = 0.0;
            double leftSumSquares = 0.0;
            int leftSize = 0;
            for (int bin = 0; bin < bins - 1; bin++) {
                leftSum += sum[base + bin];
                leftSumSquares += sumSquares[base + bin];
                leftSize += counts[base + bin];
                int rightSize = samples.length - leftSize;
                if (leftSize < minSamplesLeaf || rightSize < minSamplesLeaf) {
                    continue;
                }
                double leftMean = leftSum / leftSize;
                double rightSum = totalSum - leftSum;
                double rightMean = rightSum / rightSize;
                double leftVariance = Math.max(0.0, leftSumSquares / leftSize - leftMean * leftMean);
                double rightVariance = Math.max(0.0,
                        (totalSumSquares - leftSumSquares) / rightSize - rightMean * rightMean);
                double gain = parentImpurity
                        - (double) leftSize / samples.length * leftVariance
                        - (double) rightSize / samples.length * rightVariance;
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = feature;
                    bestBin = bin;
                    threshold = workspace.binned.threshold(feature, bin);
                }
            }
        }
        if (bestFeature < 0 || bestGain <= 1e-12 * Math.max(1.0, parentImpurity)) {
            builder.setLeaf(node, mean);
            return node;
        }
        int leftSize = 0;
        for (int sample : samples) {
            if (workspace.binned.bin(sample, bestFeature) <= bestBin) {
                leftSize++;
            }
        }
        if (leftSize < minSamplesLeaf || samples.length - leftSize < minSamplesLeaf) {
            builder.setLeaf(node, mean);
            return node;
        }
        int[] left = new int[leftSize];
        int[] right = new int[samples.length - leftSize];
        int leftIndex = 0;
        int rightIndex = 0;
        for (int sample : samples) {
            if (workspace.binned.bin(sample, bestFeature) <= bestBin) {
                left[leftIndex++] = sample;
            } else {
                right[rightIndex++] = sample;
            }
        }
        importances[bestFeature] += bestGain * samples.length;
        int leftNode = build(left, depth + 1, workspace, random, builder);
        int rightNode = build(right, depth + 1, workspace, random, builder);
        builder.setSplit(node, bestFeature, threshold, leftNode, rightNode);
        return node;
    }

    private static int histogramCount(int[] histogram, int offset, int classesCount) {
        int sum = 0;
        for (int label = 0; label < classesCount; label++) {
            sum += histogram[offset + label];
        }
        return sum;
    }

    private int[] candidateFeatures(Random random) {
        int limit = maxFeatures == 0 ? featureCount : Math.min(maxFeatures, featureCount);
        if (limit == featureCount) {
            int[] all = new int[featureCount];
            for (int i = 0; i < featureCount; i++) {
                all[i] = i;
            }
            return all;
        }
        int[] candidates = new int[featureCount];
        for (int i = 0; i < featureCount; i++) {
            candidates[i] = i;
        }
        for (int i = 0; i < limit; i++) {
            int selected = i + random.nextInt(featureCount - i);
            int temporary = candidates[i];
            candidates[i] = candidates[selected];
            candidates[selected] = temporary;
        }
        return Arrays.copyOf(candidates, limit);
    }

    public double[] predict(Matrix features) {
        MlChecks.predictionData(features, featureCount, model != null);
        double[] predictions = new double[features.rows()];
        model.addInto(features.data(), features.rows(), features.columns(), predictions, 1.0);
        return predictions;
    }

    public double[] featureImportances() {
        if (importances == null) {
            throw new IllegalStateException("DecisionTree must be fitted first");
        }
        return importances.clone();
    }

    public TreeTask task() {
        return task;
    }

    public double[] classes() {
        return classes == null ? null : classes.clone();
    }

    public int featureCount() {
        return featureCount;
    }

    public int nodeCount() {
        return model == null ? 0 : model.size();
    }

    public int depth() {
        return model == null ? 0 : model.depth();
    }

    TreeModel model() {
        return model;
    }

    /** Writes the fitted topology; the estimator configuration is reconstructed by the reader. */
    public void writeState(ModelWriter writer) {
        if (model == null) {
            throw new IllegalStateException("DecisionTree must be fitted before serialization");
        }
        writer.putInt(featureCount);
        writer.putInt(task.ordinal());
        if (classes == null) {
            writer.putInt(0);
        } else {
            writer.putInt(classes.length);
            writer.putDoubles(classes);
        }
        model.write(writer);
    }

    public static DecisionTree readState(ModelReader reader) {
        int featureCount = reader.getInt();
        int taskOrdinal = reader.getInt();
        if (taskOrdinal < 0 || taskOrdinal >= TreeTask.values().length) {
            throw new ModelFormatException("Invalid tree task " + taskOrdinal);
        }
        TreeTask task = TreeTask.values()[taskOrdinal];
        int classCount = reader.getInt();
        double[] classes = classCount == 0 ? null : reader.getDoubles(classCount);
        if (task == TreeTask.CLASSIFICATION && (classes == null || classes.length == 0)) {
            throw new ModelFormatException("Classification tree requires class labels");
        }
        TreeModel model = TreeModel.read(reader);
        DecisionTree tree = new DecisionTree(new Builder()
                .task(task)
                .criterion(task == TreeTask.REGRESSION ? ImpurityCriterion.MSE : ImpurityCriterion.GINI));
        tree.featureCount = featureCount;
        tree.classes = classes;
        tree.model = model;
        tree.importances = new double[featureCount];
        return tree;
    }

    private static final class Workspace {
        private final BinnedFeatures binned;
        private final int maxBins;
        private final int[] encoded;
        private final double[] regressionTargets;
        private final int[] totalCounts;
        private final int[] classHistogram;
        private final int[] leftCounts;
        private final int[] rightCounts;
        private final double[] regressionSum;
        private final double[] regressionSumSquares;
        private final int[] regressionCounts;

        private Workspace(BinnedFeatures binned, TreeTask task, int classCount,
                          double[] regressionTargets, int[] encoded) {
            this.binned = binned;
            this.maxBins = binned.maxBinCount();
            this.encoded = encoded;
            this.regressionTargets = regressionTargets;
            int features = binned.features();
            if (task == TreeTask.CLASSIFICATION) {
                totalCounts = new int[classCount];
                classHistogram = new int[features * maxBins * classCount];
                leftCounts = new int[classCount];
                rightCounts = new int[classCount];
                regressionSum = new double[0];
                regressionSumSquares = new double[0];
                regressionCounts = new int[0];
            } else {
                totalCounts = new int[0];
                classHistogram = new int[0];
                leftCounts = new int[0];
                rightCounts = new int[0];
                regressionSum = new double[features * maxBins];
                regressionSumSquares = new double[features * maxBins];
                regressionCounts = new int[features * maxBins];
            }
        }
    }

    public static final class Builder {
        private int maxDepth = 8;
        private int minSamplesSplit = 2;
        private int minSamplesLeaf = 1;
        private ImpurityCriterion criterion = ImpurityCriterion.GINI;
        private TreeTask task = TreeTask.CLASSIFICATION;
        private int maxFeatures;
        private int maxBins = 64;
        private long seed = 42;
        private boolean criterionExplicit;

        public Builder maxDepth(int maxDepth) {
            this.maxDepth = maxDepth;
            return this;
        }

        public Builder minSamplesSplit(int minSamplesSplit) {
            this.minSamplesSplit = minSamplesSplit;
            return this;
        }

        public Builder minSamplesLeaf(int minSamplesLeaf) {
            this.minSamplesLeaf = minSamplesLeaf;
            return this;
        }

        public Builder criterion(ImpurityCriterion criterion) {
            this.criterion = Objects.requireNonNull(criterion, "criterion");
            this.criterionExplicit = true;
            return this;
        }

        public Builder task(TreeTask task) {
            this.task = Objects.requireNonNull(task, "task");
            if (!criterionExplicit) {
                this.criterion = task == TreeTask.REGRESSION
                        ? ImpurityCriterion.MSE : ImpurityCriterion.GINI;
            }
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

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public DecisionTree build() {
            if (maxDepth < 0 || minSamplesSplit < 2 || minSamplesLeaf <= 0 || maxFeatures < 0) {
                throw new IllegalArgumentException(
                        "Require maxDepth >= 0, minSamplesSplit >= 2 and minSamplesLeaf >= 1");
            }
            if (maxBins < 2 || maxBins > BinnedFeatures.MAX_BINS) {
                throw new IllegalArgumentException("bins must be in [2, " + BinnedFeatures.MAX_BINS + "]");
            }
            if (criterion.task() != task) {
                throw new IllegalArgumentException(
                        "Criterion " + criterion + " does not support task " + task);
            }
            return new DecisionTree(this);
        }
    }
}
