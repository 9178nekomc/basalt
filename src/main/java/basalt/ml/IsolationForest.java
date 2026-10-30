package basalt.ml;

import basalt.core.Matrix;
import basalt.core.ModelFormatException;
import basalt.core.ModelKind;
import basalt.core.ModelReader;
import basalt.core.ModelWriter;
import basalt.core.Persistable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Unsupervised Isolation Forest anomaly detector based on random axis-aligned partitioning
 * trees and path-length normalization {@code s(x, n) = 2^(-E(h(x))/c(n))}.
 *
 * <p>{@link #predict(Matrix)} returns {@code +1.0} for inliers and {@code -1.0} for outliers
 * according to the configured {@code contamination} quantile threshold.</p>
 */
public final class IsolationForest implements Estimator, Persistable {
    private static final double EULER_GAMMA = 0.57721566490153286;

    private final int treeCount;
    private final int maxSamples;
    private final double contamination;
    private final long seed;

    private int featureCount;
    private int effectiveMaxSamples;
    private double threshold;
    private final List<TreeModel> trees = new ArrayList<>();

    private IsolationForest(Builder builder) {
        this.treeCount = builder.trees;
        this.maxSamples = builder.maxSamples;
        this.contamination = builder.contamination;
        this.seed = builder.seed;
    }

    public static Builder builder() {
        return new Builder();
    }

    public IsolationForest fit(Matrix features) {
        return fit(features, null);
    }

    @Override
    public IsolationForest fit(Matrix features, double[] ignoredTargets) {
        MlChecks.features(features);
        int rows = features.rows();
        featureCount = features.columns();
        effectiveMaxSamples = Math.min(rows, maxSamples);
        int heightLimit = Math.max(1, (int) Math.ceil(Math.log(effectiveMaxSamples) / Math.log(2.0)));

        trees.clear();
        double[] data = features.data();
        Random masterRandom = new Random(seed);

        for (int t = 0; t < treeCount; t++) {
            Random rng = new Random(masterRandom.nextLong());
            int[] sampleIndices = sampleWithoutReplacement(rows, effectiveMaxSamples, rng);
            TreeModel tree = new TreeModel(effectiveMaxSamples * 2);
            buildIsolationTree(data, sampleIndices, 0, heightLimit, rng, tree);
            trees.add(tree.compact());
        }

        double[] rawScores = decisionFunctionRaw(features);
        double[] sorted = rawScores.clone();
        Arrays.sort(sorted);
        int cutoffIdx = Math.max(0, Math.min(rows - 1, (int) Math.floor(rows * contamination)));
        threshold = sorted[cutoffIdx];
        return this;
    }

    private int[] sampleWithoutReplacement(int total, int count, Random rng) {
        int[] pool = new int[total];
        for (int i = 0; i < total; i++) pool[i] = i;
        for (int i = 0; i < count; i++) {
            int j = i + rng.nextInt(total - i);
            int tmp = pool[i];
            pool[i] = pool[j];
            pool[j] = tmp;
        }
        return Arrays.copyOf(pool, count);
    }

    private int buildIsolationTree(double[] data, int[] samples, int depth, int heightLimit,
                                   Random rng, TreeModel builder) {
        int node = builder.addNode();
        if (depth >= heightLimit || samples.length <= 1) {
            builder.setLeaf(node, depth + cFactor(samples.length));
            return node;
        }
        // Pick a random feature that has min < max
        int[] featOrder = new int[featureCount];
        for (int i = 0; i < featureCount; i++) featOrder[i] = i;
        for (int i = featureCount - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = featOrder[i];
            featOrder[i] = featOrder[j];
            featOrder[j] = tmp;
        }
        int chosenFeature = -1;
        double minVal = 0.0;
        double maxVal = 0.0;
        for (int f : featOrder) {
            double mn = Double.POSITIVE_INFINITY;
            double mx = Double.NEGATIVE_INFINITY;
            for (int s : samples) {
                double v = data[s * featureCount + f];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            if (mx > mn) {
                chosenFeature = f;
                minVal = mn;
                maxVal = mx;
                break;
            }
        }
        if (chosenFeature < 0) {
            builder.setLeaf(node, depth + cFactor(samples.length));
            return node;
        }
        double splitVal = minVal + rng.nextDouble() * (maxVal - minVal);
        int leftCount = 0;
        for (int s : samples) {
            if (data[s * featureCount + chosenFeature] <= splitVal) {
                leftCount++;
            }
        }
        if (leftCount == 0 || leftCount == samples.length) {
            builder.setLeaf(node, depth + cFactor(samples.length));
            return node;
        }
        int[] left = new int[leftCount];
        int[] right = new int[samples.length - leftCount];
        int li = 0, ri = 0;
        for (int s : samples) {
            if (data[s * featureCount + chosenFeature] <= splitVal) {
                left[li++] = s;
            } else {
                right[ri++] = s;
            }
        }
        int leftNode = buildIsolationTree(data, left, depth + 1, heightLimit, rng, builder);
        int rightNode = buildIsolationTree(data, right, depth + 1, heightLimit, rng, builder);
        builder.setSplit(node, chosenFeature, splitVal, leftNode, rightNode);
        return node;
    }

    static double cFactor(int n) {
        if (n <= 1) return 0.0;
        if (n == 2) return 1.0;
        return 2.0 * (Math.log(n - 1.0) + EULER_GAMMA) - 2.0 * (n - 1.0) / n;
    }

    /**
     * Anomaly score in {@code (0, 1]} where values close to 1 indicate strong anomalies and
     * values below 0.5 indicate normal inliers.
     */
    public double[] anomalyScore(Matrix features) {
        MlChecks.predictionData(features, featureCount, !trees.isEmpty());
        int rows = features.rows();
        double[] avgPath = new double[rows];
        double scale = 1.0 / trees.size();
        for (TreeModel tree : trees) {
            tree.addInto(features.data(), rows, featureCount, avgPath, scale);
        }
        double cn = Math.max(1e-12, cFactor(effectiveMaxSamples));
        double[] scores = new double[rows];
        for (int r = 0; r < rows; r++) {
            scores[r] = Math.pow(2.0, -avgPath[r] / cn);
        }
        return scores;
    }

    private double[] decisionFunctionRaw(Matrix features) {
        double[] s = anomalyScore(features);
        for (int i = 0; i < s.length; i++) {
            s[i] = 0.5 - s[i];
        }
        return s;
    }

    /**
     * Shifted decision function where positive values represent inliers and negative values
     * represent outliers.
     */
    public double[] decisionFunction(Matrix features) {
        double[] raw = decisionFunctionRaw(features);
        for (int i = 0; i < raw.length; i++) {
            raw[i] -= threshold;
        }
        return raw;
    }

    @Override
    public double[] predict(Matrix features) {
        double[] scores = decisionFunction(features);
        double[] preds = new double[scores.length];
        for (int i = 0; i < scores.length; i++) {
            preds[i] = scores[i] >= 0.0 ? 1.0 : -1.0;
        }
        return preds;
    }

    @Override
    public ModelKind kind() {
        return ModelKind.ISOLATION_FOREST;
    }

    @Override
    public void write(ModelWriter writer) {
        if (trees.isEmpty()) {
            throw new IllegalStateException("IsolationForest must be fitted before serialization");
        }
        writer.putInt(treeCount);
        writer.putInt(maxSamples);
        writer.putDouble(contamination);
        writer.putLong(seed);
        writer.putInt(featureCount);
        writer.putInt(effectiveMaxSamples);
        writer.putDouble(threshold);
        writer.putInt(trees.size());
        for (TreeModel tree : trees) {
            tree.write(writer);
        }
    }

    public static IsolationForest read(ModelReader reader) {
        int tCount = reader.getInt();
        int mSamples = reader.getInt();
        double contam = reader.getDouble();
        long seed = reader.getLong();
        int features = reader.getInt();
        int effSamples = reader.getInt();
        double thresh = reader.getDouble();
        int count = reader.getInt();
        if (tCount <= 0 || features <= 0 || count <= 0) {
            throw new ModelFormatException("Invalid IsolationForest header");
        }
        IsolationForest model = new Builder()
                .trees(tCount)
                .maxSamples(mSamples)
                .contamination(contam)
                .seed(seed)
                .build();
        model.featureCount = features;
        model.effectiveMaxSamples = effSamples;
        model.threshold = thresh;
        for (int i = 0; i < count; i++) {
            model.trees.add(TreeModel.read(reader));
        }
        return model;
    }

    public static final class Builder {
        private int trees = 100;
        private int maxSamples = 256;
        private double contamination = 0.1;
        private long seed = 42L;

        public Builder trees(int trees) {
            if (trees <= 0) throw new IllegalArgumentException("trees must be positive");
            this.trees = trees;
            return this;
        }

        public Builder maxSamples(int maxSamples) {
            if (maxSamples < 2) throw new IllegalArgumentException("maxSamples must be >= 2");
            this.maxSamples = maxSamples;
            return this;
        }

        public Builder contamination(double contamination) {
            if (!(contamination > 0.0) || contamination >= 0.5) {
                throw new IllegalArgumentException("contamination must be in (0, 0.5)");
            }
            this.contamination = contamination;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public IsolationForest build() {
            return new IsolationForest(this);
        }
    }
}
