package basalt.ml;

import basalt.core.Matrix;
import basalt.core.ModelFormatException;
import basalt.core.ModelKind;
import basalt.core.ModelReader;
import basalt.core.ModelWriter;
import basalt.core.NativeKernels;
import basalt.core.Persistable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * Second-order histogram gradient boosting with two growth policies: depth-wise (XGBoost) and
 * best-leaf-first (LightGBM).
 *
 * <p>Split gain follows the regularized objective
 * {@code 0.5 * (G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)) - gamma}.
 * Histograms use the LightGBM subtraction trick: each split accumulates the histogram of the
 * <em>smaller</em> child only, and the larger child is derived as {@code parent - small}. A
 * balanced tree therefore touches roughly half the sample-feature work of naive per-node
 * accumulation. Large accumulations are dispatched to a Rust AVX2 kernel through JNI; any native
 * failure falls back to the parallel Java path.</p>
 */
public final class HistogramGradientBoosting implements Estimator, Persistable {
    private static final long PARALLEL_WORK_THRESHOLD = 50_000L;
    private static final long NATIVE_HISTOGRAM_WORK = 200_000L;
    private static final int MIN_CHUNK_SAMPLES = 256;

    private final BoostingAlgorithm algorithm;
    private final BoostingObjective objective;
    private final int iterations;
    private final double learningRate;
    private final int maxDepth;
    private final int maxLeaves;
    private final int binCount;
    private final int minSamplesLeaf;
    private final double minChildWeight;
    private final double lambda;
    private final double gamma;
    private final int threads;
    private final double featureFraction;
    private final double validationFraction;
    private final int earlyStoppingRounds;
    private final long seed;

    private final List<TreeModel> trees = new ArrayList<>();
    private PackedTrees packed;
    private BinnedFeatures binned;
    private double[] gradients;
    private double[] hessians;
    private double baseScore;
    private int featureCount;
    private double[] lossHistory = new double[0];
    private double[] validationHistory = new double[0];
    private double[] importances = new double[0];
    private int bestIteration;
    private ForkJoinPool pool;
    private HistogramWorkspace workspace;
    private ArrayDeque<NodeHistogram> histogramPool;
    private int histogramSize;
    private long nativeHistogramNanos;
    private long nativeHistogramCalls;
    private long javaHistogramNanos;
    private long histogramSession;
    private long binNanos;
    private long gradientNanos;
    private long growNanos;
    private long addNanos;

    private HistogramGradientBoosting(Builder builder) {
        algorithm = builder.algorithm;
        objective = builder.objective;
        iterations = builder.iterations;
        learningRate = builder.learningRate;
        maxDepth = builder.maxDepth;
        maxLeaves = builder.maxLeaves;
        binCount = builder.binCount;
        minSamplesLeaf = builder.minSamplesLeaf;
        minChildWeight = builder.minChildWeight;
        lambda = builder.lambda;
        gamma = builder.gamma;
        threads = builder.threads;
        featureFraction = builder.featureFraction;
        validationFraction = builder.validationFraction;
        earlyStoppingRounds = builder.earlyStoppingRounds;
        seed = builder.seed;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public HistogramGradientBoosting fit(Matrix features, double[] targets) {
        MlChecks.trainingData(features, targets);
        if (objective == BoostingObjective.LOGISTIC) {
            for (double target : targets) {
                if (target != 0.0 && target != 1.0) {
                    throw new IllegalArgumentException("Logistic boosting targets must be 0 or 1");
                }
            }
        }
        featureCount = features.columns();
        int rows = features.rows();
        trees.clear();
        long binStart = System.nanoTime();
        binned = BinnedFeatures.build(features, binCount);
        binNanos += System.nanoTime() - binStart;
        gradients = new double[rows];
        hessians = new double[rows];

        int[] trainRows = allRows(rows);
        int[] validationRows = new int[0];
        if (validationFraction > 0.0) {
            int[] shuffled = allRows(rows);
            Random random = new Random(seed);
            for (int i = rows - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int temporary = shuffled[i];
                shuffled[i] = shuffled[j];
                shuffled[j] = temporary;
            }
            int validationSize = Math.max(1, (int) Math.round(rows * validationFraction));
            validationSize = Math.min(validationSize, rows - 2);
            validationRows = Arrays.copyOfRange(shuffled, 0, validationSize);
            trainRows = Arrays.copyOfRange(shuffled, validationSize, rows);
        }

        double mean = 0.0;
        for (int row : trainRows) {
            mean += targets[row];
        }
        mean /= trainRows.length;
        baseScore = objective == BoostingObjective.LOGISTIC
                ? Math.log(Math.max(1e-12, mean) / Math.max(1e-12, 1.0 - mean)) : mean;
        double[] raw = new double[rows];
        for (int row : trainRows) {
            raw[row] = baseScore;
        }
        double[] validationRaw = new double[validationRows.length];
        Arrays.fill(validationRaw, baseScore);
        lossHistory = new double[iterations];
        validationHistory = new double[iterations];
        importances = new double[featureCount];
        bestIteration = iterations - 1;
        double bestValidation = Double.POSITIVE_INFINITY;
        int roundsWithoutImprovement = 0;
        int trained = 0;

        pool = new ForkJoinPool(threads);
        workspace = new HistogramWorkspace(featureCount, binned.maxBinCount(), threads);
        histogramSize = featureCount * binned.maxBinCount();
        histogramPool = new ArrayDeque<>();
        histogramSession = NativeKernels.available()
                ? NativeKernels.createHistogramSession(binned.bins(), gradients, hessians,
                featureCount, binned.maxBinCount())
                : 0L;
        try {
            for (int iteration = 0; iteration < iterations; iteration++) {
                double loss = 0.0;
                long gradientStart = System.nanoTime();
                for (int row : trainRows) {
                    if (objective == BoostingObjective.LOGISTIC) {
                        // One exp per row: derive both the sigmoid and softplus from exp(-|x|).
                        double value = raw[row];
                        double exponential = Math.exp(-Math.abs(value));
                        double probability;
                        double softplus;
                        if (value >= 0.0) {
                            probability = 1.0 / (1.0 + exponential);
                            softplus = value + Math.log1p(exponential);
                        } else {
                            probability = exponential / (1.0 + exponential);
                            softplus = Math.log1p(exponential);
                        }
                        gradients[row] = probability - targets[row];
                        hessians[row] = Math.max(1e-12, probability * (1.0 - probability));
                        loss += softplus - value * targets[row];
                    } else {
                        gradients[row] = raw[row] - targets[row];
                        hessians[row] = 1.0;
                        double error = raw[row] - targets[row];
                        loss += 0.5 * error * error;
                    }
                }
                gradientNanos += System.nanoTime() - gradientStart;
                if (histogramSession != 0L) {
                    NativeKernels.updateHistogramSession(histogramSession, gradients, hessians);
                }
                lossHistory[iteration] = loss / trainRows.length;
                int[] activeFeatures = selectFeatures(iteration);
                TreeModel tree = new TreeModel(Math.max(16, trainRows.length / 8));
                long growStart = System.nanoTime();
                if (algorithm == BoostingAlgorithm.XGBOOST) {
                    NodeHistogram rootHistogram = rentHistogram();
                    accumulateHistogram(rootHistogram, trainRows, activeFeatures);
                    buildDepthWise(trainRows, 0, activeFeatures, rootHistogram, tree);
                } else {
                    buildLeafWise(trainRows, activeFeatures, tree);
                }
                growNanos += System.nanoTime() - growStart;
                TreeModel compact = tree.compact();
                trees.add(compact);
                trained = iteration + 1;
                long addStart = System.nanoTime();
                compact.addInto(features.data(), rows, featureCount, raw, learningRate);
                addNanos += System.nanoTime() - addStart;
                if (validationRows.length > 0) {
                    double validationLoss = 0.0;
                    for (int index = 0; index < validationRows.length; index++) {
                        int row = validationRows[index];
                        validationRaw[index] += learningRate
                                * compact.predict(features.data(), row * featureCount);
                        validationLoss += objectiveLoss(validationRaw[index], targets[row]);
                    }
                    validationLoss /= validationRows.length;
                    validationHistory[iteration] = validationLoss;
                    if (validationLoss < bestValidation - 1e-9) {
                        bestValidation = validationLoss;
                        bestIteration = iteration;
                        roundsWithoutImprovement = 0;
                    } else {
                        roundsWithoutImprovement++;
                        if (earlyStoppingRounds > 0
                                && roundsWithoutImprovement >= earlyStoppingRounds) {
                            break;
                        }
                    }
                } else {
                    bestIteration = iteration;
                }
            }
        } finally {
            if (histogramSession != 0L) {
                NativeKernels.destroyHistogramSession(histogramSession);
                histogramSession = 0L;
            }
            pool.shutdown();
            pool = null;
            binned = null;
            gradients = null;
            hessians = null;
            workspace = null;
            histogramPool = null;
        }
        if (earlyStoppingRounds > 0 && validationRows.length > 0
                && trees.size() > bestIteration + 1) {
            trees.subList(bestIteration + 1, trees.size()).clear();
            trained = bestIteration + 1;
        }
        packed = PackedTrees.pack(trees);
        lossHistory = Arrays.copyOf(lossHistory, trained);
        validationHistory = validationRows.length == 0
                ? new double[0] : Arrays.copyOf(validationHistory, trained);
        double total = 0.0;
        for (double importance : importances) {
            total += importance;
        }
        if (total > 0.0) {
            for (int feature = 0; feature < importances.length; feature++) {
                importances[feature] /= total;
            }
        }
        if (Boolean.getBoolean("basalt.hgb.profile")) {
            System.out.printf(Locale.ROOT,
                    "hgb native=%d/%.1fms javaHist=%.1f bin=%.1f grad=%.1f grow=%.1f add=%.1f%n",
                    nativeHistogramCalls, nativeHistogramNanos / 1e6, javaHistogramNanos / 1e6,
                    binNanos / 1e6, gradientNanos / 1e6, growNanos / 1e6, addNanos / 1e6);
        }
        return this;
    }

    private double objectiveLoss(double raw, double target) {
        if (objective == BoostingObjective.LOGISTIC) {
            return Math.max(raw, 0.0) - raw * target + Math.log1p(Math.exp(-Math.abs(raw)));
        }
        double error = raw - target;
        return 0.5 * error * error;
    }

    private int[] allRows(int rows) {
        int[] result = new int[rows];
        for (int row = 0; row < rows; row++) {
            result[row] = row;
        }
        return result;
    }

    private int[] selectFeatures(int iteration) {
        if (featureFraction >= 1.0) {
            return allRows(featureCount);
        }
        int count = Math.max(1, (int) Math.ceil(featureFraction * featureCount));
        int[] candidates = allRows(featureCount);
        Random random = new Random(seed + iteration * 7_919L);
        for (int i = 0; i < count; i++) {
            int selected = i + random.nextInt(featureCount - i);
            int temporary = candidates[i];
            candidates[i] = candidates[selected];
            candidates[selected] = temporary;
        }
        return Arrays.copyOf(candidates, count);
    }

    // ---------------------------------------------------------------- tree construction

    private int buildDepthWise(int[] samples, int depth, int[] activeFeatures,
                               NodeHistogram histogram, TreeModel builder) {
        int node = builder.addNode();
        builder.setLeaf(node, leafWeight(samples));
        if (depth >= maxDepth || samples.length < 2 * minSamplesLeaf) {
            histogramPool.push(histogram);
            return node;
        }
        SplitDecision decision = bestSplit(activeFeatures, histogram, samples.length);
        if (decision == null || decision.gain <= 0.0) {
            histogramPool.push(histogram);
            return node;
        }
        int leftSize = countAtOrBelow(histogram, decision);
        if (leftSize < minSamplesLeaf || samples.length - leftSize < minSamplesLeaf) {
            histogramPool.push(histogram);
            return node;
        }
        int[] left = new int[leftSize];
        int[] right = new int[samples.length - leftSize];
        partition(samples, decision, left, right);
        importances[decision.feature] += decision.gain * samples.length;

        boolean leftIsSmall = left.length <= right.length;
        int[] smallSamples = leftIsSmall ? left : right;
        NodeHistogram smallHistogram = rentHistogram();
        NodeHistogram largeHistogram = rentHistogram();
        accumulateHistogram(smallHistogram, smallSamples, activeFeatures);
        subtractHistogram(histogram, smallHistogram, largeHistogram, activeFeatures);
        histogramPool.push(histogram);

        int leftNode = buildDepthWise(left, depth + 1, activeFeatures,
                leftIsSmall ? smallHistogram : largeHistogram, builder);
        int rightNode = buildDepthWise(right, depth + 1, activeFeatures,
                leftIsSmall ? largeHistogram : smallHistogram, builder);
        builder.setSplit(node, decision.feature, decision.threshold, leftNode, rightNode);
        return node;
    }

    private int buildLeafWise(int[] samples, int[] activeFeatures, TreeModel builder) {
        int root = builder.addNode();
        builder.setLeaf(root, leafWeight(samples));
        PriorityQueue<LeafCandidate> frontier = new PriorityQueue<>(
                (left, right) -> Double.compare(right.decision.gain, left.decision.gain));
        NodeHistogram rootHistogram = rentHistogram();
        accumulateHistogram(rootHistogram, samples, activeFeatures);
        addCandidate(frontier, root, samples, 0, activeFeatures, rootHistogram);
        int leaves = 1;
        while (leaves < maxLeaves && !frontier.isEmpty()) {
            LeafCandidate candidate = frontier.poll();
            SplitDecision decision = candidate.decision;
            int leftSize = countAtOrBelow(candidate.histogram, decision);
            if (leftSize < minSamplesLeaf
                    || candidate.samples.length - leftSize < minSamplesLeaf) {
                histogramPool.push(candidate.histogram);
                continue;
            }
            int[] left = new int[leftSize];
            int[] right = new int[candidate.samples.length - leftSize];
            partition(candidate.samples, decision, left, right);
            importances[decision.feature] += decision.gain * candidate.samples.length;
            int leftNode = builder.addNode();
            builder.setLeaf(leftNode, leafWeight(left));
            int rightNode = builder.addNode();
            builder.setLeaf(rightNode, leafWeight(right));
            builder.setSplit(candidate.node, decision.feature, decision.threshold,
                    leftNode, rightNode);
            leaves++;

            boolean leftIsSmall = left.length <= right.length;
            int[] smallSamples = leftIsSmall ? left : right;
            NodeHistogram smallHistogram = rentHistogram();
            NodeHistogram largeHistogram = rentHistogram();
            accumulateHistogram(smallHistogram, smallSamples, activeFeatures);
            subtractHistogram(candidate.histogram, smallHistogram, largeHistogram, activeFeatures);
            histogramPool.push(candidate.histogram);

            addCandidate(frontier, leftNode, left, candidate.depth + 1, activeFeatures,
                    leftIsSmall ? smallHistogram : largeHistogram);
            addCandidate(frontier, rightNode, right, candidate.depth + 1, activeFeatures,
                    leftIsSmall ? largeHistogram : smallHistogram);
        }
        return root;
    }

    private void addCandidate(PriorityQueue<LeafCandidate> frontier, int node, int[] samples,
                              int depth, int[] activeFeatures, NodeHistogram histogram) {
        if (depth >= maxDepth || samples.length < 2 * minSamplesLeaf) {
            histogramPool.push(histogram);
            return;
        }
        SplitDecision decision = bestSplit(activeFeatures, histogram, samples.length);
        if (decision != null && decision.gain > 0.0) {
            frontier.add(new LeafCandidate(node, samples, depth, decision, histogram));
        } else {
            histogramPool.push(histogram);
        }
    }

    private void partition(int[] samples, SplitDecision decision, int[] left, int[] right) {
        int leftIndex = 0;
        int rightIndex = 0;
        for (int sample : samples) {
            if (binned.bin(sample, decision.feature) <= decision.bin) {
                left[leftIndex++] = sample;
            } else {
                right[rightIndex++] = sample;
            }
        }
    }

    private int countAtOrBelow(NodeHistogram histogram, SplitDecision decision) {
        int base = decision.feature * workspace.maxBins;
        int total = 0;
        for (int bin = 0; bin <= decision.bin; bin++) {
            total += histogram.count[base + bin];
        }
        return total;
    }

    // ---------------------------------------------------------------- histograms

    private NodeHistogram rentHistogram() {
        NodeHistogram histogram = histogramPool.poll();
        if (histogram == null) {
            return new NodeHistogram(new double[histogramSize], new double[histogramSize],
                    new int[histogramSize]);
        }
        return histogram;
    }

    /**
     * Fills {@code target} with the histogram of {@code samples}. Clearing only touches active
     * feature ranges because the gain scan never reads inactive features.
     */
    private void accumulateHistogram(NodeHistogram target, int[] samples, int[] activeFeatures) {
        for (int feature : activeFeatures) {
            int base = feature * workspace.maxBins;
            int bins = binned.binCount(feature);
            Arrays.fill(target.gradient, base, base + bins, 0.0);
            Arrays.fill(target.hessian, base, base + bins, 0.0);
            Arrays.fill(target.count, base, base + bins, 0);
        }
        long work = (long) samples.length * activeFeatures.length;
        if (histogramSession != 0L && work >= 50000L) {
            long sessionStart = System.nanoTime();
            boolean sessionDone = NativeKernels.histogramSessionAccumulateChecked(
                    histogramSession, samples, activeFeatures, threads, target.gradient,
                    target.hessian, target.count);
            nativeHistogramNanos += System.nanoTime() - sessionStart;
            if (sessionDone) {
                nativeHistogramCalls++;
                return;
            }
        }
        if (NativeKernels.available() && work >= NATIVE_HISTOGRAM_WORK) {
            long nativeStart = System.nanoTime();
            boolean nativeDone = NativeKernels.histogramAccumulateChecked(gradients, hessians,
                    binned.bins(), samples, activeFeatures, featureCount, workspace.maxBins,
                    threads, target.gradient, target.hessian, target.count);
            nativeHistogramNanos += System.nanoTime() - nativeStart;
            if (nativeDone) {
                nativeHistogramCalls++;
                return;
            }
        }
        long javaStart = System.nanoTime();
        accumulateHistogramJava(target, samples, activeFeatures);
        javaHistogramNanos += System.nanoTime() - javaStart;
    }

    private void accumulateHistogramJava(NodeHistogram target, int[] samples,
                                         int[] activeFeatures) {
        int chunks = chunkCount(samples.length, activeFeatures.length);
        if (chunks == 1) {
            accumulateInto(target.gradient, target.hessian, target.count,
                    samples, 0, samples.length, activeFeatures);
            return;
        }
        for (int chunk = 0; chunk < chunks; chunk++) {
            workspace.clear(chunk, activeFeatures, binned);
        }
        int finalChunks = chunks;
        pool.submit(() -> IntStream.range(0, finalChunks).parallel()
                .forEach(chunk -> accumulateInto(workspace.gradient[chunk],
                        workspace.hessian[chunk], workspace.count[chunk], samples,
                        chunk * samples.length / finalChunks,
                        (chunk + 1) * samples.length / finalChunks, activeFeatures))).join();
        workspace.reduce(chunks, activeFeatures, binned);
        for (int feature : activeFeatures) {
            int base = feature * workspace.maxBins;
            int bins = binned.binCount(feature);
            System.arraycopy(workspace.gradient[0], base, target.gradient, base, bins);
            System.arraycopy(workspace.hessian[0], base, target.hessian, base, bins);
            System.arraycopy(workspace.count[0], base, target.count, base, bins);
        }
    }

    /** Derives the larger child's histogram as {@code parent - small} over active features. */
    private void subtractHistogram(NodeHistogram parent, NodeHistogram small, NodeHistogram target,
                                   int[] activeFeatures) {
        for (int feature : activeFeatures) {
            int base = feature * workspace.maxBins;
            int bins = binned.binCount(feature);
            for (int index = base; index < base + bins; index++) {
                target.gradient[index] = parent.gradient[index] - small.gradient[index];
                target.hessian[index] = parent.hessian[index] - small.hessian[index];
                target.count[index] = parent.count[index] - small.count[index];
            }
        }
    }

    private void accumulateInto(double[] targetGradient, double[] targetHessian,
                                int[] targetCount, int[] samples, int from, int to,
                                int[] activeFeatures) {
        byte[] bins = binned.bins();
        int maxBins = workspace.maxBins;
        for (int index = from; index < to; index++) {
            int sample = samples[index];
            int rowBase = sample * featureCount;
            double gradient = gradients[sample];
            double hessian = hessians[sample];
            for (int feature : activeFeatures) {
                int bucket = feature * maxBins + Byte.toUnsignedInt(bins[rowBase + feature]);
                targetGradient[bucket] += gradient;
                targetHessian[bucket] += hessian;
                targetCount[bucket]++;
            }
        }
    }

    private SplitDecision bestSplit(int[] activeFeatures, NodeHistogram histogram, int samples) {
        int maxBins = workspace.maxBins;
        int firstFeature = activeFeatures[0];
        int firstBins = binned.binCount(firstFeature);
        int firstBase = firstFeature * maxBins;
        double totalGradient = 0.0;
        double totalHessian = 0.0;
        for (int bin = 0; bin < firstBins; bin++) {
            totalGradient += histogram.gradient[firstBase + bin];
            totalHessian += histogram.hessian[firstBase + bin];
        }
        SplitDecision best = null;
        for (int feature : activeFeatures) {
            int bins = binned.binCount(feature);
            int base = feature * maxBins;
            double leftGradient = 0.0;
            double leftHessian = 0.0;
            int leftCount = 0;
            for (int bin = 0; bin < bins - 1; bin++) {
                leftGradient += histogram.gradient[base + bin];
                leftHessian += histogram.hessian[base + bin];
                leftCount += histogram.count[base + bin];
                int rightCount = samples - leftCount;
                double rightHessian = totalHessian - leftHessian;
                if (leftCount < minSamplesLeaf || rightCount < minSamplesLeaf
                        || leftHessian < minChildWeight || rightHessian < minChildWeight) {
                    continue;
                }
                double rightGradient = totalGradient - leftGradient;
                double gain = 0.5 * (score(leftGradient, leftHessian)
                        + score(rightGradient, rightHessian) - score(totalGradient, totalHessian))
                        - gamma;
                if (best == null || gain > best.gain) {
                    best = new SplitDecision(feature, binned.threshold(feature, bin), bin, gain);
                }
            }
        }
        return best;
    }

    private int chunkCount(int samples, int features) {
        long work = (long) samples * features;
        if (work < PARALLEL_WORK_THRESHOLD) {
            return 1;
        }
        int maximum = Math.min(threads, Math.max(1, samples / MIN_CHUNK_SAMPLES));
        return Math.max(1, maximum);
    }

    private double score(double gradient, double hessian) {
        return gradient * gradient / (hessian + lambda);
    }

    private double leafWeight(int[] samples) {
        double gradient = 0.0;
        double hessian = 0.0;
        for (int sample : samples) {
            gradient += gradients[sample];
            hessian += hessians[sample];
        }
        return -gradient / (hessian + lambda);
    }

    // ---------------------------------------------------------------- inference

    public double[] predictRaw(Matrix features) {
        MlChecks.predictionData(features, featureCount, !trees.isEmpty());
        int rows = features.rows();
        double[] result = new double[rows];
        Arrays.fill(result, baseScore);
        double[] data = features.data();
        int columns = features.columns();
        if (NativeKernels.available() && packed != null
                && (long) rows * trees.size() >= 20_000_000L
                && NativeKernels.forestPredictChecked(data, rows, columns,
                packed.feature, packed.threshold, packed.left, packed.right,
                packed.value, packed.offsets, result, learningRate)) {
            return result;
        }
        for (TreeModel tree : trees) {
            tree.addInto(data, rows, columns, result, learningRate);
        }
        return result;
    }

    public double[] predictProbability(Matrix features) {
        if (objective != BoostingObjective.LOGISTIC) {
            throw new IllegalStateException("Probabilities require LOGISTIC objective");
        }
        double[] raw = predictRaw(features);
        for (int i = 0; i < raw.length; i++) {
            raw[i] = sigmoid(raw[i]);
        }
        return raw;
    }

    @Override
    public double[] predict(Matrix features) {
        double[] raw = predictRaw(features);
        if (objective == BoostingObjective.LOGISTIC) {
            for (int i = 0; i < raw.length; i++) {
                raw[i] = sigmoid(raw[i]) >= 0.5 ? 1.0 : 0.0;
            }
        }
        return raw;
    }

    public double[] lossHistory() {
        return lossHistory.clone();
    }

    public double[] validationHistory() {
        return validationHistory.clone();
    }

    public double[] featureImportances() {
        return importances.clone();
    }

    public int treeCount() {
        return trees.size();
    }

    public int bestIteration() {
        return bestIteration;
    }

    private static double sigmoid(double value) {
        if (value >= 0.0) {
            return 1.0 / (1.0 + Math.exp(-value));
        }
        double exponential = Math.exp(value);
        return exponential / (1.0 + exponential);
    }

    // ---------------------------------------------------------------- persistence

    @Override
    public ModelKind kind() {
        return ModelKind.HISTOGRAM_BOOSTING;
    }

    @Override
    public void write(ModelWriter writer) {
        writer.putInt(algorithm.ordinal());
        writer.putInt(objective.ordinal());
        writer.putDouble(learningRate);
        writer.putDouble(baseScore);
        writer.putInt(featureCount);
        writer.putInt(trees.size());
        for (TreeModel tree : trees) {
            tree.write(writer);
        }
    }

    public static HistogramGradientBoosting read(ModelReader reader) {
        int algorithmOrdinal = reader.getInt();
        int objectiveOrdinal = reader.getInt();
        if (algorithmOrdinal < 0 || algorithmOrdinal >= BoostingAlgorithm.values().length
                || objectiveOrdinal < 0 || objectiveOrdinal >= BoostingObjective.values().length) {
            throw new ModelFormatException("Invalid boosting enum ordinal");
        }
        double learningRate = reader.getDouble();
        double baseScore = reader.getDouble();
        int features = reader.getInt();
        int count = reader.getInt();
        if (count <= 0 || count > 100_000 || features <= 0) {
            throw new ModelFormatException("Invalid boosting header");
        }
        HistogramGradientBoosting model = new HistogramGradientBoosting(new Builder()
                .algorithm(BoostingAlgorithm.values()[algorithmOrdinal])
                .objective(BoostingObjective.values()[objectiveOrdinal])
                .learningRate(learningRate));
        model.featureCount = features;
        model.baseScore = baseScore;
        for (int i = 0; i < count; i++) {
            model.trees.add(TreeModel.read(reader));
        }
        model.packed = PackedTrees.pack(model.trees);
        model.lossHistory = new double[0];
        model.validationHistory = new double[0];
        model.importances = new double[features];
        model.bestIteration = count - 1;
        return model;
    }

    // ---------------------------------------------------------------- internals

    private static final class NodeHistogram {
        private final double[] gradient;
        private final double[] hessian;
        private final int[] count;

        private NodeHistogram(double[] gradient, double[] hessian, int[] count) {
            this.gradient = gradient;
            this.hessian = hessian;
            this.count = count;
        }
    }

    private static final class HistogramWorkspace {
        private final double[][] gradient;
        private final double[][] hessian;
        private final int[][] count;
        private final int maxBins;

        private HistogramWorkspace(int features, int maxBins, int chunks) {
            this.maxBins = maxBins;
            gradient = new double[chunks][features * maxBins];
            hessian = new double[chunks][features * maxBins];
            count = new int[chunks][features * maxBins];
        }

        private void clear(int chunk, int[] activeFeatures, BinnedFeatures binned) {
            for (int feature : activeFeatures) {
                int base = feature * maxBins;
                int bins = binned.binCount(feature);
                Arrays.fill(gradient[chunk], base, base + bins, 0.0);
                Arrays.fill(hessian[chunk], base, base + bins, 0.0);
                Arrays.fill(count[chunk], base, base + bins, 0);
            }
        }

        private void reduce(int chunks, int[] activeFeatures, BinnedFeatures binned) {
            for (int feature : activeFeatures) {
                int base = feature * maxBins;
                int bins = binned.binCount(feature);
                for (int chunk = 1; chunk < chunks; chunk++) {
                    for (int index = base; index < base + bins; index++) {
                        gradient[0][index] += gradient[chunk][index];
                        hessian[0][index] += hessian[chunk][index];
                        count[0][index] += count[chunk][index];
                    }
                }
            }
        }
    }

    private record SplitDecision(int feature, double threshold, int bin, double gain) {
    }

    private record LeafCandidate(int node, int[] samples, int depth, SplitDecision decision,
                                 NodeHistogram histogram) {
    }

    public static final class Builder {
        private BoostingAlgorithm algorithm = BoostingAlgorithm.XGBOOST;
        private BoostingObjective objective = BoostingObjective.SQUARED_ERROR;
        private int iterations = 100;
        private double learningRate = 0.1;
        private int maxDepth = 6;
        private int maxLeaves = 31;
        private int binCount = 64;
        private int minSamplesLeaf = 1;
        private double minChildWeight = 1e-12;
        private double lambda = 1.0;
        private double gamma;
        private int threads = Runtime.getRuntime().availableProcessors();
        private double featureFraction = 1.0;
        private double validationFraction;
        private int earlyStoppingRounds;
        private long seed = 42;

        public Builder algorithm(BoostingAlgorithm algorithm) {
            this.algorithm = java.util.Objects.requireNonNull(algorithm, "algorithm");
            return this;
        }

        public Builder objective(BoostingObjective objective) {
            this.objective = java.util.Objects.requireNonNull(objective, "objective");
            return this;
        }

        public Builder iterations(int iterations) {
            this.iterations = iterations;
            return this;
        }

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder maxDepth(int maxDepth) {
            this.maxDepth = maxDepth;
            return this;
        }

        public Builder maxLeaves(int maxLeaves) {
            this.maxLeaves = maxLeaves;
            return this;
        }

        public Builder bins(int binCount) {
            this.binCount = binCount;
            return this;
        }

        public Builder minSamplesLeaf(int minSamplesLeaf) {
            this.minSamplesLeaf = minSamplesLeaf;
            return this;
        }

        public Builder minChildWeight(double minChildWeight) {
            this.minChildWeight = minChildWeight;
            return this;
        }

        public Builder lambda(double lambda) {
            this.lambda = lambda;
            return this;
        }

        public Builder gamma(double gamma) {
            this.gamma = gamma;
            return this;
        }

        public Builder threads(int threads) {
            this.threads = threads;
            return this;
        }

        public Builder featureFraction(double featureFraction) {
            this.featureFraction = featureFraction;
            return this;
        }

        public Builder validationFraction(double validationFraction) {
            this.validationFraction = validationFraction;
            return this;
        }

        public Builder earlyStoppingRounds(int earlyStoppingRounds) {
            this.earlyStoppingRounds = earlyStoppingRounds;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public HistogramGradientBoosting build() {
            if (iterations <= 0 || maxDepth <= 0 || maxLeaves < 2
                    || binCount < 2 || binCount > BinnedFeatures.MAX_BINS
                    || minSamplesLeaf <= 0 || threads <= 0) {
                throw new IllegalArgumentException("Invalid histogram boosting configuration");
            }
            if (!(featureFraction > 0.0) || featureFraction > 1.0) {
                throw new IllegalArgumentException("featureFraction must be in (0, 1]");
            }
            if (!(validationFraction >= 0.0) || validationFraction >= 0.5) {
                throw new IllegalArgumentException("validationFraction must be in [0, 0.5)");
            }
            if (earlyStoppingRounds < 0) {
                throw new IllegalArgumentException("earlyStoppingRounds must be non-negative");
            }
            if (earlyStoppingRounds > 0 && validationFraction <= 0.0) {
                throw new IllegalArgumentException(
                        "Early stopping requires a positive validationFraction");
            }
            MlChecks.positiveFinite(learningRate, "learningRate");
            MlChecks.nonNegativeFinite(minChildWeight, "minChildWeight");
            MlChecks.nonNegativeFinite(lambda, "lambda");
            MlChecks.nonNegativeFinite(gamma, "gamma");
            return new HistogramGradientBoosting(this);
        }
    }
}
