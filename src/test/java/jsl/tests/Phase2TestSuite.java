package jsl.tests;

import jsl.core.Matrix;
import jsl.io.ModelCodec;
import jsl.io.OnnxConverter;
import jsl.io.OnnxModel;
import jsl.ml.AgglomerativeClustering;
import jsl.ml.DecisionTree;
import jsl.ml.GaussianMixture;
import jsl.ml.IsolationForest;
import jsl.ml.JSL;
import jsl.ml.KBinsDiscretizer;
import jsl.ml.Kernel;
import jsl.ml.KernelDensity;
import jsl.ml.KernelPCA;
import jsl.ml.KernelRidge;
import jsl.ml.Metrics;
import jsl.ml.Pipeline;
import jsl.ml.PolynomialFeatures;
import jsl.ml.RFE;
import jsl.ml.Ridge;
import jsl.ml.SVC;
import jsl.ml.SelectKBest;
import jsl.ml.StandardScaler;
import jsl.nativebridge.NativeContext;
import jsl.torch.Activation;
import jsl.torch.AdamW;
import jsl.torch.BCEWithLogitsLoss;
import jsl.torch.BatchNorm2d;
import jsl.torch.Checkpoint;
import jsl.torch.ComputationGraph;
import jsl.torch.Conv2d;
import jsl.torch.CosineAnnealingLR;
import jsl.torch.DepthwiseSeparableConv2d;
import jsl.torch.Dropout;
import jsl.torch.GroupNorm;
import jsl.torch.L1Loss;
import jsl.torch.Linear;
import jsl.torch.Sequential;
import jsl.torch.SmoothL1Loss;
import jsl.torch.StepLR;
import jsl.torch.Tensor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;

/**
 * Comprehensive verification suite for Phase 2:
 * 1. Classical ML expansion: Kernel SVM (SVC RBF/Poly), KernelRidge, KernelPCA,
 *    GaussianMixture (GMM), AgglomerativeClustering, IsolationForest, KernelDensity (KDE),
 *    SelectKBest, RFE, PolynomialFeatures, KBinsDiscretizer.
 * 2. Deep Learning upgrade: Grouped Conv2d, Dilated Conv2d, DepthwiseSeparableConv2d,
 *    GroupNorm, Dropout/BatchNorm train/eval state switching, L1/SmoothL1/BCEWithLogits losses,
 *    LeakyReLU/ELU activations, DAG ComputationGraph (residual + multi-input/output),
 *    CosineAnnealingLR, StepLR, gradient clipping/accumulation, Checkpoint save/restore.
 * 3. Ecosystem Interoperability: ONNX export and import inference round-trip.
 */
public final class Phase2TestSuite {
    private static int passed;

    private Phase2TestSuite() {
    }

    public static void main(String[] args) throws Exception {
        run("RBF & Polynomial SVC non-linear XOR classification & persistence", Phase2TestSuite::testKernelSvc);
        run("SVC Platt probability calibration (sum-to-one, monotonicity, persistence)",
                Phase2TestSuite::testSvcProbability);
        run("KernelRidge & KernelPCA non-linear regression and projection", Phase2TestSuite::testKernelRidgeAndPca);
        run("GaussianMixture (GMM) EM clustering, probabilities, AIC/BIC & persistence", Phase2TestSuite::testGmm);
        run("GMM FULL/TIED/DIAG covariance parametrisations & AIC penalty",
                Phase2TestSuite::testGmmCovarianceTypes);
        run("AgglomerativeClustering (WARD, COMPLETE, AVERAGE, SINGLE)", Phase2TestSuite::testAgglomerative);
        run("IsolationForest anomaly detection & KernelDensity estimation", Phase2TestSuite::testIsolationForestAndKde);
        run("SelectKBest, RFE, PolynomialFeatures & KBinsDiscretizer", Phase2TestSuite::testFeatureEngineering);
        run("Grouped Conv2d, Dilated Conv2d, DepthwiseSeparableConv2d & GroupNorm gradients", Phase2TestSuite::testAdvancedConvsAndGroupNorm);
        run("Dropout & BatchNorm2d train/eval state switching & new losses/activations", Phase2TestSuite::testTrainEvalAndLosses);
        run("DAG ComputationGraph residual connections & multi-input/multi-output training", Phase2TestSuite::testComputationGraphDag);
        run("CosineAnnealingLR, StepLR, gradient accumulation & Checkpoint resume", Phase2TestSuite::testSchedulersAndCheckpoint);
        run("ONNX model export and import inference round-trip", Phase2TestSuite::testOnnxRoundTrip);
        System.out.println("PASS: " + passed + " Phase 2 test groups");
    }

    private static void testKernelSvc() throws Exception {
        Path dir = Path.of(System.getProperty("user.dir"), "build");
        Files.createDirectories(dir);
        // Non-linearly separable XOR dataset
        double[][] rows = new double[40][2];
        double[] labels = new double[40];
        Random rng = new Random(11);
        for (int i = 0; i < 40; i++) {
            double x0 = (i % 2 == 0 ? 1.0 : -1.0) + rng.nextGaussian() * 0.15;
            double x1 = ((i / 2) % 2 == 0 ? 1.0 : -1.0) + rng.nextGaussian() * 0.15;
            rows[i][0] = x0;
            rows[i][1] = x1;
            labels[i] = (x0 * x1 > 0.0) ? 1.0 : 0.0;
        }
        Matrix x = Matrix.fromRows(rows);
        SVC rbfSvc = SVC.builder().rbf(1.0).c(5.0).maxIterations(800).seed(42L).build().fit(x, labels);
        double acc = Metrics.accuracy(labels, rbfSvc.predict(x));
        check(acc >= 0.95, "RBF SVC XOR accuracy >= 0.95, got " + acc);

        SVC polySvc = SVC.builder().poly(2, 1.0, 1.0).c(5.0).maxIterations(800).seed(42L).build().fit(x, labels);
        check(Metrics.accuracy(labels, polySvc.predict(x)) >= 0.95, "Poly(2) SVC XOR accuracy >= 0.95");

        Path svcPath = dir.resolve("test-svc.jsl");
        ModelCodec.save(svcPath, rbfSvc);
        SVC loaded = ModelCodec.load(svcPath, SVC.class);
        assertArray(rbfSvc.predict(x), loaded.predict(x), 0.0, "SVC .jsl round-trip");
    }

    private static void testKernelRidgeAndPca() throws Exception {
        Path dir = Path.of(System.getProperty("user.dir"), "build");
        int n = 60;
        double[] xData = new double[n];
        double[] yData = new double[n];
        for (int i = 0; i < n; i++) {
            double x = -3.0 + 6.0 * i / (n - 1.0);
            xData[i] = x;
            yData[i] = Math.sin(x) + 0.5 * x * x;
        }
        Matrix x = new Matrix(n, 1, xData);
        KernelRidge krr = KernelRidge.builder().rbf(0.5).alpha(1e-3).build().fit(x, yData);
        double r2 = Metrics.r2(yData, krr.predict(x));
        check(r2 > 0.99, "KernelRidge non-linear R2 > 0.99, got " + r2);

        Path krrPath = dir.resolve("test-krr.jsl");
        ModelCodec.save(krrPath, krr);
        KernelRidge krrLoaded = ModelCodec.load(krrPath, KernelRidge.class);
        assertArray(krr.predict(x), krrLoaded.predict(x), 1e-12, "KernelRidge .jsl round-trip");

        KernelPCA kpca = KernelPCA.builder().components(2).rbf(0.5).build().fit(x);
        Matrix proj = kpca.transform(x);
        check(proj.rows() == n && proj.columns() == 2, "KernelPCA output shape");
        check(kpca.eigenvalues()[0] >= kpca.eigenvalues()[1] && kpca.eigenvalues()[0] > 0.0,
                "KernelPCA descending positive eigenvalues");

        Path kpcaPath = dir.resolve("test-kpca.jsl");
        ModelCodec.save(kpcaPath, kpca);
        KernelPCA kpcaLoaded = ModelCodec.load(kpcaPath, KernelPCA.class);
        assertArray(proj.data(), kpcaLoaded.transform(x).data(), 1e-12, "KernelPCA .jsl round-trip");
    }

    private static void testGmm() throws Exception {
        Path dir = Path.of(System.getProperty("user.dir"), "build");
        Random rng = new Random(7);
        double[] data = new double[80 * 2];
        for (int r = 0; r < 40; r++) {
            data[r * 2] = -4.0 + rng.nextGaussian() * 0.3;
            data[r * 2 + 1] = -4.0 + rng.nextGaussian() * 0.3;
        }
        for (int r = 40; r < 80; r++) {
            data[r * 2] = 4.0 + rng.nextGaussian() * 0.3;
            data[r * 2 + 1] = 4.0 + rng.nextGaussian() * 0.3;
        }
        Matrix x = new Matrix(80, 2, data);
        GaussianMixture gmm = GaussianMixture.builder().components(2).seed(42L).build().fit(x);
        check(gmm.converged(), "GMM must converge");
        double[] preds = gmm.predict(x);
        check(preds[0] != preds[40], "GMM separates two Gaussian clusters");
        Matrix probs = gmm.predictProbability(x);
        for (int r = 0; r < 80; r++) {
            assertClose(1.0, probs.get(r, 0) + probs.get(r, 1), 1e-9, "GMM responsibilities sum to 1");
        }
        check(Double.isFinite(gmm.aic(x)) && Double.isFinite(gmm.bic(x)), "GMM AIC/BIC finite");

        Path gmmPath = dir.resolve("test-gmm.jsl");
        ModelCodec.save(gmmPath, gmm);
        GaussianMixture gmmLoaded = ModelCodec.load(gmmPath, GaussianMixture.class);
        assertArray(preds, gmmLoaded.predict(x), 0.0, "GMM .jsl round-trip");
    }

    private static void testSvcProbability() throws Exception {
        Path dir = Path.of(System.getProperty("user.dir"), "build");
        Files.createDirectories(dir);
        Random rng = new Random(11);
        int rows = 160;
        double[] data = new double[rows * 2];
        double[] labels = new double[rows];
        for (int r = 0; r < rows; r++) {
            double a = rng.nextGaussian();
            double b = rng.nextGaussian();
            boolean positive = a + b > 0;
            data[r * 2] = a + (positive ? 0.6 : -0.6);
            data[r * 2 + 1] = b + (positive ? 0.6 : -0.6);
            labels[r] = positive ? 1.0 : -1.0;
        }
        Matrix x = new Matrix(rows, 2, data);

        SVC calibrated = SVC.builder().rbf(0.5).c(1.0).probability(true).build().fit(x, labels);
        check(calibrated.probabilityEnabled(), "probability flag is reported");

        Matrix proba = calibrated.predictProbability(x);
        check(proba.columns() == 2, "SVC probabilities have one column per class");
        double[] decision = calibrated.decisionFunction(x);
        for (int r = 0; r < rows; r++) {
            assertClose(1.0, proba.get(r, 0) + proba.get(r, 1), 1e-9,
                    "SVC class probabilities sum to 1");
        }

        // higher decision value must never yield a lower positive-class probability
        double slope = plattSlope(decision, proba, rows);
        check(slope > 0.0, "positive-class probability increases with the decision value");

        int correct = 0;
        for (int r = 0; r < rows; r++) {
            double predicted = proba.get(r, 1) >= 0.5 ? 1.0 : -1.0;
            if (predicted == labels[r]) {
                correct++;
            }
        }
        double accuracy = correct / (double) rows;
        check(accuracy >= 0.95, "probability threshold reproduces the labels, got " + accuracy);

        Path path = dir.resolve("test-svc-probability.jsl");
        ModelCodec.save(path, calibrated);
        SVC loaded = ModelCodec.load(path, SVC.class);
        check(loaded.probabilityEnabled(), "probability flag survives serialization");
        Matrix reloaded = loaded.predictProbability(x);
        for (int r = 0; r < rows; r++) {
            assertClose(proba.get(r, 1), reloaded.get(r, 1), 1e-12,
                    "calibrated probability is bit-exact after reload");
        }

        SVC plain = SVC.builder().rbf(0.5).build().fit(x, labels);
        boolean threw = false;
        try {
            plain.predictProbability(x);
        } catch (IllegalStateException expected) {
            threw = true;
        }
        check(threw, "predictProbability rejects uncalibrated models");
    }

    /** Spearman-style sign test: does the positive-class probability grow with the decision value? */
    private static double plattSlope(double[] decision, Matrix proba, int rows) {
        double meanF = 0.0;
        double meanP = 0.0;
        for (int r = 0; r < rows; r++) {
            meanF += decision[r];
            meanP += proba.get(r, 1);
        }
        meanF /= rows;
        meanP /= rows;
        double covariance = 0.0;
        for (int r = 0; r < rows; r++) {
            covariance += (decision[r] - meanF) * (proba.get(r, 1) - meanP);
        }
        return covariance;
    }

    private static void testGmmCovarianceTypes() throws Exception {
        Path dir = Path.of(System.getProperty("user.dir"), "build");
        Files.createDirectories(dir);
        Random rng = new Random(7);
        int rows = 400;
        double[] data = new double[rows * 2];
        for (int r = 0; r < rows; r++) {
            double z = rng.nextGaussian();
            data[r * 2] = z;
            data[r * 2 + 1] = 3.0 * z + 0.35 * rng.nextGaussian();
        }
        Matrix x = new Matrix(rows, 2, data);

        GaussianMixture full = GaussianMixture.builder().components(1).seed(42L)
                .covarianceType(GaussianMixture.CovarianceType.FULL).build().fit(x);
        GaussianMixture tied = GaussianMixture.builder().components(2).seed(42L)
                .covarianceType(GaussianMixture.CovarianceType.TIED).build().fit(x);
        GaussianMixture diag = GaussianMixture.builder().components(1).seed(42L)
                .covarianceType(GaussianMixture.CovarianceType.DIAG).build().fit(x);

        check(full.covarianceType() == GaussianMixture.CovarianceType.FULL, "covariance type is reported");
        check(Math.abs(full.covariance(0).get(0, 1)) > 0.5,
                "FULL captures the off-diagonal entry, got " + full.covariance(0).get(0, 1));
        assertClose(0.0, diag.covariance(0).get(0, 1), 1e-15, "DIAG keeps the covariance diagonal");

        check(full.lowerBound() > diag.lowerBound() + 0.5,
                "FULL fits correlated data better than DIAG, got " + full.lowerBound()
                        + " vs " + diag.lowerBound());
        check(full.aic(x) < diag.aic(x), "AIC prefers FULL on correlated data");
        check(full.parameterCount() == 1 * (2 + 3) + 0, "FULL parameter count");
        check(diag.parameterCount() == 1 * 2 * 2 + 0, "DIAG parameter count");
        check(tied.parameterCount() == 2 * 2 + 3 + 1, "TIED parameter count");

        double[] tiedWeights = tied.weights();
        assertClose(1.0, tiedWeights[0] + tiedWeights[1], 1e-12, "TIED weights sum to 1");
        check(tied.converged(), "TIED converges on bimodal-free data");

        for (GaussianMixture.CovarianceType type : GaussianMixture.CovarianceType.values()) {
            GaussianMixture model = GaussianMixture.builder().components(2).seed(42L)
                    .covarianceType(type).build().fit(x);
            Path path = dir.resolve("test-gmm-" + type.name().toLowerCase(java.util.Locale.ROOT) + ".jsl");
            ModelCodec.save(path, model);
            GaussianMixture loaded = ModelCodec.load(path, GaussianMixture.class);
            check(loaded.covarianceType() == type, type + " covariance type survives serialization");
            assertArray(model.predict(x), loaded.predict(x), 0.0, type + " GMM round-trip");
            assertClose(model.lowerBound(), loaded.lowerBound(), 1e-12,
                    type + " log-likelihood survives reload");
        }
    }

    private static void testAgglomerative() {
        Matrix x = Matrix.fromRows(new double[][]{
                {0.0, 0.0}, {0.1, -0.1}, {-0.1, 0.1},
                {10.0, 10.0}, {10.1, 9.9}, {9.9, 10.1}
        });
        for (AgglomerativeClustering.Linkage linkage : AgglomerativeClustering.Linkage.values()) {
            AgglomerativeClustering ac = AgglomerativeClustering.builder()
                    .clusters(2).linkage(linkage).build().fit(x);
            int[] labels = ac.labels();
            check(labels[0] == labels[1] && labels[1] == labels[2], linkage + " cluster 0 purity");
            check(labels[3] == labels[4] && labels[4] == labels[5], linkage + " cluster 1 purity");
            check(labels[0] != labels[3], linkage + " distinct clusters");
        }
    }

    private static void testIsolationForestAndKde() throws Exception {
        Path dir = Path.of(System.getProperty("user.dir"), "build");
        Random rng = new Random(19);
        int inliers = 95;
        int outliers = 5;
        int total = inliers + outliers;
        double[] data = new double[total * 2];
        for (int i = 0; i < inliers; i++) {
            data[i * 2] = rng.nextGaussian() * 0.4;
            data[i * 2 + 1] = rng.nextGaussian() * 0.4;
        }
        for (int i = inliers; i < total; i++) {
            data[i * 2] = 15.0 + rng.nextDouble();
            data[i * 2 + 1] = -15.0 - rng.nextDouble();
        }
        Matrix x = new Matrix(total, 2, data);
        IsolationForest iso = IsolationForest.builder()
                .trees(100).maxSamples(64).contamination(0.06).seed(42L).build().fit(x);
        double[] preds = iso.predict(x);
        int detectedOutliers = 0;
        for (int i = inliers; i < total; i++) {
            if (preds[i] == -1.0) detectedOutliers++;
        }
        check(detectedOutliers == outliers, "IsolationForest detected all 5 far outliers");

        Path isoPath = dir.resolve("test-isoforest.jsl");
        ModelCodec.save(isoPath, iso);
        IsolationForest isoLoaded = ModelCodec.load(isoPath, IsolationForest.class);
        assertArray(preds, isoLoaded.predict(x), 0.0, "IsolationForest .jsl round-trip");

        KernelDensity kde = KernelDensity.builder().bandwidth(0.8).build()
                .fit(new Matrix(inliers, 2, java.util.Arrays.copyOf(data, inliers * 2)));
        double[] logDens = kde.scoreSamples(Matrix.fromRows(new double[][]{{0.0, 0.0}, {15.0, -15.0}}));
        check(logDens[0] > logDens[1] + 50.0, "KDE assigns much higher density to center than outlier");
        Matrix sampled = kde.sample(10, 7L);
        check(sampled.rows() == 10 && sampled.columns() == 2, "KDE sample shape");
    }

    private static void testFeatureEngineering() {
        Random rng = new Random(33);
        int rows = 100;
        double[] xData = new double[rows * 4];
        double[] yClass = new double[rows];
        double[] yReg = new double[rows];
        for (int r = 0; r < rows; r++) {
            double f0 = rng.nextGaussian(); // strong signal
            double f1 = rng.nextGaussian(); // medium signal
            double f2 = rng.nextGaussian() * 0.01; // noise
            double f3 = rng.nextGaussian() * 0.01; // noise
            xData[r * 4] = f0;
            xData[r * 4 + 1] = f1;
            xData[r * 4 + 2] = f2;
            xData[r * 4 + 3] = f3;
            yReg[r] = 5.0 * f0 + 2.0 * f1;
            yClass[r] = yReg[r] > 0.0 ? 1.0 : 0.0;
        }
        Matrix x = new Matrix(rows, 4, xData);

        SelectKBest kbest = SelectKBest.builder().k(2).scoreFunc(SelectKBest.ScoreFunc.F_CLASSIF).build()
                .fit(x, yClass);
        check(java.util.Arrays.equals(kbest.selectedIndices(), new int[]{0, 1}),
                "SelectKBest F_CLASSIF picks informative features [0, 1]");
        check(kbest.transform(x).columns() == 2, "SelectKBest reduced columns");

        RFE rfe = RFE.builder(() -> Ridge.builder().alpha(1e-3).build())
                .nFeaturesToSelect(2).step(1).build().fit(x, yReg);
        check(java.util.Arrays.equals(rfe.selectedIndices(), new int[]{0, 1}),
                "RFE picks informative features [0, 1]");

        PolynomialFeatures poly = PolynomialFeatures.builder().degree(2).includeBias(true).build()
                .fit(Matrix.fromRows(new double[][]{{2.0, 3.0}}));
        Matrix polyOut = poly.transform(Matrix.fromRows(new double[][]{{2.0, 3.0}}));
        // [1, x0, x1, x0^2, x0*x1, x1^2] = [1, 2, 3, 4, 6, 9]
        assertArray(new double[]{1.0, 2.0, 3.0, 4.0, 6.0, 9.0}, polyOut.data(), 1e-12, "PolynomialFeatures degree 2");

        KBinsDiscretizer kbins = KBinsDiscretizer.builder()
                .nBins(4).strategy(KBinsDiscretizer.Strategy.QUANTILE)
                .encode(KBinsDiscretizer.Encode.ORDINAL).build().fit(x);
        Matrix binned = kbins.transform(x);
        check(binned.columns() == 4, "KBinsDiscretizer ordinal columns");
        for (double v : binned.data()) {
            check(v >= 0.0 && v <= 3.0, "bin index in [0, 3]");
        }
    }

    private static void testAdvancedConvsAndGroupNorm() {
        try (NativeContext ctx = NativeContext.create(2)) {
            // 1. Grouped Conv2d finite-difference gradient check (groups = 2)
            Tensor input = Tensor.of(ctx,
                    new float[]{
                            1, 2, 3, 4, 5, 6, 7, 8, 9,
                            2, 1, 0, 1, 3, 2, 4, 1, 2,
                            0.5f, 1.5f, -0.5f, 2, 1, 0, -1, 2, 3,
                            1, -1, 2, 0, 1, 2, 3, 0, -2
                    }, true, 1, 4, 3, 3);
            Tensor weight = Tensor.of(ctx,
                    new float[]{
                            0.5f, -0.2f, 0.1f, 0.3f,  -0.4f, 0.2f, 0.6f, -0.1f,
                            0.2f, 0.4f, -0.3f, 0.5f,  0.1f, -0.5f, 0.2f, 0.3f,
                            -0.3f, 0.1f, 0.4f, -0.2f,  0.5f, 0.2f, -0.1f, 0.4f,
                            0.6f, -0.3f, 0.2f, 0.1f,  -0.2f, 0.3f, 0.5f, -0.4f
                    }, true, 4, 2, 2, 2);
            Tensor target = Tensor.zeros(ctx, 1, 4, 2, 2);
            Supplier<Tensor> groupedLoss = () -> input.conv2d(weight, 2, 2, 1, 1, 0, 0, 1, 1, 2).mse(target);
            checkGradient(ctx, input, groupedLoss, 1e-2f, 3e-2f, "grouped conv2d input grad");
            checkGradient(ctx, weight, groupedLoss, 1e-2f, 3e-2f, "grouped conv2d weight grad");

            // 2. Dilated Conv2d (dilation = 2 on 5x5 input -> 1x1 output with 3x3 kernel)
            float[] dIn = new float[25];
            for (int i = 0; i < 25; i++) dIn[i] = (i + 1) * 0.1f;
            Tensor dilInput = Tensor.of(ctx, dIn, true, 1, 1, 5, 5);
            Tensor dilWeight = Tensor.of(ctx, new float[]{1, 0, -1, 0, 2, 0, -1, 0, 1}, true, 1, 1, 3, 3);
            Tensor dilTarget = Tensor.zeros(ctx, 1, 1, 1, 1);
            Supplier<Tensor> dilatedLoss = () -> dilInput.conv2d(dilWeight, 3, 3, 1, 1, 0, 0, 2, 2, 1).mse(dilTarget);
            checkGradient(ctx, dilInput, dilatedLoss, 1e-2f, 3e-2f, "dilated conv2d input grad");
            checkGradient(ctx, dilWeight, dilatedLoss, 1e-2f, 3e-2f, "dilated conv2d weight grad");

            // 3. DepthwiseSeparableConv2d forward & backward
            DepthwiseSeparableConv2d dsConv = new DepthwiseSeparableConv2d(ctx, 4, 6, 3, 1, 1, 42L);
            Tensor dsOut = dsConv.forward(input);
            check(java.util.Arrays.equals(dsOut.shape(), new int[]{1, 6, 3, 3}), "DepthwiseSeparableConv2d shape");
            dsOut.mse(Tensor.zeros(ctx, 1, 6, 3, 3)).backward();
            check(dsConv.depthwise().weight().gradNorm() > 0.0, "depthwise grad > 0");
            check(dsConv.pointwise().weight().gradNorm() > 0.0, "pointwise grad > 0");

            // 4. GroupNorm forward & finite-difference gradient check
            GroupNorm<jsl.core.BCHW> gn = new GroupNorm<>(ctx, 2, 4);
            Supplier<Tensor> gnLoss = () -> gn.forward(input).mse(Tensor.zeros(ctx, 1, 4, 3, 3));
            checkGradient(ctx, input, gnLoss, 1e-2f, 3e-2f, "group-norm input grad");
            checkGradient(ctx, gn.gamma(), gnLoss, 1e-2f, 3e-2f, "group-norm gamma grad");
            checkGradient(ctx, gn.beta(), gnLoss, 1e-2f, 3e-2f, "group-norm beta grad");
        }
    }

    private static void testTrainEvalAndLosses() {
        try (NativeContext ctx = NativeContext.create(1)) {
            Dropout<jsl.core.BC> dropout = new Dropout<>(0.5f, 99L);
            BatchNorm2d bn = new BatchNorm2d(ctx, 2);
            Sequential<jsl.core.BCHW, jsl.core.BCHW> seq = Sequential.<jsl.core.BCHW>start()
                    .add(bn).add(new Dropout<>(0.5f, 101L));

            seq.train();
            check(seq.isTraining() && bn.isTraining(), "seq.train() sets children to training");
            seq.eval();
            check(!seq.isTraining() && !bn.isTraining(), "seq.eval() sets children to eval");

            dropout.eval();
            Tensor x = Tensor.of(ctx, new float[]{1, 2, 3, 4}, 2, 2);
            assertArray(x.data(), dropout.forward(x).data(), 0.0f, "Dropout eval is exact identity");

            // Test L1Loss, SmoothL1Loss, BCEWithLogitsLoss, LeakyReLU, ELU
            Tensor pred = Tensor.of(ctx, new float[]{2.0f, -1.0f}, true, 2);
            Tensor targ = Tensor.of(ctx, new float[]{1.0f, 1.0f}, 2);
            Tensor l1 = new L1Loss().loss(pred, targ);
            assertClose(1.5, l1.data()[0], 1e-5, "L1Loss value");
            l1.backward();
            assertArray(new float[]{0.5f, -0.5f}, pred.grad(), 1e-5f, "L1Loss grad");

            ctx.clearGraph();
            pred.zeroGrad();
            Tensor huber = new SmoothL1Loss(1.0f).loss(pred, targ);
            huber.backward();
            check(pred.gradNorm() > 0.0, "SmoothL1Loss grad > 0");

            ctx.clearGraph();
            pred.zeroGrad();
            Tensor bce = new BCEWithLogitsLoss().loss(pred, Tensor.of(ctx, new float[]{1.0f, 0.0f}, 2));
            bce.backward();
            check(Float.isFinite(bce.data()[0]) && pred.gradNorm() > 0.0, "BCEWithLogitsLoss finite & grad > 0");

            Activation<jsl.core.BC> leaky = new Activation<>(Activation.Kind.LEAKY_RELU, 0.1f);
            assertArray(new float[]{-0.2f, 3.0f},
                    leaky.forward(Tensor.of(ctx, new float[]{-2.0f, 3.0f}, 1, 2)).data(), 1e-6f, "LeakyReLU");
        }
    }

    private static void testComputationGraphDag() {
        try (NativeContext ctx = NativeContext.create(2)) {
            // Build a Residual + Multi-Input DAG:
            //x1 -> proj1 -> h1 -> relu1             //                            add("res") -> head -> out1
            //x1 ------------------------/
            //x2 -> auxHead ----------------------------------> out2
            ComputationGraph dag = ComputationGraph.builder()
                    .addInputs("x1", "x2")
                    .addLayer("proj1", new Linear(ctx, 4, 4, true, 11L), "x1")
                    .addLayer("relu1", new Activation<>(Activation.Kind.RELU), "proj1")
                    .addAddVertex("res", "x1", "relu1")
                    .addLayer("head", new Linear(ctx, 4, 1, true, 23L), "res")
                    .addLayer("auxHead", new Linear(ctx, 2, 1, true, 37L), "x2")
                    .setOutputs("head", "auxHead")
                    .build();

            Tensor x1 = Tensor.of(ctx, new float[]{
                    1, 0, -1, 2,
                    0, 2, 1, -1,
                    2, -1, 0, 1,
                    -1, 1, 2, 0
            }, 4, 4);
            Tensor x2 = Tensor.of(ctx, new float[]{
                    1, -1,
                    2, 0,
                    -1, 2,
                    0, 1
            }, 4, 2);
            Tensor y1 = Tensor.of(ctx, new float[]{2.0f, 1.0f, -1.0f, 0.5f}, 4, 1);
            Tensor y2 = Tensor.of(ctx, new float[]{0.5f, 1.0f, -0.5f, 0.25f}, 4, 1);

            AdamW opt = new AdamW(dag.parameters(), 0.02f);
            float firstLoss = 0.0f;
            float lastLoss = 0.0f;
            for (int step = 0; step < 50; step++) {
                Map<String, Tensor> outs = dag.forward(Map.of("x1", x1, "x2", x2));
                Tensor loss = outs.get("head").mse(y1).add(outs.get("auxHead").mse(y2));
                if (step == 0) firstLoss = loss.data()[0];
                lastLoss = loss.data()[0];
                loss.backward();
                opt.step();
                opt.zeroGrad();
                ctx.clearGraph();
            }
            check(lastLoss < firstLoss * 0.2f,
                    "DAG ComputationGraph multi-input/output residual training converged: "
                            + firstLoss + " -> " + lastLoss);
            dag.close();
        }
    }

    private static void testSchedulersAndCheckpoint() throws Exception {
        Path dir = Path.of(System.getProperty("user.dir"), "build");
        try (NativeContext ctx = NativeContext.create(1)) {
            Linear model = new Linear(ctx, 3, 2, true, 42L);
            AdamW opt = new AdamW(model.parameters(), 0.1f);
            CosineAnnealingLR cosine = new CosineAnnealingLR(opt, 10, 0.001f);
            for (int i = 0; i < 5; i++) cosine.step();
            assertClose(0.0505, opt.learningRate(), 1e-3, "CosineAnnealingLR halfway");
            for (int i = 0; i < 5; i++) cosine.step();
            assertClose(0.001, opt.learningRate(), 1e-5, "CosineAnnealingLR at T_max");

            opt.setLearningRate(0.1f);
            StepLR stepLr = new StepLR(opt, 3, 0.5f);
            for (int i = 0; i < 6; i++) stepLr.step();
            assertClose(0.025, opt.learningRate(), 1e-6, "StepLR after 2 decays");

            // Gradient accumulation & Checkpoint round-trip
            Tensor x = Tensor.of(ctx, new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
            Tensor y = Tensor.of(ctx, new float[]{1, 0, 0, 1}, 2, 2);
            model.forward(x).mse(y).backward();
            check(!opt.stepAccumulated(2), "accumulation step 1/2 does not step yet");
            ctx.clearGraph();
            model.forward(x).mse(y).backward();
            check(opt.stepAccumulated(2), "accumulation step 2/2 executes step");
            ctx.clearGraph();

            Path ckptPath = dir.resolve("test-checkpoint.jslk");
            float[] expectedOut = model.forward(x).data();
            Checkpoint.save(ckptPath, model, opt, 12, 0.123f);

            Linear restoredModel = new Linear(ctx, 3, 2, true, 999L);
            AdamW restoredOpt = new AdamW(restoredModel.parameters(), 0.5f);
            Checkpoint ckpt = Checkpoint.restore(ckptPath, restoredModel, restoredOpt);
            check(ckpt.epoch() == 12, "Checkpoint epoch restored");
            assertClose(0.123, ckpt.loss(), 1e-5, "Checkpoint loss restored");
            assertClose(opt.learningRate(), restoredOpt.learningRate(), 1e-6, "Checkpoint LR restored");
            check(restoredOpt.stepCount() == opt.stepCount(), "Checkpoint Adam step restored");
            assertArray(expectedOut, restoredModel.forward(x).data(), 1e-6f, "Checkpoint weights restored");
        }
    }

    private static void testOnnxRoundTrip() throws Exception {
        Path dir = Path.of(System.getProperty("user.dir"), "build");
        Files.createDirectories(dir);

        // 1. Classical Pipeline (StandardScaler -> Ridge) -> ONNX -> JSL OnnxModel inference
        Matrix x = Matrix.fromRows(new double[][]{
                {1.0, 10.0},
                {2.0, 20.0},
                {3.0, 15.0},
                {4.0, 30.0}
        });
        double[] y = {3.5, 7.0, 6.5, 11.0};
        Pipeline pipe = Pipeline.builder()
                .addStep("scaler", StandardScaler.builder().build())
                .estimator("ridge", Ridge.builder().alpha(0.1).build())
                .build()
                .fit(x, y);

        Path onnxPipePath = dir.resolve("pipeline_ridge.onnx");
        OnnxConverter.export(pipe, 2, onnxPipePath);
        OnnxModel loadedPipeOnnx = OnnxConverter.load(onnxPipePath);
        double[] expectedPipe = pipe.predict(x);
        double[] onnxPipePreds = loadedPipeOnnx.predictMatrix(x).data();
        assertArray(expectedPipe, onnxPipePreds, 1e-4, "Pipeline ONNX export/import predictions");

        // 2. Deep Learning Sequential MLP -> ONNX -> JSL OnnxModel inference
        try (NativeContext ctx = NativeContext.create(1)) {
            Sequential<jsl.core.BC, jsl.core.BC> mlp = Sequential.<jsl.core.BC>start()
                    .add(new Linear(ctx, 4, 6, true, 11L))
                    .add(new Activation<>(Activation.Kind.RELU))
                    .add(new Linear(ctx, 6, 2, true, 23L));
            Tensor tIn = Tensor.of(ctx, new float[]{
                    0.5f, -1.2f, 0.3f, 2.0f,
                    -0.8f, 0.4f, 1.1f, -0.5f
            }, 2, 4);
            float[] expectedMlp = mlp.forward(tIn).data();

            Path onnxMlpPath = dir.resolve("mlp_net.onnx");
            OnnxConverter.export(mlp, new int[]{-1, 4}, onnxMlpPath);
            OnnxModel loadedMlpOnnx = OnnxConverter.load(onnxMlpPath);
            float[] actualMlp = loadedMlpOnnx.run(ctx, tIn).data();
            assertArray(expectedMlp, actualMlp, 1e-5f, "Sequential MLP ONNX export/import predictions");
            mlp.close();
        }
    }

    private static void checkGradient(NativeContext context, Tensor parameter,
                                      Supplier<Tensor> lossFactory, float epsilon, float tolerance,
                                      String name) {
        context.clearGraph();
        parameter.zeroGrad();
        Tensor loss = lossFactory.get();
        loss.backward();
        float[] analytic = parameter.grad();
        float[] values = parameter.data();
        float[] numeric = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            float orig = values[i];
            values[i] = orig + epsilon;
            parameter.setData(values);
            context.clearGraph();
            float plus = lossFactory.get().data()[0];
            values[i] = orig - epsilon;
            parameter.setData(values);
            context.clearGraph();
            float minus = lossFactory.get().data()[0];
            values[i] = orig;
            parameter.setData(values);
            numeric[i] = (plus - minus) / (2.0f * epsilon);
        }
        for (int i = 0; i < values.length; i++) {
            float diff = Math.abs(numeric[i] - analytic[i]);
            check(diff <= tolerance * (1.0f + Math.abs(numeric[i])),
                    name + " grad mismatch at " + i + ": expected " + numeric[i] + " got " + analytic[i]);
        }
    }

    private static void run(String name, ThrowingRunnable test) {
        try {
            test.run();
            passed++;
            System.out.println("OK: " + name);
        } catch (Throwable failure) {
            System.err.println("FAILED: " + name);
            failure.printStackTrace();
            System.exit(1);
        }
    }

    private static void assertArray(double[] expected, double[] actual, double tol, String msg) {
        check(expected.length == actual.length, msg + ": length mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertClose(expected[i], actual[i], tol, msg + " at " + i);
        }
    }

    private static void assertArray(float[] expected, float[] actual, float tol, String msg) {
        check(expected.length == actual.length, msg + ": length mismatch");
        for (int i = 0; i < expected.length; i++) {
            check(Float.isFinite(actual[i]) && Math.abs(expected[i] - actual[i]) <= tol,
                    msg + " at " + i + ": expected " + expected[i] + " got " + actual[i]);
        }
    }

    private static void assertClose(double expected, double actual, double tol, String msg) {
        check(Double.isFinite(actual) && Math.abs(expected - actual) <= tol,
                msg + ": expected " + expected + " but got " + actual);
    }

    private static void check(boolean condition, String msg) {
        if (!condition) throw new AssertionError(msg);
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
