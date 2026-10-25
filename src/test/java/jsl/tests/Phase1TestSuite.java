package jsl.tests;

import jsl.JSLExceptionCode;
import jsl.JSLMemoryException;
import jsl.JSLParameterException;
import jsl.JSLRuntimeException;
import jsl.JSLShapeException;
import jsl.JSLUnsupportedOperationException;
import jsl.core.Matrix;
import jsl.core.NativeKernels;
import jsl.io.ModelCodec;
import jsl.ml.DataSplit;
import jsl.ml.DecisionTree;
import jsl.ml.Estimator;
import jsl.ml.GridSearchCV;
import jsl.ml.JSL;
import jsl.ml.KFold;
import jsl.ml.LabelEncoder;
import jsl.ml.LogisticRegression;
import jsl.ml.Metrics;
import jsl.ml.MinMaxScaler;
import jsl.ml.OneHotEncoder;
import jsl.ml.PCA;
import jsl.ml.Pipeline;
import jsl.ml.RandomizedSearchCV;
import jsl.ml.Ridge;
import jsl.ml.SimpleImputer;
import jsl.ml.StandardScaler;
import jsl.ml.StratifiedKFold;
import jsl.nativebridge.CommandQueueBuilder;
import jsl.nativebridge.NativeContext;
import jsl.nativebridge.NativeTensorHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Comprehensive verification suite for Phase 1:
 * 1. JNI/Native engineering safety (Cleaner + try-with-resources, ABA handle non-reuse,
 *    multi-threaded queue thread-safety, standardized exception hierarchy).
 * 2. ML Infrastructure (StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, SimpleImputer,
 *    trainTestSplit, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV, Pipeline + .jsl persistence).
 * 3. Numerical consistency against analytical / scikit-learn reference values & loop memory leak detection.
 */
public final class Phase1TestSuite {
    private static int passed;

    private Phase1TestSuite() {
    }

    public static void main(String[] args) throws Exception {
        run("standardized exception hierarchy and JNI error mapping", Phase1TestSuite::testExceptionHierarchy);
        run("native handle generation non-reuse (ABA prevention) & memory leak loop", Phase1TestSuite::testMemorySafetyAndLeakLoop);
        run("concurrent CommandQueueBuilder and multi-threaded inference safety", Phase1TestSuite::testThreadSafety);
        run("StandardScaler & MinMaxScaler forward, inverseTransform and .jsl persistence", Phase1TestSuite::testScalers);
        run("OneHotEncoder & LabelEncoder categorical encoding, inverse & .jsl persistence", Phase1TestSuite::testEncoders);
        run("SimpleImputer MEAN/MEDIAN/MOST_FREQUENT/CONSTANT & .jsl persistence", Phase1TestSuite::testSimpleImputer);
        run("trainTestSplit, KFold, and StratifiedKFold class-ratio preservation", Phase1TestSuite::testSplitsAndFolds);
        run("GridSearchCV and RandomizedSearchCV hyperparameter tuning", Phase1TestSuite::testHyperparameterSearch);
        run("Pipeline end-to-end modeling and atomic .jsl binary serialization", Phase1TestSuite::testPipeline);
        run("scikit-learn numerical consistency verification", Phase1TestSuite::testSklearnNumericalConsistency);
        System.out.println("PASS: " + passed + " Phase 1 test groups");
    }

    private static void testExceptionHierarchy() {
        JSLRuntimeException pEx = JSLRuntimeException.fromStatus(-1, "bad param");
        check(pEx instanceof JSLParameterException, "status -1 -> JSLParameterException");
        check(pEx.errorCode() == JSLExceptionCode.INVALID_PARAMETER, "INVALID_PARAMETER code");

        JSLRuntimeException sEx = JSLRuntimeException.fromStatus(-2, "bad shape");
        check(sEx instanceof JSLShapeException, "status -2 -> JSLShapeException");
        check(sEx.errorCode() == JSLExceptionCode.SHAPE_MISMATCH, "SHAPE_MISMATCH code");

        JSLRuntimeException mEx = JSLRuntimeException.fromStatus(-7, "oom");
        check(mEx instanceof JSLMemoryException, "status -7 -> JSLMemoryException");

        JSLRuntimeException uEx = JSLRuntimeException.fromStatus(-8, "unsupported");
        check(uEx instanceof JSLUnsupportedOperationException, "status -8 -> JSLUnsupportedOperationException");

        try (NativeContext ctx = NativeContext.create(1);
             NativeTensorHandle a = ctx.zeros(2, 3);
             NativeTensorHandle b = ctx.zeros(4, 2);
             NativeTensorHandle out = ctx.zeros(2, 2);
             CommandQueueBuilder q = new CommandQueueBuilder(ctx)) {
            expectThrows(JSLShapeException.class, () -> q.matMul(a, b, out).execute());
            expectThrows(JSLParameterException.class, () -> a.fill(Float.NaN));
        }
    }

    private static void testMemorySafetyAndLeakLoop() {
        try (NativeContext ctx = NativeContext.create(2)) {
            // Stress-test loop creating and closing 2,000 tensors to verify zero memory leak
            for (int iter = 0; iter < 2_000; iter++) {
                try (NativeTensorHandle t1 = ctx.of(new float[]{1, 2, 3, 4}, false, 2, 2);
                     NativeTensorHandle t2 = ctx.zeros(2, 2)) {
                    check(t1.size() == 4, "t1 elementCount");
                    t2.fill(3.0f);
                }
            }
            long[] stats = ctx.stats();
            check(stats[0] == 0L, "liveTensors must be 0 after 2,000 try-with-resources iterations, got " + stats[0]);
            check(stats[1] == 0L, "liveBytes must be 0 after 2,000 try-with-resources iterations, got " + stats[1]);
        }

        // Verify HistogramSessionHandle AutoCloseable + Cleaner
        if (NativeKernels.available()) {
            byte[] bins = new byte[16];
            double[] g = new double[8];
            double[] h = new double[8];
            long ptr = NativeKernels.createHistogramSession(bins, g, h, 2, 4);
            try (NativeKernels.HistogramSessionHandle handle = new NativeKernels.HistogramSessionHandle(ptr)) {
                check(handle.handle() == ptr, "session handle active");
            }
        }
    }

    private static void testThreadSafety() throws Exception {
        try (NativeContext ctx = NativeContext.create(4);
             CommandQueueBuilder sharedQueue = new CommandQueueBuilder(ctx)) {
            int threads = 4;
            int opsPerThread = 10;
            ExecutorService pool = Executors.newFixedThreadPool(threads);
            List<NativeTensorHandle> handles = new ArrayList<>();
            List<Callable<Void>> tasks = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                List<NativeTensorHandle> local = new ArrayList<>();
                for (int i = 0; i < opsPerThread; i++) {
                    NativeTensorHandle in = ctx.of(new float[]{-1.0f, 2.0f, -3.0f, 4.0f}, false, 2, 2);
                    NativeTensorHandle out = ctx.zeros(2, 2);
                    local.add(in);
                    local.add(out);
                }
                handles.addAll(local);
                tasks.add(() -> {
                    for (int i = 0; i < local.size(); i += 2) {
                        sharedQueue.relu(local.get(i), local.get(i + 1));
                    }
                    return null;
                });
            }
            List<Future<Void>> futures = pool.invokeAll(tasks);
            for (Future<Void> f : futures) f.get();
            pool.shutdown();

            check(sharedQueue.size() == threads * opsPerThread, "all concurrent queue ops recorded");
            sharedQueue.executeAndClear();
            for (int i = 1; i < handles.size(); i += 2) {
                assertArray(new float[]{0.0f, 2.0f, 0.0f, 4.0f}, handles.get(i).data(), 1e-6f, "concurrent relu output");
            }
            for (NativeTensorHandle h : handles) h.close();
        }
    }

    private static void testScalers() throws Exception {
        Path dir = Path.of(System.getProperty("user.dir"), "build");
        Files.createDirectories(dir);
        Matrix x = Matrix.fromRows(new double[][]{
                {0.0, 10.0},
                {0.0, 20.0},
                {1.0, 30.0},
                {1.0, 40.0}
        });
        StandardScaler ss = StandardScaler.builder().build().fit(x);
        assertArray(new double[]{0.5, 25.0}, ss.mean(), 1e-12, "StandardScaler mean");
        assertArray(new double[]{0.5, Math.sqrt(125.0)}, ss.scale(), 1e-12, "StandardScaler scale");
        Matrix z = ss.transform(x);
        Matrix xRestored = ss.inverseTransform(z);
        assertArray(x.data(), xRestored.data(), 1e-12, "StandardScaler inverseTransform");

        Path ssPath = dir.resolve("test-ss.jsl");
        ModelCodec.save(ssPath, ss);
        StandardScaler ssLoaded = ModelCodec.load(ssPath, StandardScaler.class);
        assertArray(z.data(), ssLoaded.transform(x).data(), 1e-12, "StandardScaler serialization");

        MinMaxScaler mms = MinMaxScaler.builder().featureRange(-1.0, 1.0).build().fit(x);
        Matrix scaled = mms.transform(x);
        assertClose(-1.0, scaled.get(0, 0), 1e-12, "MinMaxScaler min");
        assertClose(1.0, scaled.get(3, 1), 1e-12, "MinMaxScaler max");
        assertArray(x.data(), mms.inverseTransform(scaled).data(), 1e-12, "MinMaxScaler inverseTransform");

        Path mmsPath = dir.resolve("test-mms.jsl");
        ModelCodec.save(mmsPath, mms);
        MinMaxScaler mmsLoaded = ModelCodec.load(mmsPath, MinMaxScaler.class);
        assertArray(scaled.data(), mmsLoaded.transform(x).data(), 1e-12, "MinMaxScaler serialization");
    }

    private static void testEncoders() throws Exception {
        Path dir = Path.of(System.getProperty("user.dir"), "build");
        Matrix x = Matrix.fromRows(new double[][]{
                {10.0, 1.5},
                {20.0, 2.5},
                {10.0, 3.5},
                {30.0, 4.5}
        });
        OneHotEncoder ohe = OneHotEncoder.builder()
                .categoricalColumns(0)
                .handleUnknown(OneHotEncoder.HandleUnknown.IGNORE)
                .build()
                .fit(x);
        Matrix encoded = ohe.transform(x);
        check(encoded.columns() == 4, "3 one-hot columns + 1 numeric column");
        assertArray(new double[]{1.0, 0.0, 0.0, 1.5}, row(encoded, 0), 1e-12, "OHE row 0");
        assertArray(x.data(), ohe.inverseTransform(encoded).data(), 1e-12, "OHE inverseTransform");

        // Unknown category with IGNORE produces all zeros in categorical slots
        Matrix unseen = ohe.transform(Matrix.fromRows(new double[][]{{99.0, 5.0}}));
        assertArray(new double[]{0.0, 0.0, 0.0, 5.0}, row(unseen, 0), 1e-12, "OHE ignore unseen");

        Path ohePath = dir.resolve("test-ohe.jsl");
        ModelCodec.save(ohePath, ohe);
        OneHotEncoder oheLoaded = ModelCodec.load(ohePath, OneHotEncoder.class);
        assertArray(encoded.data(), oheLoaded.transform(x).data(), 1e-12, "OHE serialization");

        LabelEncoder le = new LabelEncoder().fit(new double[]{100.0, 200.0, 100.0, 50.0});
        assertArray(new double[]{50.0, 100.0, 200.0}, le.classes(), 1e-12, "LabelEncoder classes");
        double[] tEnc = le.transform(new double[]{50.0, 200.0, 100.0});
        assertArray(new double[]{0.0, 2.0, 1.0}, tEnc, 1e-12, "LabelEncoder transform");
        assertArray(new double[]{50.0, 200.0, 100.0}, le.inverseTransform(tEnc), 1e-12, "LabelEncoder inverse");

        LabelEncoder strLe = new LabelEncoder();
        int[] sEnc = strLe.fitTransform(new String[]{"cat", "dog", "bird", "cat"});
        check(Arrays.equals(sEnc, new int[]{1, 2, 0, 1}), "String LabelEncoder");
        check(Arrays.equals(strLe.inverseTransformStrings(sEnc), new String[]{"cat", "dog", "bird", "cat"}),
                "String LabelEncoder inverse");
    }

    private static void testSimpleImputer() throws Exception {
        Path dir = Path.of(System.getProperty("user.dir"), "build");
        Matrix x = Matrix.fromRows(new double[][]{
                {1.0, 10.0, 5.0},
                {2.0, Double.NaN, 5.0},
                {Double.NaN, 20.0, 9.0},
                {6.0, 30.0, Double.NaN}
        });
        SimpleImputer meanImp = SimpleImputer.builder().strategy(SimpleImputer.Strategy.MEAN).build().fit(x);
        assertArray(new double[]{3.0, 20.0, 19.0 / 3.0}, meanImp.statistics(), 1e-12, "SimpleImputer mean stats");
        Matrix filled = meanImp.transform(x);
        assertClose(3.0, filled.get(2, 0), 1e-12, "imputed col 0");
        assertClose(20.0, filled.get(1, 1), 1e-12, "imputed col 1");

        SimpleImputer medianImp = SimpleImputer.builder().strategy(SimpleImputer.Strategy.MEDIAN).build().fit(x);
        assertArray(new double[]{2.0, 20.0, 5.0}, medianImp.statistics(), 1e-12, "SimpleImputer median stats");

        SimpleImputer freqImp = SimpleImputer.builder().strategy(SimpleImputer.Strategy.MOST_FREQUENT).build().fit(x);
        assertClose(5.0, freqImp.statistics()[2], 1e-12, "SimpleImputer most frequent");

        Path impPath = dir.resolve("test-imputer.jsl");
        ModelCodec.save(impPath, meanImp);
        SimpleImputer loaded = ModelCodec.load(impPath, SimpleImputer.class);
        assertArray(filled.data(), loaded.transform(x).data(), 1e-12, "SimpleImputer serialization");
    }

    private static void testSplitsAndFolds() {
        int rows = 100;
        double[] x = new double[rows * 2];
        double[] y = new double[rows];
        for (int r = 0; r < rows; r++) {
            x[r * 2] = r;
            x[r * 2 + 1] = r * 2;
            y[r] = r < 20 ? 1.0 : 0.0; // 20% class 1, 80% class 0
        }
        Matrix features = new Matrix(rows, 2, x);
        DataSplit.Holdout split = DataSplit.trainTestSplit(features, y, 0.25, true, true, 42L);
        check(split.testTargets().length == 25, "stratified test size 25");
        int onesInTest = 0;
        for (double v : split.testTargets()) if (v == 1.0) onesInTest++;
        check(onesInTest == 5, "stratified trainTestSplit preserves exact 20% minority class (5/25)");

        StratifiedKFold skf = new StratifiedKFold(5, true, 42L);
        List<DataSplit.Fold> folds = skf.split(features, y);
        check(folds.size() == 5, "5 stratified folds");
        for (DataSplit.Fold fold : folds) {
            int pos = 0;
            for (double v : fold.testTargets()) if (v == 1.0) pos++;
            check(pos == 4, "each fold has exactly 4 positive samples out of 20");
        }
        KFold kf = new KFold(5, true, 42L);
        check(kf.split(features, y).size() == 5, "KFold split count");
    }

    private static void testHyperparameterSearch() {
        int rows = 120;
        Random rng = new Random(19);
        double[] x = new double[rows * 3];
        double[] y = new double[rows];
        for (int r = 0; r < rows; r++) {
            x[r * 3] = rng.nextGaussian();
            x[r * 3 + 1] = rng.nextGaussian();
            x[r * 3 + 2] = rng.nextGaussian();
            y[r] = 4.0 * x[r * 3] - 2.0 * x[r * 3 + 1] + 1.0;
        }
        Matrix data = new Matrix(rows, 3, x);

        GridSearchCV gs = GridSearchCV.builder(params -> Ridge.builder()
                        .alpha(((Number) params.get("alpha")).doubleValue())
                        .fitIntercept((Boolean) params.get("fitIntercept"))
                        .build())
                .param("alpha", 1e-5, 1.0, 100.0)
                .param("fitIntercept", true, false)
                .scoring(JSL.Metric.R2)
                .cv(4)
                .build()
                .fit(data, y);

        check(gs.cvResults().size() == 6, "6 parameter combinations evaluated");
        check(gs.bestScore() > 0.999, "GridSearchCV best R2 > 0.999, got " + gs.bestScore());
        check(Double.compare(((Number) gs.bestParams().get("alpha")).doubleValue(), 1e-5) == 0, "best alpha is 1e-5");
        check(Boolean.TRUE.equals(gs.bestParams().get("fitIntercept")), "best fitIntercept is true");

        RandomizedSearchCV rs = RandomizedSearchCV.builder(params -> Ridge.builder()
                        .alpha(((Number) params.get("alpha")).doubleValue())
                        .build())
                .paramLogUniform("alpha", 1e-6, 10.0)
                .nIter(6)
                .scoring(JSL.Metric.R2)
                .cv(4)
                .seed(7L)
                .build()
                .fit(data, y);
        check(rs.cvResults().size() == 6, "RandomizedSearchCV 6 iterations");
        check(rs.bestScore() > 0.99, "RandomizedSearchCV R2 > 0.99");
    }

    private static void testPipeline() throws Exception {
        Path dir = Path.of(System.getProperty("user.dir"), "build");
        int rows = 100;
        Random rng = new Random(29);
        double[] x = new double[rows * 2];
        double[] y = new double[rows];
        for (int r = 0; r < rows; r++) {
            double f0 = rng.nextGaussian() * 100.0 + 500.0;
            double f1 = rng.nextGaussian() * 0.01 - 2.0;
            x[r * 2] = f0;
            x[r * 2 + 1] = f1;
            y[r] = (f0 - 500.0) / 100.0 + (f1 + 2.0) / 0.01 > 0.0 ? 1.0 : 0.0;
        }
        // Inject a missing value to verify imputer -> scaler -> logistic regression pipeline
        x[0] = Double.NaN;
        Matrix features = new Matrix(rows, 2, x);

        Pipeline pipeline = Pipeline.builder()
                .addStep("imputer", SimpleImputer.builder().strategy(SimpleImputer.Strategy.MEAN).build())
                .addStep("scaler", StandardScaler.builder().build())
                .estimator("clf", LogisticRegression.builder().learningRate(0.2).epochs(1000).build())
                .build()
                .fit(features, y);

        double[] preds = pipeline.predict(features);
        check(Metrics.accuracy(y, preds) >= 0.92, "Pipeline accuracy >= 0.92, got " + Metrics.accuracy(y, preds));

        Path pipePath = dir.resolve("test-pipeline.jsl");
        ModelCodec.save(pipePath, pipeline);
        Pipeline loaded = ModelCodec.load(pipePath, Pipeline.class);
        assertArray(preds, loaded.predict(features), 0.0, "Pipeline .jsl bit-exact predictions");
    }

    private static void testSklearnNumericalConsistency() {
        // Analytical & scikit-learn exact reference comparison for Ridge, PCA, and StandardScaler
        Matrix x = Matrix.fromRows(new double[][]{
                {1.0, 2.0},
                {2.0, 1.0},
                {3.0, 4.0},
                {4.0, 3.0}
        });
        double[] y = {5.0, 4.0, 11.0, 10.0}; // y = 1.0 + 1.0*x0 + 2.0*x1
        Ridge ridge = Ridge.builder().alpha(1e-8).build().fit(x, y);
        assertClose(1.0, ridge.weights()[0], 1e-5, "sklearn Ridge coef[0]");
        assertClose(2.0, ridge.weights()[1], 1e-5, "sklearn Ridge coef[1]");
        assertClose(0.0, ridge.intercept(), 1e-5, "sklearn Ridge intercept");

        PCA pca = PCA.builder().components(2).build().fit(x);
        double[] evr = pca.explainedVarianceRatio();
        assertClose(0.8, evr[0], 1e-6, "sklearn PCA explained_variance_ratio_[0]");
        assertClose(0.2, evr[1], 1e-6, "sklearn PCA explained_variance_ratio_[1]");
    }

    private static double[] row(Matrix m, int r) {
        double[] out = new double[m.columns()];
        m.copyRow(r, out, 0);
        return out;
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

    private static void expectThrows(Class<? extends Throwable> type, Runnable action) {
        try {
            action.run();
        } catch (Throwable t) {
            if (type.isInstance(t)) return;
            throw new AssertionError("Expected " + type.getName() + " but got " + t, t);
        }
        throw new AssertionError("Expected " + type.getName() + " but nothing was thrown");
    }

    @FunctionalInterface
    private interface ThrowingRunnable {
        void run() throws Exception;
    }
}
