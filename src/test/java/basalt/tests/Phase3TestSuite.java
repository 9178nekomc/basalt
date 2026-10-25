package basalt.tests;

import basalt.Basalt;
import basalt.BasaltMetrics;
import basalt.core.Matrix;
import basalt.ml.FeatureImportance;
import basalt.ml.HistogramGradientBoosting;
import basalt.ml.LimeExplainer;
import basalt.ml.RandomForest;
import basalt.ml.Ridge;
import basalt.ml.ShapExplainer;
import basalt.ml.TreeTask;
import basalt.torch.Linear;
import basalt.torch.Tensor;
import java.util.Random;

/**
 * Comprehensive verification suite for Phase 3:
 * 1. Model Explainability: Intrinsic Tree & Linear Feature Importance, Permutation Feature Importance,
 *    SHAP Value Calculation (exact additivity & LinearSHAP / Coalition TreeSHAP), and LIME local
 *    surrogate explanations.
 * 2. Observability Enhancement: JFR events & BasaltMetrics standardized latency percentiles (p50/p95/p99),
 *    memory footprint tracking, throughput calculation, and JSON/Prometheus export.
 */
public final class Phase3TestSuite {
    private static int passed;

    private Phase3TestSuite() {
    }

    public static void main(String[] args) {
        run("Tree & Permutation Feature Importance ranking", Phase3TestSuite::testFeatureImportance);
        run("SHAP values exact additivity (LinearSHAP & Coalition/TreeSHAP)", Phase3TestSuite::testShapExplainer);
        run("LIME local surrogate explanation fidelity & feature attribution", Phase3TestSuite::testLimeExplainer);
        run("JFR events & BasaltMetrics latency/memory/throughput JSON & Prometheus export", Phase3TestSuite::testObservabilityMetrics);
        System.out.println("PASS: " + passed + " Phase 3 test groups");
    }

    private static void testFeatureImportance() {
        Random rng = new Random(17);
        int rows = 150;
        double[] xData = new double[rows * 3];
        double[] y = new double[rows];
        for (int r = 0; r < rows; r++) {
            double f0 = rng.nextGaussian();
            double f1 = rng.nextGaussian();
            double f2 = rng.nextGaussian();
            xData[r * 3] = f0;
            xData[r * 3 + 1] = f1;
            xData[r * 3 + 2] = f2;
            y[r] = 10.0 * f0 - 3.0 * f1 + 0.05 * f2;
        }
        Matrix x = new Matrix(rows, 3, xData);
        RandomForest rf = RandomForest.builder().task(TreeTask.REGRESSION).trees(30).maxDepth(6).seed(42L).build()
                .fit(x, y);

        double[] intrinsic = FeatureImportance.of(rf);
        check(intrinsic[0] > intrinsic[1] && intrinsic[1] > intrinsic[2],
                "Intrinsic RF importance orders f0 > f1 > f2");

        FeatureImportance.PermutationResult perm = FeatureImportance.permutationImportance(
                rf, x, y, basalt.ml.Estimators.Metric.R2, 5, 42L);
        int[] ranked = perm.rankedFeaturesDescending();
        check(ranked[0] == 0 && ranked[1] == 1 && ranked[2] == 2,
                "Permutation importance ranks f0 > f1 > f2");
    }

    private static void testShapExplainer() {
        Random rng = new Random(23);
        int rows = 80;
        double[] xData = new double[rows * 3];
        double[] y = new double[rows];
        for (int r = 0; r < rows; r++) {
            double f0 = rng.nextGaussian();
            double f1 = rng.nextGaussian();
            double f2 = rng.nextGaussian();
            xData[r * 3] = f0;
            xData[r * 3 + 1] = f1;
            xData[r * 3 + 2] = f2;
            y[r] = 4.0 * f0 - 2.0 * f1 + 0.5;
        }
        Matrix x = new Matrix(rows, 3, xData);

        // 1. LinearSHAP on Ridge
        Ridge ridge = Ridge.builder().alpha(1e-4).build().fit(x, y);
        ShapExplainer ridgeShap = new ShapExplainer(ridge, x);
        Matrix testInstances = Matrix.fromRows(new double[][]{
                {1.5, -1.0, 0.5},
                {-0.5, 2.0, -1.0}
        });
        ShapExplainer.Explanation exp1 = ridgeShap.explain(testInstances);
        for (int r = 0; r < testInstances.rows(); r++) {
            double sumPhi = exp1.baseValue();
            for (int c = 0; c < 3; c++) sumPhi += exp1.shapValues().get(r, c);
            assertClose(exp1.predictions()[r], sumPhi, 1e-6, "LinearSHAP local additivity row " + r);
        }
        check(exp1.meanAbsShap()[0] > exp1.meanAbsShap()[2], "SHAP identifies f0 >> f2");

        // 2. Coalition Shapley on non-linear HistogramGradientBoosting
        HistogramGradientBoosting hgb = HistogramGradientBoosting.builder()
                .iterations(25).maxDepth(4).seed(42L).build().fit(x, y);
        ShapExplainer treeShap = new ShapExplainer(hgb, x);
        ShapExplainer.Explanation exp2 = treeShap.explain(testInstances);
        for (int r = 0; r < testInstances.rows(); r++) {
            double sumPhi = exp2.baseValue();
            for (int c = 0; c < 3; c++) sumPhi += exp2.shapValues().get(r, c);
            assertClose(exp2.predictions()[r], sumPhi, 1e-5, "Tree/Coalition SHAP local additivity row " + r);
        }
    }

    private static void testLimeExplainer() {
        Random rng = new Random(41);
        int rows = 100;
        double[] xData = new double[rows * 3];
        double[] y = new double[rows];
        for (int r = 0; r < rows; r++) {
            double f0 = rng.nextGaussian();
            double f1 = rng.nextGaussian();
            double f2 = rng.nextGaussian();
            xData[r * 3] = f0;
            xData[r * 3 + 1] = f1;
            xData[r * 3 + 2] = f2;
            y[r] = 6.0 * f0 - 2.5 * f1 + 1.0;
        }
        Matrix x = new Matrix(rows, 3, xData);
        Ridge ridge = Ridge.builder().alpha(1e-4).build().fit(x, y);

        LimeExplainer lime = new LimeExplainer(ridge, x, 400, 0.0, 42L);
        LimeExplainer.LocalExplanation local = lime.explainInstance(new double[]{1.0, -1.0, 0.2});
        check(local.r2Fidelity() > 0.98, "LIME local surrogate R2 fidelity > 0.98, got " + local.r2Fidelity());
        assertClose(local.modelPrediction(), local.localPrediction(), 0.05, "LIME local prediction matches model");
        assertClose(6.0, local.featureWeights()[0], 0.1, "LIME recovers local slope f0 ~ 6.0");
        assertClose(-2.5, local.featureWeights()[1], 0.1, "LIME recovers local slope f1 ~ -2.5");
        check(local.topFeatures()[0] == 0, "LIME ranks feature 0 as top contributor");
    }

    private static void testObservabilityMetrics() {
        BasaltMetrics.reset();
        try (Basalt engine = Basalt.create(2)) {
            Tensor x = engine.randn(16, 8);
            Linear layer = engine.linear(8, 4);
            for (int i = 0; i < 10; i++) {
                Tensor out = BasaltMetrics.time("linear.forward", x.size(), x.size() * 4L,
                        () -> layer.forward(x).relu());
                out.close();
                engine.clearGraph();
            }
            BasaltMetrics.MetricsSnapshot snap = engine.metrics();
            check(snap.memory().liveNativeTensors() >= 3, "Tracks live native tensors");
            check(snap.memory().liveNativeBytes() > 0, "Tracks live native off-heap bytes");
            check(snap.memory().jvmHeapUsedBytes() > 0, "Tracks JVM heap bytes");

            BasaltMetrics.OperatorSummary qSummary = snap.operators().get("native.command_queue");
            check(qSummary != null && qSummary.calls() >= 20, "Native command queue calls recorded");
            check(qSummary.p50Micros() >= 0.0 && qSummary.p99Micros() >= qSummary.p50Micros(),
                    "Percentiles ordered p99 >= p50");
            check(qSummary.throughputOpsPerSec() > 0.0, "Throughput ops/sec > 0");

            String json = snap.toJson();
            check(json.contains("\"liveNativeBytes\"") && json.contains("\"native.command_queue\""),
                    "JSON metrics export contains memory and operator stats");
            String prom = snap.toPrometheus();
            check(prom.contains("basalt_operator_calls_total{op=\"native.command_queue\"}")
                    && prom.contains("basalt_native_live_bytes"),
                    "Prometheus exposition format valid");
        }
    }

    private static void run(String name, Runnable test) {
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

    private static void assertClose(double expected, double actual, double tol, String msg) {
        check(Double.isFinite(actual) && Math.abs(expected - actual) <= tol,
                msg + ": expected " + expected + " but got " + actual);
    }

    private static void check(boolean condition, String msg) {
        if (!condition) throw new AssertionError(msg);
    }
}
