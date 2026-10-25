package jsl;

import jsl.core.Layout;
import jsl.core.Matrix;
import jsl.ml.Estimator;
import jsl.nativebridge.NativeContext;
import jsl.torch.AdamW;
import jsl.torch.BCEWithLogitsLoss;
import jsl.torch.ComputationGraph;
import jsl.torch.Conv2d;
import jsl.torch.CosineAnnealingLR;
import jsl.torch.CrossEntropyLoss;
import jsl.torch.DepthwiseSeparableConv2d;
import jsl.torch.Graph;
import jsl.torch.GroupNorm;
import jsl.torch.L1Loss;
import jsl.torch.Linear;
import jsl.torch.MSELoss;
import jsl.torch.SGD;
import jsl.torch.Sequential;
import jsl.torch.SmoothL1Loss;
import jsl.torch.StepLR;
import jsl.torch.Tensor;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Unified SDK entry point for JSL (Java Scikit-Learn + PyTorch-style native runtime).
 */
public final class JSL implements AutoCloseable {
    private final NativeContext context;
    private final AtomicLong seed = new AtomicLong(0x9E3779B97F4A7C15L);

    private JSL(NativeContext context) {
        this.context = context;
    }

    public static JSL create() {
        return new JSL(NativeContext.create());
    }

    public static JSL create(int threads) {
        return new JSL(NativeContext.create(threads));
    }

    public NativeContext context() {
        return context;
    }

    public int threads() {
        return context.threads();
    }

    public Tensor tensor(float[] data, int... shape) {
        return Tensor.of(context, JSLRuntimeException.requireNonNull(data, "data"), shape);
    }

    public Tensor parameter(float[] data, int... shape) {
        return Tensor.of(context, JSLRuntimeException.requireNonNull(data, "data"), true, shape);
    }

    public Tensor zeros(int... shape) {
        return Tensor.zeros(context, shape);
    }

    public Tensor randn(int... shape) {
        return Tensor.random(context, shape, seed.getAndAdd(0x9E3779B97F4A7C15L), 1.0f, false);
    }

    public Tensor parameter(int... shape) {
        return Tensor.parameter(context, shape);
    }

    public Linear linear(int inputSize, int outputSize) {
        return new Linear(context, inputSize, outputSize, true,
                seed.getAndAdd(0x9E3779B97F4A7C15L));
    }

    public Conv2d conv2d(int inputChannels, int outputChannels, int kernelSize) {
        return conv2d(inputChannels, outputChannels, kernelSize, 1, 0);
    }

    public Conv2d conv2d(int inputChannels, int outputChannels, int kernelSize, int stride,
                         int padding) {
        return new Conv2d(context, inputChannels, outputChannels, kernelSize, kernelSize,
                stride, stride, padding, padding, true, seed.getAndAdd(0x9E3779B97F4A7C15L));
    }

    public Conv2d conv2d(int inputChannels, int outputChannels, int kernelSize, int stride,
                         int padding, int dilation, int groups) {
        return new Conv2d(context, inputChannels, outputChannels, kernelSize, kernelSize,
                stride, stride, padding, padding, dilation, dilation, groups, true,
                seed.getAndAdd(0x9E3779B97F4A7C15L));
    }

    public DepthwiseSeparableConv2d depthwiseSeparableConv2d(
            int inputChannels, int outputChannels, int kernelSize, int stride, int padding) {
        return new DepthwiseSeparableConv2d(context, inputChannels, outputChannels,
                kernelSize, stride, padding, seed.getAndAdd(0x9E3779B97F4A7C15L));
    }

    public <L extends Layout> GroupNorm<L> groupNorm(int numGroups, int numChannels) {
        return new GroupNorm<>(context, numGroups, numChannels);
    }

    public <L extends Layout> Sequential<L, L> sequential() {
        return Sequential.start();
    }

    public ComputationGraph.Builder computationGraph() {
        return ComputationGraph.builder();
    }

    public <L extends Layout> jsl.torch.Activation<L> activation(jsl.nn.Activation activation) {
        JSLRuntimeException.requireNonNull(activation, "activation");
        jsl.torch.Activation.Kind kind = switch (activation) {
            case RELU -> jsl.torch.Activation.Kind.RELU;
            case GELU -> jsl.torch.Activation.Kind.GELU;
            case SILU -> jsl.torch.Activation.Kind.SILU;
            case SIGMOID -> jsl.torch.Activation.Kind.SIGMOID;
            case TANH -> jsl.torch.Activation.Kind.TANH;
            case NONE -> throw new JSLRuntimeException("NONE has no tensor kernel");
            case SOFTMAX -> throw new JSLRuntimeException(
                    "use Tensor.softmax(dimension) for SOFTMAX");
        };
        return new jsl.torch.Activation<>(kind);
    }

    public jsl.torch.Optimizer optimizer(jsl.nn.Optimizer optimizer, List<Tensor> parameters,
                                         float learningRate) {
        JSLRuntimeException.requireNonNull(optimizer, "optimizer");
        JSLRuntimeException.requireNonNull(parameters, "parameters");
        return switch (optimizer) {
            case SGD -> new SGD(parameters, learningRate);
            case ADAM -> new AdamW(parameters, learningRate, 0.9f, 0.999f, 1e-8f, 0.0f);
            case ADAMW -> new AdamW(parameters, learningRate);
        };
    }

    public AdamW adamw(List<Tensor> parameters, float learningRate) {
        return new AdamW(parameters, learningRate);
    }

    public SGD sgd(List<Tensor> parameters, float learningRate) {
        return new SGD(parameters, learningRate);
    }

    public CosineAnnealingLR cosineAnnealingLR(jsl.torch.Optimizer optimizer, int tMax,
                                               float etaMin) {
        return new CosineAnnealingLR(optimizer, tMax, etaMin);
    }

    public StepLR stepLR(jsl.torch.Optimizer optimizer, int stepSize, float gamma) {
        return new StepLR(optimizer, stepSize, gamma);
    }

    public CrossEntropyLoss crossEntropy() {
        return new CrossEntropyLoss();
    }

    public MSELoss mse() {
        return new MSELoss();
    }

    public L1Loss l1Loss() {
        return new L1Loss();
    }

    public SmoothL1Loss smoothL1Loss(float beta) {
        return new SmoothL1Loss(beta);
    }

    public BCEWithLogitsLoss bceWithLogitsLoss() {
        return new BCEWithLogitsLoss();
    }

    public Graph graph() {
        return new Graph(context);
    }

    public void clearGraph() {
        context.clearGraph();
    }

    public long[] stats() {
        return context.stats();
    }

    public JslMetrics.MetricsSnapshot metrics() {
        return JslMetrics.snapshot(context);
    }

    /** Classical-model facade; delegates to {@link jsl.ml.JSL}. */
    public static Estimator fit(jsl.ml.JSL.Algorithm algorithm, Matrix features,
                                double[] targets) {
        return jsl.ml.JSL.fit(algorithm, features, targets);
    }

    public static double[] predict(Estimator model, Matrix features) {
        return jsl.ml.JSL.predict(model, features);
    }

    public static double crossValidate(jsl.ml.JSL.Algorithm algorithm, Matrix features,
                                       double[] targets, int folds,
                                       jsl.ml.JSL.Metric metric) {
        return jsl.ml.JSL.crossValidate(algorithm, features, targets, folds, metric);
    }

    @Override
    public void close() {
        context.close();
    }
}
