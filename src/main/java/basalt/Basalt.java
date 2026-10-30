package basalt;

import basalt.core.Layout;
import basalt.core.Matrix;
import basalt.ml.Estimator;
import basalt.nativebridge.NativeContext;
import basalt.torch.AdamW;
import basalt.torch.BCEWithLogitsLoss;
import basalt.torch.ComputationGraph;
import basalt.torch.Conv2d;
import basalt.torch.CosineAnnealingLR;
import basalt.torch.CrossEntropyLoss;
import basalt.torch.DepthwiseSeparableConv2d;
import basalt.torch.Graph;
import basalt.torch.GroupNorm;
import basalt.torch.L1Loss;
import basalt.torch.Linear;
import basalt.torch.MSELoss;
import basalt.torch.SGD;
import basalt.torch.Sequential;
import basalt.torch.SmoothL1Loss;
import basalt.torch.StepLR;
import basalt.torch.Tensor;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Unified SDK entry point for Basalt: eager tensors, autograd and classical estimators
 * over a Rust native core.
 */
public final class Basalt implements AutoCloseable {
    private final NativeContext context;
    private final AtomicLong seed = new AtomicLong(0x9E3779B97F4A7C15L);

    private Basalt(NativeContext context) {
        this.context = context;
    }

    public static Basalt create() {
        return new Basalt(NativeContext.create());
    }

    public static Basalt create(int threads) {
        return new Basalt(NativeContext.create(threads));
    }

    public NativeContext context() {
        return context;
    }

    public int threads() {
        return context.threads();
    }

    public Tensor tensor(float[] data, int... shape) {
        return Tensor.of(context, BasaltRuntimeException.requireNonNull(data, "data"), shape);
    }

    public Tensor parameter(float[] data, int... shape) {
        return Tensor.of(context, BasaltRuntimeException.requireNonNull(data, "data"), true, shape);
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

    public <L extends Layout> basalt.torch.Activation<L> activation(basalt.nn.Activation activation) {
        BasaltRuntimeException.requireNonNull(activation, "activation");
        basalt.torch.Activation.Kind kind = switch (activation) {
            case RELU -> basalt.torch.Activation.Kind.RELU;
            case GELU -> basalt.torch.Activation.Kind.GELU;
            case SILU -> basalt.torch.Activation.Kind.SILU;
            case SIGMOID -> basalt.torch.Activation.Kind.SIGMOID;
            case TANH -> basalt.torch.Activation.Kind.TANH;
            case NONE -> throw new BasaltRuntimeException("NONE has no tensor kernel");
            case SOFTMAX -> throw new BasaltRuntimeException(
                    "use Tensor.softmax(dimension) for SOFTMAX");
        };
        return new basalt.torch.Activation<>(kind);
    }

    public basalt.torch.Optimizer optimizer(basalt.nn.Optimizer optimizer, List<Tensor> parameters,
                                         float learningRate) {
        BasaltRuntimeException.requireNonNull(optimizer, "optimizer");
        BasaltRuntimeException.requireNonNull(parameters, "parameters");
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

    public CosineAnnealingLR cosineAnnealingLR(basalt.torch.Optimizer optimizer, int tMax,
                                               float etaMin) {
        return new CosineAnnealingLR(optimizer, tMax, etaMin);
    }

    public StepLR stepLR(basalt.torch.Optimizer optimizer, int stepSize, float gamma) {
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

    public BasaltMetrics.MetricsSnapshot metrics() {
        return BasaltMetrics.snapshot(context);
    }

    /** Classical-model facade; delegates to {@link basalt.ml.Estimators}. */
    public static Estimator fit(basalt.ml.Estimators.Algorithm algorithm, Matrix features,
                                double[] targets) {
        return basalt.ml.Estimators.fit(algorithm, features, targets);
    }

    public static double[] predict(Estimator model, Matrix features) {
        return basalt.ml.Estimators.predict(model, features);
    }

    public static double crossValidate(basalt.ml.Estimators.Algorithm algorithm, Matrix features,
                                       double[] targets, int folds,
                                       basalt.ml.Estimators.Metric metric) {
        return basalt.ml.Estimators.crossValidate(algorithm, features, targets, folds, metric);
    }

    @Override
    public void close() {
        context.close();
    }
}
