package basalt.torch;

import basalt.core.BCHW;
import basalt.nativebridge.NativeContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Batch normalization over NCHW. Running statistics are state, not trainable parameters. */
public final class BatchNorm2d extends Module<BCHW, BCHW> {
    private final Tensor gamma;
    private final Tensor beta;
    private final Tensor runningMean;
    private final Tensor runningVar;
    private final float epsilon;
    private final float momentum;

    public BatchNorm2d(NativeContext context, int channels) {
        this(context, channels, 1e-5f, 0.1f);
    }

    public BatchNorm2d(NativeContext context, int channels, float epsilon, float momentum) {
        if (channels <= 0) {
            throw new IllegalArgumentException("channels must be positive");
        }
        this.epsilon = epsilon;
        this.momentum = momentum;
        gamma = Tensor.parameter(context, channels);
        beta = Tensor.parameter(context, channels);
        gamma.fill(1.0f);
        beta.fill(0.0f);
        runningMean = Tensor.zeros(context, channels);
        runningVar = Tensor.zeros(context, channels);
        runningVar.fill(1.0f);
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.batchNorm2d(gamma, beta, runningMean, runningVar, epsilon, momentum,
                isTraining());
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> parameters = new ArrayList<>();
        parameters.add(gamma);
        parameters.add(beta);
        return parameters;
    }

    public float epsilon() {
        return epsilon;
    }

    public Tensor gamma() {
        return gamma;
    }

    public Tensor beta() {
        return beta;
    }

    public Tensor runningMean() {
        return runningMean;
    }

    public Tensor runningVar() {
        return runningVar;
    }

    @Override
    public List<Map.Entry<String, Tensor>> namedParameters() {
        List<Map.Entry<String, Tensor>> entries = new ArrayList<>();
        entries.add(Map.entry("gamma", gamma));
        entries.add(Map.entry("beta", beta));
        entries.add(Map.entry("running_mean", runningMean));
        entries.add(Map.entry("running_var", runningVar));
        return entries;
    }

    @Override
    public void close() {
        gamma.close();
        beta.close();
        runningMean.close();
        runningVar.close();
    }
}
