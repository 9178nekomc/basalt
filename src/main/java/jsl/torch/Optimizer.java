package jsl.torch;

import jsl.JSLRuntimeException;
import jsl.nativebridge.NativeTensorHandle;
import java.util.ArrayList;
import java.util.List;

/** Base optimizer over a parameter list. */
public abstract class Optimizer {
    protected final List<Tensor> parameters;
    protected final float learningRate;

    protected Optimizer(List<Tensor> parameters, float learningRate) {
        JSLRuntimeException.requireNonNull(parameters, "parameters");
        if (parameters.isEmpty()) {
            throw new IllegalArgumentException("optimizer needs at least one parameter");
        }
        if (!(learningRate > 0.0f) || !Float.isFinite(learningRate)) {
            throw new IllegalArgumentException("learning rate must be finite and positive");
        }
        this.parameters = List.copyOf(parameters);
        this.learningRate = learningRate;
    }

    public List<Tensor> parameters() {
        return parameters;
    }

    public void zeroGrad() {
        for (Tensor parameter : parameters) {
            parameter.zeroGrad();
        }
    }

    public abstract void step();

    /** Global L2 gradient clipping; returns the pre-clip norm. */
    public double clipGradNorm(float maxNorm) {
        List<NativeTensorHandle> handles = new ArrayList<>(parameters.size());
        for (Tensor parameter : parameters) {
            handles.add(parameter.handle());
        }
        return parameters.get(0).context().clipGradNorm(handles, maxNorm);
    }
}
