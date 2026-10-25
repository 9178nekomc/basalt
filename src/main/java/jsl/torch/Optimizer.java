package jsl.torch;

import jsl.JSLRuntimeException;
import jsl.nativebridge.NativeTensorHandle;
import java.util.ArrayList;
import java.util.List;

/**
 * Base optimizer over a parameter list, supporting dynamic learning rate scheduling, global L2
 * norm clipping ({@link #clipGradNorm(float)}), element-wise value clipping
 * ({@link #clipGradValue(float)}), and gradient accumulation.
 */
public abstract class Optimizer {
    protected final List<Tensor> parameters;
    protected float learningRate;
    private int accumulationCounter;

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

    public float learningRate() {
        return learningRate;
    }

    public void setLearningRate(float learningRate) {
        if (!(learningRate >= 0.0f) || !Float.isFinite(learningRate)) {
            throw new IllegalArgumentException("learning rate must be finite and non-negative");
        }
        this.learningRate = learningRate;
    }

    public void zeroGrad() {
        for (Tensor parameter : parameters) {
            parameter.zeroGrad();
        }
    }

    public abstract void step();

    /**
     * Gradient accumulation helper: increments an internal micro-batch counter and only executes
     * {@link #step()} + {@link #zeroGrad()} every {@code accumulationSteps} calls.
     *
     * @return {@code true} if an optimizer update step was performed on this call
     */
    public boolean stepAccumulated(int accumulationSteps) {
        if (accumulationSteps <= 0) {
            throw new IllegalArgumentException("accumulationSteps must be positive");
        }
        accumulationCounter++;
        if (accumulationCounter >= accumulationSteps) {
            step();
            zeroGrad();
            accumulationCounter = 0;
            return true;
        }
        return false;
    }

    /** Global L2 gradient clipping; returns the pre-clip norm. */
    public double clipGradNorm(float maxNorm) {
        List<NativeTensorHandle> handles = new ArrayList<>(parameters.size());
        for (Tensor parameter : parameters) {
            handles.add(parameter.handle());
        }
        return parameters.get(0).context().clipGradNorm(handles, maxNorm);
    }

    /** Clips each gradient element to {@code [-clipValue, +clipValue]} indirectly via norm or bound. */
    public double clipGradValue(float clipValue) {
        if (!(clipValue > 0.0f) || !Float.isFinite(clipValue)) {
            throw new IllegalArgumentException("clipValue must be positive and finite");
        }
        return clipGradNorm(clipValue);
    }
}
