package basalt.torch;

import basalt.BasaltRuntimeException;
import basalt.core.Layout;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Base class for a typed neural-network module.
 *
 * <p>The two type parameters are phantom layout tags. They make the <em>pipeline</em> checkable at
 * compile time: {@code Conv2d} is {@code Module<BCHW, BCHW>}, {@code Flatten} is
 * {@code Module<BCHW, BC>}, and {@code Linear} is {@code Module<BC, BC>}. Chaining them through
 * {@link Sequential#add} therefore rejects a {@code Conv2d -> Linear} connection at compile time,
 * something a dynamically typed framework can only detect at runtime.</p>
 */
public abstract class Module<I extends Layout, O extends Layout> implements AutoCloseable {
    private boolean training = true;

    public abstract Tensor forward(Tensor input);

    public abstract List<Tensor> parameters();

    /** State-dict entries; defaults to the trainable parameters with positional names. */
    public List<Map.Entry<String, Tensor>> namedParameters() {
        List<Map.Entry<String, Tensor>> entries = new ArrayList<>();
        List<Tensor> parameters = parameters();
        for (int index = 0; index < parameters.size(); index++) {
            entries.add(Map.entry("param" + index, parameters.get(index)));
        }
        return entries;
    }

    public Tensor apply(Tensor input) {
        BasaltRuntimeException.requireNonNull(input, "input");
        return forward(input);
    }

    public void train() {
        training = true;
    }

    public void eval() {
        training = false;
    }

    public boolean isTraining() {
        return training;
    }

    public void zeroGrad() {
        for (Tensor parameter : parameters()) {
            parameter.zeroGrad();
        }
    }

    /** Flattens the state dict into name -> float[] for serialization. */
    public Map<String, float[]> stateDict() {
        Map<String, float[]> state = new LinkedHashMap<>();
        for (Map.Entry<String, Tensor> entry : namedParameters()) {
            state.put(entry.getKey(), entry.getValue().data());
        }
        return state;
    }

    public void loadStateDict(Map<String, float[]> state) {
        for (Map.Entry<String, Tensor> entry : namedParameters()) {
            float[] values = state.get(entry.getKey());
            if (values == null) {
                throw new IllegalArgumentException("missing state entry " + entry.getKey());
            }
            entry.getValue().setData(values);
        }
    }

    @Override
    public void close() {
        for (Tensor parameter : parameters()) {
            parameter.close();
        }
    }
}
