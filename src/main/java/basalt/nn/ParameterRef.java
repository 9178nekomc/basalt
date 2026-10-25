package basalt.nn;

import java.util.List;

/**
 * Live view over one trainable tensor: parameter values and the gradient buffer produced by the
 * most recent backward pass. Both arrays are owned by the layer and must not be resized.
 */
public record ParameterRef(String name, double[] values, double[] gradient) {
    public ParameterRef {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Parameter name must not be blank");
        }
        if (values.length != gradient.length) {
            throw new IllegalArgumentException("Parameter and gradient lengths differ");
        }
    }

    public int size() {
        return values.length;
    }

    public static ParameterRef empty(String name) {
        return new ParameterRef(name, new double[0], new double[0]);
    }

    public static List<ParameterRef> none() {
        return List.of();
    }
}
