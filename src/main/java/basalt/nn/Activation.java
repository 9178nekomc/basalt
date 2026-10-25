package basalt.nn;

/** Strongly typed activation functions. SOFTMAX is applied independently per batch row. */
public enum Activation {
    NONE,
    RELU,
    SIGMOID,
    TANH,
    SOFTMAX,
    GELU,
    SILU;

    double applyScalar(double value) {
        return switch (this) {
            case NONE -> value;
            case RELU -> Math.max(0.0, value);
            case SIGMOID -> sigmoid(value);
            case TANH -> Math.tanh(value);
            case GELU -> 0.5 * value * (1.0 + Math.tanh(
                    Math.sqrt(2.0 / Math.PI) * (value + 0.044715 * value * value * value)));
            case SILU -> value * sigmoid(value);
            case SOFTMAX -> throw new IllegalStateException("Softmax requires a complete row");
        };
    }

    void applyRow(double[] values, int offset, int length) {
        if (this != SOFTMAX) {
            for (int i = 0; i < length; i++) {
                values[offset + i] = applyScalar(values[offset + i]);
            }
            return;
        }
        double maximum = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < length; i++) {
            maximum = Math.max(maximum, values[offset + i]);
        }
        double sum = 0.0;
        for (int i = 0; i < length; i++) {
            double exponential = Math.exp(values[offset + i] - maximum);
            values[offset + i] = exponential;
            sum += exponential;
        }
        if (!Double.isFinite(sum) || sum <= 0.0) {
            throw new IllegalStateException("Softmax produced a non-finite normalizer");
        }
        for (int i = 0; i < length; i++) {
            values[offset + i] /= sum;
        }
    }

    double derivative(double preActivation, double activated) {
        return switch (this) {
            case NONE -> 1.0;
            case RELU -> preActivation > 0.0 ? 1.0 : 0.0;
            case SIGMOID -> activated * (1.0 - activated);
            case TANH -> 1.0 - activated * activated;
            case GELU -> {
                double inner = Math.sqrt(2.0 / Math.PI)
                        * (preActivation + 0.044715 * preActivation * preActivation * preActivation);
                double tanh = Math.tanh(inner);
                yield 0.5 * (1.0 + tanh)
                        + 0.5 * preActivation * (1.0 - tanh * tanh)
                        * Math.sqrt(2.0 / Math.PI)
                        * (1.0 + 3.0 * 0.044715 * preActivation * preActivation);
            }
            case SILU -> {
                double sigmoid = sigmoid(preActivation);
                yield sigmoid * (1.0 + preActivation * (1.0 - sigmoid));
            }
            case SOFTMAX -> throw new IllegalStateException("Softmax uses its full Jacobian");
        };
    }

    private static double sigmoid(double value) {
        if (value >= 0.0) {
            return 1.0 / (1.0 + Math.exp(-value));
        }
        double exponential = Math.exp(value);
        return exponential / (1.0 + exponential);
    }
}
