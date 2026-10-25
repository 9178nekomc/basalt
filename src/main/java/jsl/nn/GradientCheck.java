package jsl.nn;

import jsl.core.BC;
import jsl.core.BSF;
import jsl.core.Tensor;
import java.util.List;
import java.util.Objects;

/**
 * Central-difference gradient checker for layers and sequence networks.
 *
 * <p>For a scalar objective {@code L = <upstream, f(x)>} the directional derivative is compared
 * against {@code (L(x + eps) - L(x - eps)) / (2 eps)}. Relative error is normalized by
 * {@code max(1, |analytic| + |numeric|)} so the verdict is scale invariant. Parameter checks
 * sample evenly spaced indices to keep runtime bounded on large layers.</p>
 */
public final class GradientCheck {
    private GradientCheck() {
    }

    public record Report(double maxInputError, double maxParameterError,
                         int inputChecks, int parameterChecks) {
        public boolean passed(double tolerance) {
            return maxInputError <= tolerance && maxParameterError <= tolerance;
        }

        @Override
        public String toString() {
            return String.format(java.util.Locale.ROOT,
                    "inputs=%d maxErr=%.3e, parameters=%d maxErr=%.3e",
                    inputChecks, maxInputError, parameterChecks, maxParameterError);
        }
    }

    public static Report checkLayer(Layer<BSF, BSF> layer, Tensor<BSF> input,
                                    Tensor<BSF> upstream, double epsilon) {
        return checkLayer(layer, input, upstream, epsilon, 8);
    }

    public static Report checkLayer(Layer<BSF, BSF> layer, Tensor<BSF> input,
                                    Tensor<BSF> upstream, double epsilon, int samplesPerParameter) {
        Objects.requireNonNull(layer, "layer");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(upstream, "upstream");
        if (!(epsilon > 0.0)) {
            throw new IllegalArgumentException("epsilon must be positive");
        }
        Tensor<BSF> output = layer.forward(input);
        if (output.size() != upstream.size()) {
            throw new IllegalArgumentException("Upstream gradient shape mismatch");
        }
        double[] analyticInput = layer.backward(upstream).data().clone();
        double[] values = input.data();
        double[] weights = upstream.data();
        double maximumInputError = 0.0;
        double[] plus = values.clone();
        double[] minus = values.clone();
        for (int index = 0; index < values.length; index++) {
            plus[index] = values[index] + epsilon;
            minus[index] = values[index] - epsilon;
            double plusLoss = weightedSum(layer.forward(Tensor.ofBSF(
                    input.dimension(0), input.dimension(1), input.dimension(2), plus)), weights);
            double minusLoss = weightedSum(layer.forward(Tensor.ofBSF(
                    input.dimension(0), input.dimension(1), input.dimension(2), minus)), weights);
            plus[index] = values[index];
            minus[index] = values[index];
            double numerical = (plusLoss - minusLoss) / (2.0 * epsilon);
            maximumInputError = Math.max(maximumInputError,
                    relativeError(analyticInput[index], numerical));
        }

        int parameterChecks = 0;
        double maximumParameterError = 0.0;
        List<ParameterRef> parameters = layer.parameters();
        for (ParameterRef parameter : parameters) {
            double[] parameterValues = parameter.values();
            double[] analytic = parameter.gradient().clone();
            int step = Math.max(1, parameterValues.length / Math.max(1, samplesPerParameter));
            for (int index = 0; index < parameterValues.length; index += step) {
                double original = parameterValues[index];
                parameterValues[index] = original + epsilon;
                double plusLoss = weightedSum(layer.forward(input), weights);
                parameterValues[index] = original - epsilon;
                double minusLoss = weightedSum(layer.forward(input), weights);
                parameterValues[index] = original;
                double numerical = (plusLoss - minusLoss) / (2.0 * epsilon);
                maximumParameterError = Math.max(maximumParameterError,
                        relativeError(analytic[index], numerical));
                parameterChecks++;
            }
        }
        layer.forward(input);
        return new Report(maximumInputError, maximumParameterError,
                values.length, parameterChecks);
    }

    public static Report checkNetwork(SequenceNetwork network, Tensor<BSF> input,
                                      Tensor<BC> target, double epsilon) {
        return checkNetwork(network, input, target, epsilon, 8);
    }

    public static Report checkNetwork(SequenceNetwork network, Tensor<BSF> input,
                                      Tensor<BC> target, double epsilon, int samplesPerParameter) {
        Objects.requireNonNull(network, "network");
        if (!(epsilon > 0.0)) {
            throw new IllegalArgumentException("epsilon must be positive");
        }
        LossEvaluation evaluation = network.evaluate(input, target);
        if (!Double.isFinite(evaluation.loss())) {
            throw new IllegalStateException("Loss is not finite");
        }
        network.backward(evaluation);
        List<ParameterRef> parameters = network.parameters();
        double maximumParameterError = 0.0;
        int checks = 0;
        for (ParameterRef parameter : parameters) {
            double[] values = parameter.values();
            double[] analytic = parameter.gradient().clone();
            int step = Math.max(1, values.length / Math.max(1, samplesPerParameter));
            for (int index = 0; index < values.length; index += step) {
                double original = values[index];
                values[index] = original + epsilon;
                double plusLoss = network.evaluate(input, target).loss();
                values[index] = original - epsilon;
                double minusLoss = network.evaluate(input, target).loss();
                values[index] = original;
                double numerical = (plusLoss - minusLoss) / (2.0 * epsilon);
                maximumParameterError = Math.max(maximumParameterError,
                        relativeError(analytic[index], numerical));
                checks++;
            }
        }
        network.evaluate(input, target);
        return new Report(0.0, maximumParameterError, 0, checks);
    }

    private static double weightedSum(Tensor<BSF> output, double[] weights) {
        double[] values = output.data();
        double sum = 0.0;
        for (int index = 0; index < values.length; index++) {
            sum += values[index] * weights[index];
        }
        return sum;
    }

    private static double relativeError(double analytic, double numerical) {
        double denominator = Math.max(1.0, Math.abs(analytic) + Math.abs(numerical));
        return Math.abs(analytic - numerical) / denominator;
    }
}
