package basalt.nn;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/** Trainable array with private SGD/Adam state; both updates run through SIMD loops. */
final class Parameter {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private final double[] values;
    private final double[] firstMoment;
    private final double[] secondMoment;
    private long step;

    Parameter(double[] values) {
        this.values = values;
        this.firstMoment = new double[values.length];
        this.secondMoment = new double[values.length];
    }

    double[] values() {
        return values;
    }

    long step() {
        return step;
    }

    void update(double[] gradient, OptimizerConfig config) {
        if (gradient.length != values.length) {
            throw new IllegalArgumentException("Parameter gradient shape mismatch");
        }
        if (config.optimizer() == Optimizer.SGD) {
            double rate = config.learningRate();
            int index = 0;
            int upper = SPECIES.loopBound(values.length);
            DoubleVector factor = DoubleVector.broadcast(SPECIES, rate);
            for (; index < upper; index += SPECIES.length()) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, values, index);
                DoubleVector g = DoubleVector.fromArray(SPECIES, gradient, index);
                v.sub(g.mul(factor)).intoArray(values, index);
            }
            for (; index < values.length; index++) {
                values[index] -= rate * gradient[index];
            }
            return;
        }

        step++;
        double beta1 = config.beta1();
        double beta2 = config.beta2();
        double rate = config.learningRate();
        double epsilon = config.epsilon();
        double firstCorrection = 1.0 - Math.pow(beta1, step);
        double secondCorrection = 1.0 - Math.pow(beta2, step);
        double firstScale = 1.0 / firstCorrection;
        double secondScale = 1.0 / secondCorrection;
        DoubleVector b1 = DoubleVector.broadcast(SPECIES, beta1);
        DoubleVector b2 = DoubleVector.broadcast(SPECIES, beta2);
        DoubleVector oneMinusB1 = DoubleVector.broadcast(SPECIES, 1.0 - beta1);
        DoubleVector oneMinusB2 = DoubleVector.broadcast(SPECIES, 1.0 - beta2);
        DoubleVector firstHatScale = DoubleVector.broadcast(SPECIES, firstScale);
        DoubleVector secondHatScale = DoubleVector.broadcast(SPECIES, secondScale);
        DoubleVector stepSize = DoubleVector.broadcast(SPECIES, rate);
        DoubleVector eps = DoubleVector.broadcast(SPECIES, epsilon);
        int index = 0;
        int upper = SPECIES.loopBound(values.length);
        for (; index < upper; index += SPECIES.length()) {
            DoubleVector g = DoubleVector.fromArray(SPECIES, gradient, index);
            DoubleVector m = DoubleVector.fromArray(SPECIES, firstMoment, index);
            DoubleVector v = DoubleVector.fromArray(SPECIES, secondMoment, index);
            m = m.mul(b1).add(g.mul(oneMinusB1));
            v = v.mul(b2).add(g.mul(g).mul(oneMinusB2));
            m.intoArray(firstMoment, index);
            v.intoArray(secondMoment, index);
            DoubleVector denominator = v.mul(secondHatScale).sqrt().add(eps);
            DoubleVector parameter = DoubleVector.fromArray(SPECIES, values, index);
            parameter.sub(m.mul(firstHatScale).div(denominator).mul(stepSize)).intoArray(values, index);
        }
        for (; index < values.length; index++) {
            double g = gradient[index];
            firstMoment[index] = beta1 * firstMoment[index] + (1.0 - beta1) * g;
            secondMoment[index] = beta2 * secondMoment[index] + (1.0 - beta2) * g * g;
            double firstHat = firstMoment[index] / firstCorrection;
            double secondHat = secondMoment[index] / secondCorrection;
            values[index] -= rate * firstHat / (Math.sqrt(secondHat) + epsilon);
        }
    }

    /** Sum of squares of the current gradient-free parameter vector, used for diagnostics. */
    double sumSquares() {
        DoubleVector sum = DoubleVector.zero(SPECIES);
        int index = 0;
        int upper = SPECIES.loopBound(values.length);
        for (; index < upper; index += SPECIES.length()) {
            DoubleVector v = DoubleVector.fromArray(SPECIES, values, index);
            sum = v.fma(v, sum);
        }
        double result = sum.reduceLanes(VectorOperators.ADD);
        for (; index < values.length; index++) {
            result += values[index] * values[index];
        }
        return result;
    }
}
