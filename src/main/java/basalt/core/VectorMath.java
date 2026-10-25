package basalt.core;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD primitives shared by matrix multiplication, distance-based estimators, optimizers and
 * normalization. Every kernel is written as a vector main loop plus a scalar tail so arbitrary
 * lengths and unaligned offsets are safe.
 */
public final class VectorMath {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private VectorMath() {
    }

    public static int speciesLength() {
        return SPECIES.length();
    }

    public static double dot(double[] a, int aOffset, double[] b, int bOffset, int length) {
        DoubleVector sum = DoubleVector.zero(SPECIES);
        int index = 0;
        int upper = SPECIES.loopBound(length);
        for (; index < upper; index += SPECIES.length()) {
            DoubleVector left = DoubleVector.fromArray(SPECIES, a, aOffset + index);
            DoubleVector right = DoubleVector.fromArray(SPECIES, b, bOffset + index);
            sum = left.fma(right, sum);
        }
        double result = sum.reduceLanes(VectorOperators.ADD);
        for (; index < length; index++) {
            result = Math.fma(a[aOffset + index], b[bOffset + index], result);
        }
        return result;
    }

    /** {@code target += alpha * source} without allocation. */
    public static void axpy(double alpha, double[] source, double[] target) {
        if (source.length != target.length) {
            throw new IllegalArgumentException("axpy operands must have equal length");
        }
        DoubleVector factor = DoubleVector.broadcast(SPECIES, alpha);
        int index = 0;
        int upper = SPECIES.loopBound(source.length);
        for (; index < upper; index += SPECIES.length()) {
            DoubleVector s = DoubleVector.fromArray(SPECIES, source, index);
            DoubleVector t = DoubleVector.fromArray(SPECIES, target, index);
            t.add(s.mul(factor)).intoArray(target, index);
        }
        for (; index < source.length; index++) {
            target[index] += alpha * source[index];
        }
    }

    /** {@code target += source}. */
    public static void addInto(double[] source, double[] target) {
        axpy(1.0, source, target);
    }

    /** {@code values *= factor}. */
    public static void scaleInPlace(double[] values, double factor) {
        DoubleVector vector = DoubleVector.broadcast(SPECIES, factor);
        int index = 0;
        int upper = SPECIES.loopBound(values.length);
        for (; index < upper; index += SPECIES.length()) {
            DoubleVector.fromArray(SPECIES, values, index).mul(vector).intoArray(values, index);
        }
        for (; index < values.length; index++) {
            values[index] *= factor;
        }
    }

    public static double sum(double[] values) {
        DoubleVector sum = DoubleVector.zero(SPECIES);
        int index = 0;
        int upper = SPECIES.loopBound(values.length);
        for (; index < upper; index += SPECIES.length()) {
            sum = sum.add(DoubleVector.fromArray(SPECIES, values, index));
        }
        double result = sum.reduceLanes(VectorOperators.ADD);
        for (; index < values.length; index++) {
            result += values[index];
        }
        return result;
    }

    public static double sumSquares(double[] values) {
        DoubleVector sum = DoubleVector.zero(SPECIES);
        int index = 0;
        int upper = SPECIES.loopBound(values.length);
        for (; index < upper; index += SPECIES.length()) {
            DoubleVector vector = DoubleVector.fromArray(SPECIES, values, index);
            sum = vector.fma(vector, sum);
        }
        double result = sum.reduceLanes(VectorOperators.ADD);
        for (; index < values.length; index++) {
            result = Math.fma(values[index], values[index], result);
        }
        return result;
    }

    public static double norm(double[] values) {
        return Math.sqrt(sumSquares(values));
    }

    public static double maxAbs(double[] values) {
        DoubleVector maximum = DoubleVector.zero(SPECIES);
        int index = 0;
        int upper = SPECIES.loopBound(values.length);
        for (; index < upper; index += SPECIES.length()) {
            maximum = maximum.max(DoubleVector.fromArray(SPECIES, values, index).abs());
        }
        double result = maximum.reduceLanes(VectorOperators.MAX);
        for (; index < values.length; index++) {
            result = Math.max(result, Math.abs(values[index]));
        }
        return result;
    }

    /**
     * Scales {@code values} down so that its Euclidean norm does not exceed {@code limit}.
     *
     * @return the norm before clipping, or {@code 0} for an empty or zero vector
     */
    public static double clipNorm(double[] values, double limit) {
        double norm = norm(values);
        if (norm > limit && norm > 0.0) {
            scaleInPlace(values, limit / norm);
        }
        return norm;
    }

    public static double squaredEuclidean(
            double[] a, int aOffset, double[] b, int bOffset, int length) {
        DoubleVector sum = DoubleVector.zero(SPECIES);
        int index = 0;
        int upper = SPECIES.loopBound(length);
        for (; index < upper; index += SPECIES.length()) {
            DoubleVector delta = DoubleVector.fromArray(SPECIES, a, aOffset + index)
                    .sub(DoubleVector.fromArray(SPECIES, b, bOffset + index));
            sum = delta.fma(delta, sum);
        }
        double result = sum.reduceLanes(VectorOperators.ADD);
        for (; index < length; index++) {
            double delta = a[aOffset + index] - b[bOffset + index];
            result = Math.fma(delta, delta, result);
        }
        return result;
    }

    public static double manhattan(
            double[] a, int aOffset, double[] b, int bOffset, int length) {
        DoubleVector sum = DoubleVector.zero(SPECIES);
        int index = 0;
        int upper = SPECIES.loopBound(length);
        for (; index < upper; index += SPECIES.length()) {
            DoubleVector delta = DoubleVector.fromArray(SPECIES, a, aOffset + index)
                    .sub(DoubleVector.fromArray(SPECIES, b, bOffset + index));
            sum = sum.add(delta.abs());
        }
        double result = sum.reduceLanes(VectorOperators.ADD);
        for (; index < length; index++) {
            result += Math.abs(a[aOffset + index] - b[bOffset + index]);
        }
        return result;
    }
}
