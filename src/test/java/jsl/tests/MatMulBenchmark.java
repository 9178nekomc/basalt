package jsl.tests;

import jsl.core.MatMul;
import jsl.core.Matrix;
import jsl.core.VectorMath;
import java.util.Locale;
import java.util.Random;

/** Lightweight zero-dependency benchmark; use JMH externally for rigorous publication numbers. */
public final class MatMulBenchmark {
    private MatMulBenchmark() {
    }

    public record Result(int size, int warmup, int iterations,
                         double scalarMillis, double simdParallelMillis,
                         double speedup, double checksum) {
    }

    public static void main(String[] args) {
        int size = args.length > 0 ? Integer.parseInt(args[0]) : 256;
        int warmup = args.length > 1 ? Integer.parseInt(args[1]) : 3;
        int iterations = args.length > 2 ? Integer.parseInt(args[2]) : 5;
        Result result = benchmark(size, warmup, iterations);
        System.out.printf(Locale.ROOT,
                "Matrix %dx%d | vector lanes=%d | warmup=%d | iterations=%d%n",
                result.size(), result.size(), VectorMath.speciesLength(), result.warmup(), result.iterations());
        System.out.printf(Locale.ROOT, "Scalar for-loop:     %.3f ms/op%n", result.scalarMillis());
        System.out.printf(Locale.ROOT, "SIMD + ForkJoin:     %.3f ms/op%n", result.simdParallelMillis());
        System.out.printf(Locale.ROOT, "Measured speedup:    %.2fx%n", result.speedup());
        System.out.printf(Locale.ROOT, "Result checksum:     %.6f%n", result.checksum());
    }

    public static Result benchmark(int size, int warmup, int iterations) {
        if (size <= 0 || warmup < 0 || iterations <= 0) {
            throw new IllegalArgumentException("Require size > 0, warmup >= 0 and iterations > 0");
        }
        double[] leftValues = new double[size * size];
        double[] rightValues = new double[size * size];
        Random random = new Random(1234);
        for (int i = 0; i < leftValues.length; i++) {
            leftValues[i] = random.nextDouble() * 2.0 - 1.0;
            rightValues[i] = random.nextDouble() * 2.0 - 1.0;
        }
        Matrix left = new Matrix(size, size, leftValues);
        Matrix right = new Matrix(size, size, rightValues);
        Matrix scalarResult = null;
        Matrix simdResult = null;
        try (MatMul engine = MatMul.create()) {
            for (int i = 0; i < warmup; i++) {
                scalarResult = MatMul.scalar(left, right);
                simdResult = engine.multiply(left, right);
            }
            long scalarStart = System.nanoTime();
            for (int i = 0; i < iterations; i++) {
                scalarResult = MatMul.scalar(left, right);
            }
            double scalarMillis = (System.nanoTime() - scalarStart) / 1e6 / iterations;

            long simdStart = System.nanoTime();
            for (int i = 0; i < iterations; i++) {
                simdResult = engine.multiply(left, right);
            }
            double simdMillis = (System.nanoTime() - simdStart) / 1e6 / iterations;

            double[] expected = scalarResult.data();
            double[] actual = simdResult.data();
            double checksum = 0.0;
            for (int i = 0; i < expected.length; i++) {
                if (Math.abs(expected[i] - actual[i]) > 1e-9 * size) {
                    throw new AssertionError("Benchmark implementations disagree at " + i);
                }
                checksum += actual[i];
            }
            return new Result(size, warmup, iterations, scalarMillis, simdMillis,
                    scalarMillis / simdMillis, checksum);
        }
    }
}
