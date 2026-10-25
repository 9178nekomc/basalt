package jsl.core;

import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

/** Scalar reference and blocked parallel SIMD matrix multiplication. */
public final class MatMul implements AutoCloseable {
    private static final int ROW_BLOCK = 8;
    private static final int WORK_THRESHOLD = 65_536;
    private static final long NATIVE_THRESHOLD = 100_000_000L;

    private final ForkJoinPool pool;
    private final boolean ownsPool;
    private final int threads;

    private MatMul(ForkJoinPool pool, boolean ownsPool, int threads) {
        this.pool = pool;
        this.ownsPool = ownsPool;
        this.threads = threads;
    }

    public static MatMul create() {
        int threads = Runtime.getRuntime().availableProcessors();
        return new MatMul(new ForkJoinPool(threads), true, threads);
    }

    /** Creates an engine with an explicit parallelism level, used by scaling benchmarks. */
    public static MatMul create(int threads) {
        if (threads <= 0) {
            throw new IllegalArgumentException("threads must be positive");
        }
        return new MatMul(new ForkJoinPool(threads), true, threads);
    }

    public static MatMul commonPool() {
        return new MatMul(ForkJoinPool.commonPool(), false,
                Runtime.getRuntime().availableProcessors());
    }

    /** Straightforward triple-loop implementation used as a correctness and speed baseline. */
    public static Matrix scalar(Matrix left, Matrix right) {
        validateOperands(left, right);
        int rows = left.rows();
        int shared = left.columns();
        int columns = right.columns();
        double[] a = left.data();
        double[] b = right.data();
        double[] output = new double[checkedOutputSize(rows, columns)];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                double sum = 0.0;
                for (int k = 0; k < shared; k++) {
                    sum += a[row * shared + k] * b[k * columns + column];
                }
                output[row * columns + column] = sum;
            }
        }
        return new Matrix(rows, columns, output);
    }

    /**
     * Uses the Rust AVX2/FMA kernel for large products and the blocked Java SIMD path otherwise.
     * A native failure falls back to Java rather than propagating a JNI error.
     */
    public Matrix multiply(Matrix left, Matrix right) {
        validateOperands(left, right);
        int rows = left.rows();
        int shared = left.columns();
        int columns = right.columns();
        long work = (long) rows * columns * shared;
        if (NativeKernels.available() && work >= NATIVE_THRESHOLD) {
            double[] nativeOutput = new double[checkedOutputSize(rows, columns)];
            try {
                NativeKernels.gemmChecked(left.data(), right.data(), nativeOutput,
                        rows, shared, columns, threads);
                return Matrix.wrap(rows, columns, nativeOutput);
            } catch (IllegalStateException failure) {
                // Fall through to the portable implementation.
            }
        }
        double[] transposed = transpose(right);
        double[] output = new double[checkedOutputSize(rows, columns)];
        pool.invoke(new RowTask(left.data(), transposed, output, shared, columns, 0, rows));
        return new Matrix(rows, columns, output);
    }

    private static double[] transpose(Matrix matrix) {
        int rows = matrix.rows();
        int columns = matrix.columns();
        double[] source = matrix.data();
        double[] target = new double[source.length];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                target[column * rows + row] = source[row * columns + column];
            }
        }
        return target;
    }

    private static int checkedOutputSize(int rows, int columns) {
        long size = (long) rows * columns;
        if (size > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Matrix multiplication output exceeds Java array limits");
        }
        return (int) size;
    }

    private static void validateOperands(Matrix left, Matrix right) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");
        if (left.columns() != right.rows()) {
            throw new IllegalArgumentException(
                    "Cannot multiply " + left.rows() + "x" + left.columns()
                            + " by " + right.rows() + "x" + right.columns());
        }
    }

    @Override
    public void close() {
        if (ownsPool) {
            pool.shutdown();
        }
    }

    private static final class RowTask extends RecursiveAction {
        private static final long serialVersionUID = 1L;

        private final double[] left;
        private final double[] rightTransposed;
        private final double[] output;
        private final int shared;
        private final int columns;
        private final int startRow;
        private final int endRow;

        private RowTask(double[] left, double[] rightTransposed, double[] output,
                        int shared, int columns, int startRow, int endRow) {
            this.left = left;
            this.rightTransposed = rightTransposed;
            this.output = output;
            this.shared = shared;
            this.columns = columns;
            this.startRow = startRow;
            this.endRow = endRow;
        }

        @Override
        protected void compute() {
            int count = endRow - startRow;
            long work = (long) count * shared * columns;
            if (count <= ROW_BLOCK || work <= WORK_THRESHOLD) {
                computeBlock();
                return;
            }
            int middle = startRow + count / 2;
            invokeAll(
                    new RowTask(left, rightTransposed, output, shared, columns, startRow, middle),
                    new RowTask(left, rightTransposed, output, shared, columns, middle, endRow));
        }

        private void computeBlock() {
            for (int row = startRow; row < endRow; row++) {
                int leftOffset = row * shared;
                int outputOffset = row * columns;
                for (int column = 0; column < columns; column++) {
                    output[outputOffset + column] = VectorMath.dot(
                            left, leftOffset, rightTransposed, column * shared, shared);
                }
            }
        }
    }
}
