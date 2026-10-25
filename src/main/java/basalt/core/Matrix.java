package basalt.core;

import java.util.Objects;

/** A mutable, row-major matrix backed by one contiguous double array. */
public final class Matrix {
    private final int rows;
    private final int columns;
    private final double[] data;

    public Matrix(int rows, int columns, double[] data) {
        this(rows, columns, validateAndCopy(rows, columns, data), false);
    }

    private Matrix(int rows, int columns, double[] data, boolean ignored) {
        this.rows = rows;
        this.columns = columns;
        this.data = data;
    }

    private static double[] validateAndCopy(int rows, int columns, double[] data) {
        validateDimensions(rows, columns);
        Objects.requireNonNull(data, "data");
        int size = checkedSize(rows, columns);
        if (data.length != size) {
            throw new IllegalArgumentException(
                    "Expected " + size + " values but got " + data.length);
        }
        return data.clone();
    }

    public static Matrix zeros(int rows, int columns) {
        validateDimensions(rows, columns);
        return new Matrix(rows, columns, new double[checkedSize(rows, columns)], false);
    }

    /** Takes ownership of {@code data} without copying; used by native kernels. */
    static Matrix wrap(int rows, int columns, double[] data) {
        validateDimensions(rows, columns);
        if (data.length != checkedSize(rows, columns)) {
            throw new IllegalArgumentException("Matrix data length mismatch");
        }
        return new Matrix(rows, columns, data, false);
    }

    public static Matrix fromRows(double[][] rows) {
        Objects.requireNonNull(rows, "rows");
        if (rows.length == 0 || rows[0].length == 0) {
            throw new IllegalArgumentException("Matrix must not be empty");
        }
        int columns = rows[0].length;
        double[] data = new double[checkedSize(rows.length, columns)];
        for (int row = 0; row < rows.length; row++) {
            Objects.requireNonNull(rows[row], "rows[" + row + "]");
            if (rows[row].length != columns) {
                throw new IllegalArgumentException("All rows must have equal length");
            }
            System.arraycopy(rows[row], 0, data, row * columns, columns);
        }
        return new Matrix(rows.length, columns, data, false);
    }

    private static void validateDimensions(int rows, int columns) {
        if (rows <= 0 || columns <= 0) {
            throw new IllegalArgumentException("Matrix dimensions must be positive");
        }
    }

    private static int checkedSize(int rows, int columns) {
        long size = (long) rows * columns;
        if (size > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Matrix is too large for a contiguous Java array");
        }
        return (int) size;
    }

    public int rows() {
        return rows;
    }

    public int columns() {
        return columns;
    }

    public int size() {
        return data.length;
    }

    public double get(int row, int column) {
        return data[index(row, column)];
    }

    public void set(int row, int column, double value) {
        data[index(row, column)] = value;
    }

    public int index(int row, int column) {
        if (row < 0 || row >= rows || column < 0 || column >= columns) {
            throw new IndexOutOfBoundsException(
                    "Index (" + row + ", " + column + ") outside " + rows + "x" + columns);
        }
        return row * columns + column;
    }

    /**
     * Returns the live contiguous storage for numerical kernels. Callers must not resize it;
     * use {@link #toArray()} when a defensive copy is required.
     */
    public double[] data() {
        return data;
    }

    public double[] toArray() {
        return data.clone();
    }

    public Matrix copy() {
        return new Matrix(rows, columns, data.clone(), false);
    }

    public Matrix addInPlace(Matrix other) {
        requireSameShape(other);
        for (int i = 0; i < data.length; i++) {
            data[i] += other.data[i];
        }
        return this;
    }

    public Matrix scaleInPlace(double scalar) {
        for (int i = 0; i < data.length; i++) {
            data[i] *= scalar;
        }
        return this;
    }

    public Matrix matMul(Matrix other) {
        return MatMul.commonPool().multiply(this, other);
    }

    /** Returns a transposed copy with cache-friendly blocked traversal. */
    public Matrix transpose() {
        double[] target = new double[data.length];
        int block = 32;
        for (int rowBlock = 0; rowBlock < rows; rowBlock += block) {
            int rowEnd = Math.min(rows, rowBlock + block);
            for (int columnBlock = 0; columnBlock < columns; columnBlock += block) {
                int columnEnd = Math.min(columns, columnBlock + block);
                for (int row = rowBlock; row < rowEnd; row++) {
                    int sourceOffset = row * columns;
                    for (int column = columnBlock; column < columnEnd; column++) {
                        target[column * rows + row] = data[sourceOffset + column];
                    }
                }
            }
        }
        return new Matrix(columns, rows, target, false);
    }

    /** Copies one row into {@code target}, which must hold at least {@code columns} values. */
    public void copyRow(int row, double[] target, int offset) {
        if (row < 0 || row >= rows) {
            throw new IndexOutOfBoundsException("Row " + row + " outside " + rows);
        }
        System.arraycopy(data, row * columns, target, offset, columns);
    }

    private void requireSameShape(Matrix other) {
        Objects.requireNonNull(other, "other");
        if (rows != other.rows || columns != other.columns) {
            throw new IllegalArgumentException("Matrix shapes differ");
        }
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder("Matrix[").append(rows).append('x').append(columns).append("] ");
        int limit = Math.min(rows, 8);
        for (int row = 0; row < limit; row++) {
            result.append(System.lineSeparator()).append('[');
            for (int column = 0; column < columns; column++) {
                if (column > 0) {
                    result.append(", ");
                }
                result.append(get(row, column));
            }
            result.append(']');
        }
        if (rows > limit) {
            result.append(System.lineSeparator()).append("...");
        }
        return result.toString();
    }
}
