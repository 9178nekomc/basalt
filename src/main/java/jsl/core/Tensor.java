package jsl.core;

import java.util.Arrays;
import java.util.Objects;

/** A mutable, contiguous tensor whose layout is checked by the Java type system. */
public final class Tensor<S extends Layout> {
    private final int[] shape;
    private final double[] data;

    private Tensor(int[] shape, double[] data, boolean copy) {
        this.shape = shape.clone();
        this.data = copy ? data.clone() : data;
    }

    public static Tensor<BCHW> ofBCHW(
            int batch, int channels, int height, int width, double... data) {
        return create(new int[]{batch, channels, height, width}, data);
    }

    public static Tensor<BHW> ofBHW(int batch, int height, int width, double... data) {
        return create(new int[]{batch, height, width}, data);
    }

    public static Tensor<BC> ofBC(int batch, int channels, double... data) {
        return create(new int[]{batch, channels}, data);
    }

    public static Tensor<BSF> ofBSF(int batch, int sequence, int features, double... data) {
        return create(new int[]{batch, sequence, features}, data);
    }

    public static Tensor<BCHW> zerosBCHW(int batch, int channels, int height, int width) {
        int[] shape = {batch, channels, height, width};
        return new Tensor<>(shape, new double[checkedSize(shape)], false);
    }

    public static Tensor<BC> zerosBC(int batch, int channels) {
        int[] shape = {batch, channels};
        return new Tensor<>(shape, new double[checkedSize(shape)], false);
    }

    public static Tensor<BSF> zerosBSF(int batch, int sequence, int features) {
        int[] shape = {batch, sequence, features};
        return new Tensor<>(shape, new double[checkedSize(shape)], false);
    }

    private static <S extends Layout> Tensor<S> create(int[] shape, double[] data) {
        Objects.requireNonNull(data, "data");
        int size = checkedSize(shape);
        if (data.length != size) {
            throw new IllegalArgumentException(
                    "Tensor shape requires " + size + " values but got " + data.length);
        }
        return new Tensor<>(shape, data, true);
    }

    private static int checkedSize(int[] shape) {
        Objects.requireNonNull(shape, "shape");
        if (shape.length == 0) {
            throw new IllegalArgumentException("Tensor rank must be positive");
        }
        long size = 1;
        for (int dimension : shape) {
            if (dimension <= 0) {
                throw new IllegalArgumentException("Tensor dimensions must be positive: "
                        + Arrays.toString(shape));
            }
            size *= dimension;
            if (size > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Tensor is too large for a contiguous Java array");
            }
        }
        return (int) size;
    }

    public int rank() {
        return shape.length;
    }

    public int dimension(int axis) {
        return shape[axis];
    }

    public int[] shape() {
        return shape.clone();
    }

    public int size() {
        return data.length;
    }

    public int index(int... coordinates) {
        Objects.requireNonNull(coordinates, "coordinates");
        if (coordinates.length != shape.length) {
            throw new IllegalArgumentException(
                    "Expected " + shape.length + " coordinates but got " + coordinates.length);
        }
        int index = 0;
        for (int axis = 0; axis < shape.length; axis++) {
            int coordinate = coordinates[axis];
            if (coordinate < 0 || coordinate >= shape[axis]) {
                throw new IndexOutOfBoundsException(
                        "Coordinate " + coordinate + " outside axis " + axis
                                + " with length " + shape[axis]);
            }
            index = index * shape[axis] + coordinate;
        }
        return index;
    }

    public double get(int... coordinates) {
        return data[index(coordinates)];
    }

    public void set(double value, int... coordinates) {
        data[index(coordinates)] = value;
    }

    /** Live storage for kernels; use {@link #toArray()} for a defensive copy. */
    public double[] data() {
        return data;
    }

    public double[] toArray() {
        return data.clone();
    }

    public Tensor<S> copy() {
        return new Tensor<>(shape, data, true);
    }

    /**
     * Copies a contiguous batch range. All supported layouts place the batch axis first, so the
     * slice is a single {@link System#arraycopy}.
     */
    public Tensor<S> sliceBatch(int fromBatch, int toBatch) {
        if (fromBatch < 0 || toBatch > shape[0] || fromBatch >= toBatch) {
            throw new IllegalArgumentException(
                    "Invalid batch slice [" + fromBatch + ", " + toBatch + ") for " + shape[0]);
        }
        int perBatch = data.length / shape[0];
        double[] slice = new double[(toBatch - fromBatch) * perBatch];
        System.arraycopy(data, fromBatch * perBatch, slice, 0, slice.length);
        int[] slicedShape = shape.clone();
        slicedShape[0] = toBatch - fromBatch;
        return new Tensor<>(slicedShape, slice, false);
    }

    /** Fills every element with {@code value}. */
    public Tensor<S> fill(double value) {
        Arrays.fill(data, value);
        return this;
    }
}
