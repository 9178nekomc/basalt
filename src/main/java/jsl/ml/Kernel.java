package jsl.ml;

import jsl.core.Matrix;
import jsl.core.ModelFormatException;
import jsl.core.ModelReader;
import jsl.core.ModelWriter;
import jsl.core.VectorMath;
import java.util.Objects;

/**
 * Positive semi-definite kernel functions shared by {@link SVC}, {@link KernelRidge}, and
 * {@link KernelPCA}:
 * <ul>
 *   <li>{@link Type#LINEAR}: {@code K(x, y) = x · y}</li>
 *   <li>{@link Type#RBF}: {@code K(x, y) = exp(-gamma * ||x - y||^2)}</li>
 *   <li>{@link Type#POLY}: {@code K(x, y) = (gamma * (x · y) + coef0)^degree}</li>
 *   <li>{@link Type#SIGMOID}: {@code K(x, y) = tanh(gamma * (x · y) + coef0)}</li>
 * </ul>
 */
public final class Kernel {
    public enum Type {
        LINEAR,
        RBF,
        POLY,
        SIGMOID
    }

    private final Type type;
    private final double gamma;
    private final int degree;
    private final double coef0;

    public Kernel(Type type, double gamma, int degree, double coef0) {
        this.type = Objects.requireNonNull(type, "type");
        if (!Double.isFinite(gamma) || gamma < 0.0) {
            throw new IllegalArgumentException("gamma must be finite and non-negative");
        }
        if (degree < 1) {
            throw new IllegalArgumentException("degree must be >= 1");
        }
        if (!Double.isFinite(coef0)) {
            throw new IllegalArgumentException("coef0 must be finite");
        }
        this.gamma = gamma;
        this.degree = degree;
        this.coef0 = coef0;
    }

    public static Kernel linear() {
        return new Kernel(Type.LINEAR, 1.0, 3, 0.0);
    }

    public static Kernel rbf(double gamma) {
        MlChecks.positiveFinite(gamma, "gamma");
        return new Kernel(Type.RBF, gamma, 3, 0.0);
    }

    public static Kernel poly(int degree, double gamma, double coef0) {
        MlChecks.positiveFinite(gamma, "gamma");
        return new Kernel(Type.POLY, gamma, degree, coef0);
    }

    public static Kernel sigmoid(double gamma, double coef0) {
        MlChecks.positiveFinite(gamma, "gamma");
        return new Kernel(Type.SIGMOID, gamma, 3, coef0);
    }

    public Type type() {
        return type;
    }

    public double gamma() {
        return gamma;
    }

    public int degree() {
        return degree;
    }

    public double coef0() {
        return coef0;
    }

    public double evaluate(double[] x, int xOffset, double[] y, int yOffset, int length) {
        double g = gamma == 0.0 ? 1.0 / Math.max(1, length) : gamma;
        return switch (type) {
            case LINEAR -> VectorMath.dot(x, xOffset, y, yOffset, length);
            case RBF -> Math.exp(-g * VectorMath.squaredEuclidean(x, xOffset, y, yOffset, length));
            case POLY -> Math.pow(g * VectorMath.dot(x, xOffset, y, yOffset, length) + coef0, degree);
            case SIGMOID -> Math.tanh(g * VectorMath.dot(x, xOffset, y, yOffset, length) + coef0);
        };
    }

    public Matrix gram(Matrix a, Matrix b) {
        if (a.columns() != b.columns()) {
            throw new IllegalArgumentException("Kernel Gram matrix requires equal feature counts");
        }
        int rowsA = a.rows();
        int rowsB = b.rows();
        int cols = a.columns();
        double[] dataA = a.data();
        double[] dataB = b.data();
        double[] out = new double[rowsA * rowsB];
        for (int i = 0; i < rowsA; i++) {
            int offA = i * cols;
            int rowBase = i * rowsB;
            for (int j = 0; j < rowsB; j++) {
                out[rowBase + j] = evaluate(dataA, offA, dataB, j * cols, cols);
            }
        }
        return new Matrix(rowsA, rowsB, out);
    }

    public void write(ModelWriter writer) {
        writer.putInt(type.ordinal());
        writer.putDouble(gamma);
        writer.putInt(degree);
        writer.putDouble(coef0);
    }

    public static Kernel read(ModelReader reader) {
        int ord = reader.getInt();
        if (ord < 0 || ord >= Type.values().length) {
            throw new ModelFormatException("Invalid Kernel type ordinal");
        }
        double gamma = reader.getDouble();
        int degree = reader.getInt();
        double coef0 = reader.getDouble();
        return new Kernel(Type.values()[ord], gamma, degree, coef0);
    }
}
