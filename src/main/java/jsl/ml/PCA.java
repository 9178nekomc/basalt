package jsl.ml;

import jsl.core.Matrix;
import jsl.core.ModelFormatException;
import jsl.core.ModelKind;
import jsl.core.ModelReader;
import jsl.core.ModelWriter;
import jsl.core.Persistable;
import jsl.core.VectorMath;
import java.util.Arrays;
import java.util.Objects;

/**
 * Centered PCA with two exact eigensolvers.
 *
 * <ul>
 *   <li>{@link EigenSolver#JACOBI} diagonalizes the {@code p x p} covariance matrix. Cost is
 *       {@code O(n p^2)} for the covariance plus cyclic Jacobi sweeps, ideal when {@code n >> p}.</li>
 *   <li>{@link EigenSolver#GRAM} diagonalizes the {@code n x n} Gram matrix and recovers principal
 *       directions as {@code v = X^T u / ||X^T u||}. Cost is {@code O(n^2 p)}, ideal when
 *       {@code p >> n} (the classic dual/PCA trick that avoids a huge covariance matrix).</li>
 * </ul>
 *
 * <p>Both paths return orthonormal components and variances on the covariance scale, so their
 * outputs agree up to sign and numerical noise. Whitening divides each projection by the component
 * standard deviation.</p>
 */
public final class PCA implements Estimator, Persistable {
    private static final int GRAM_ROW_LIMIT = 20_000;

    private final int componentCount;
    private final EigenSolver solver;
    private final int maxIterations;
    private final double tolerance;
    private final boolean whiten;
    private double[] means;
    private double[] components;
    private double[] eigenvalues;
    private double totalVariance;
    private int featureCount;
    private EigenSolver resolvedSolver;

    private PCA(Builder builder) {
        componentCount = builder.components;
        solver = builder.solver;
        maxIterations = builder.maxIterations;
        tolerance = builder.tolerance;
        whiten = builder.whiten;
    }

    public static Builder builder() {
        return new Builder();
    }

    public PCA fit(Matrix features) {
        return fit(features, null);
    }

    @Override
    public PCA fit(Matrix features, double[] ignoredTargets) {
        MlChecks.features(features);
        int rows = features.rows();
        if (rows < 2) {
            throw new IllegalArgumentException("PCA covariance requires at least two samples");
        }
        featureCount = features.columns();
        if (componentCount > Math.min(featureCount, rows)) {
            throw new IllegalArgumentException("components cannot exceed min(samples, features)");
        }
        resolvedSolver = resolveSolver(rows);
        means = new double[featureCount];
        double[] data = features.data();
        for (int row = 0; row < rows; row++) {
            int offset = row * featureCount;
            for (int feature = 0; feature < featureCount; feature++) {
                means[feature] += data[offset + feature];
            }
        }
        for (int feature = 0; feature < featureCount; feature++) {
            means[feature] /= rows;
        }
        if (resolvedSolver == EigenSolver.GRAM) {
            fitGram(data, rows);
        } else {
            fitCovariance(data, rows);
        }
        return this;
    }

    private EigenSolver resolveSolver(int rows) {
        if (solver != EigenSolver.AUTO) {
            if (solver == EigenSolver.GRAM && rows > GRAM_ROW_LIMIT) {
                throw new IllegalArgumentException(
                        "GRAM solver is limited to " + GRAM_ROW_LIMIT + " samples");
            }
            return solver;
        }
        if (featureCount > rows && rows <= GRAM_ROW_LIMIT) {
            return EigenSolver.GRAM;
        }
        return EigenSolver.JACOBI;
    }

    private void fitCovariance(double[] data, int rows) {
        double[] covariance = new double[featureCount * featureCount];
        for (int row = 0; row < rows; row++) {
            int offset = row * featureCount;
            for (int i = 0; i < featureCount; i++) {
                double left = data[offset + i] - means[i];
                int base = i * featureCount;
                for (int j = i; j < featureCount; j++) {
                    covariance[base + j] += left * (data[offset + j] - means[j]);
                }
            }
        }
        double denominator = rows - 1;
        totalVariance = 0.0;
        for (int i = 0; i < featureCount; i++) {
            totalVariance += covariance[i * featureCount + i] / denominator;
            for (int j = i; j < featureCount; j++) {
                double value = covariance[i * featureCount + j] / denominator;
                covariance[i * featureCount + j] = value;
                covariance[j * featureCount + i] = value;
            }
        }
        EigenDecomposition decomposition = jacobi(covariance, featureCount);
        int[] order = descendingOrder(decomposition.eigenvalues);
        eigenvalues = new double[componentCount];
        components = new double[componentCount * featureCount];
        for (int component = 0; component < componentCount; component++) {
            int eigenIndex = order[component];
            eigenvalues[component] = Math.max(0.0, decomposition.eigenvalues[eigenIndex]);
            copyComponent(decomposition.eigenvectors, featureCount, eigenIndex, component);
        }
    }

    private void fitGram(double[] data, int rows) {
        double[] centered = new double[rows * featureCount];
        for (int row = 0; row < rows; row++) {
            int offset = row * featureCount;
            for (int feature = 0; feature < featureCount; feature++) {
                centered[offset + feature] = data[offset + feature] - means[feature];
            }
        }
        double[] gram = new double[rows * rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j <= i; j++) {
                double value = VectorMath.dot(centered, i * featureCount,
                        centered, j * featureCount, featureCount);
                gram[i * rows + j] = value;
                gram[j * rows + i] = value;
            }
        }
        totalVariance = 0.0;
        for (int feature = 0; feature < featureCount; feature++) {
            double sum = 0.0;
            for (int row = 0; row < rows; row++) {
                double value = centered[row * featureCount + feature];
                sum += value * value;
            }
            totalVariance += sum / (rows - 1);
        }
        EigenDecomposition decomposition = jacobi(gram, rows);
        int[] order = descendingOrder(decomposition.eigenvalues);
        eigenvalues = new double[componentCount];
        components = new double[componentCount * featureCount];
        for (int component = 0; component < componentCount; component++) {
            int eigenIndex = order[component];
            double gramEigenvalue = Math.max(0.0, decomposition.eigenvalues[eigenIndex]);
            eigenvalues[component] = gramEigenvalue / (rows - 1);
            double norm = Math.sqrt(gramEigenvalue);
            if (norm <= 1e-12) {
                continue;
            }
            int base = component * featureCount;
            for (int feature = 0; feature < featureCount; feature++) {
                double sum = 0.0;
                for (int row = 0; row < rows; row++) {
                    sum += centered[row * featureCount + feature]
                            * decomposition.eigenvectors[row * rows + eigenIndex];
                }
                components[base + feature] = sum / norm;
            }
            normalizeComponent(base);
        }
    }

    private void copyComponent(double[] eigenvectors, int dimension, int eigenIndex, int component) {
        int base = component * featureCount;
        for (int feature = 0; feature < featureCount; feature++) {
            components[base + feature] = eigenvectors[feature * dimension + eigenIndex];
        }
        normalizeComponent(base);
    }

    /** Normalizes to unit length and fixes the sign so the largest-magnitude entry is positive. */
    private void normalizeComponent(int base) {
        double norm = 0.0;
        for (int feature = 0; feature < featureCount; feature++) {
            norm += components[base + feature] * components[base + feature];
        }
        norm = Math.sqrt(norm);
        if (!(norm > 0.0)) {
            Arrays.fill(components, base, base + featureCount, 0.0);
            return;
        }
        int pivot = 0;
        for (int feature = 1; feature < featureCount; feature++) {
            if (Math.abs(components[base + feature]) > Math.abs(components[base + pivot])) {
                pivot = feature;
            }
        }
        double sign = components[base + pivot] < 0.0 ? -1.0 : 1.0;
        double factor = sign / norm;
        for (int feature = 0; feature < featureCount; feature++) {
            components[base + feature] *= factor;
        }
    }

    /**
     * Cyclic Jacobi diagonalization. Each sweep annihilates every off-diagonal pair once; the
     * relative off-diagonal norm is the convergence criterion, which makes the routine scale
     * invariant.
     */
    private EigenDecomposition jacobi(double[] matrix, int dimension) {
        double[] eigenvectors = new double[dimension * dimension];
        for (int i = 0; i < dimension; i++) {
            eigenvectors[i * dimension + i] = 1.0;
        }
        double frobenius = 0.0;
        for (double value : matrix) {
            frobenius += value * value;
        }
        frobenius = Math.sqrt(frobenius);
        double threshold = tolerance * Math.max(1.0, frobenius);
        for (int sweep = 0; sweep < maxIterations; sweep++) {
            double offDiagonal = 0.0;
            for (int p = 0; p < dimension - 1; p++) {
                for (int q = p + 1; q < dimension; q++) {
                    double apq = matrix[p * dimension + q];
                    offDiagonal += apq * apq;
                    if (apq == 0.0) {
                        continue;
                    }
                    double app = matrix[p * dimension + p];
                    double aqq = matrix[q * dimension + q];
                    double angle = 0.5 * Math.atan2(2.0 * apq, aqq - app);
                    double cosine = Math.cos(angle);
                    double sine = Math.sin(angle);
                    matrix[p * dimension + p] = cosine * cosine * app
                            - 2.0 * sine * cosine * apq + sine * sine * aqq;
                    matrix[q * dimension + q] = sine * sine * app
                            + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
                    matrix[p * dimension + q] = 0.0;
                    matrix[q * dimension + p] = 0.0;
                    for (int k = 0; k < dimension; k++) {
                        if (k == p || k == q) {
                            continue;
                        }
                        double akp = matrix[k * dimension + p];
                        double akq = matrix[k * dimension + q];
                        double newKp = cosine * akp - sine * akq;
                        double newKq = sine * akp + cosine * akq;
                        matrix[k * dimension + p] = newKp;
                        matrix[p * dimension + k] = newKp;
                        matrix[k * dimension + q] = newKq;
                        matrix[q * dimension + k] = newKq;
                    }
                    rotateEigenvectors(eigenvectors, dimension, p, q, cosine, sine);
                }
            }
            if (Math.sqrt(offDiagonal) <= threshold) {
                break;
            }
        }
        double[] eigenvalues = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            eigenvalues[i] = matrix[i * dimension + i];
        }
        return new EigenDecomposition(eigenvalues, eigenvectors);
    }

    /** Applies {@code V <- V * J(p,q,theta)} to every row of the eigenvector matrix. */
    private static void rotateEigenvectors(double[] eigenvectors, int dimension,
                                           int p, int q, double cosine, double sine) {
        for (int row = 0; row < dimension; row++) {
            int base = row * dimension;
            double vp = eigenvectors[base + p];
            double vq = eigenvectors[base + q];
            eigenvectors[base + p] = cosine * vp - sine * vq;
            eigenvectors[base + q] = sine * vp + cosine * vq;
        }
    }

    private static int[] descendingOrder(double[] values) {
        int[] order = new int[values.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        for (int i = 1; i < order.length; i++) {
            int selected = order[i];
            int position = i;
            while (position > 0 && values[order[position - 1]] < values[selected]) {
                order[position] = order[position - 1];
                position--;
            }
            order[position] = selected;
        }
        return order;
    }

    public Matrix transform(Matrix features) {
        MlChecks.predictionData(features, featureCount, components != null);
        Matrix result = Matrix.zeros(features.rows(), componentCount);
        double[] centered = new double[featureCount];
        for (int row = 0; row < features.rows(); row++) {
            for (int feature = 0; feature < featureCount; feature++) {
                centered[feature] = features.get(row, feature) - means[feature];
            }
            for (int component = 0; component < componentCount; component++) {
                double value = VectorMath.dot(components, component * featureCount,
                        centered, 0, featureCount);
                if (whiten) {
                    double deviation = Math.sqrt(Math.max(eigenvalues[component], 1e-12));
                    value /= deviation;
                }
                result.set(row, component, value);
            }
        }
        return result;
    }

    public Matrix inverseTransform(Matrix transformed) {
        if (components == null) {
            throw new IllegalStateException("PCA must be fitted first");
        }
        MlChecks.features(transformed);
        if (transformed.columns() != componentCount) {
            throw new IllegalArgumentException("Transformed feature count mismatch");
        }
        Matrix result = Matrix.zeros(transformed.rows(), featureCount);
        for (int row = 0; row < transformed.rows(); row++) {
            for (int feature = 0; feature < featureCount; feature++) {
                double value = means[feature];
                for (int component = 0; component < componentCount; component++) {
                    double score = transformed.get(row, component);
                    if (whiten) {
                        score *= Math.sqrt(Math.max(eigenvalues[component], 1e-12));
                    }
                    value += score * components[component * featureCount + feature];
                }
                result.set(row, feature, value);
            }
        }
        return result;
    }

    @Override
    public double[] predict(Matrix features) {
        return transform(features).toArray();
    }

    public double[] explainedVariance() {
        return eigenvalues.clone();
    }

    public double[] explainedVarianceRatio() {
        double[] ratio = new double[componentCount];
        if (totalVariance > 0.0) {
            for (int i = 0; i < ratio.length; i++) {
                ratio[i] = eigenvalues[i] / totalVariance;
            }
        }
        return ratio;
    }

    public Matrix components() {
        if (components == null) {
            throw new IllegalStateException("PCA must be fitted first");
        }
        return new Matrix(componentCount, featureCount, components);
    }

    public double[] means() {
        return means.clone();
    }

    public EigenSolver solver() {
        return resolvedSolver;
    }

    @Override
    public ModelKind kind() {
        return ModelKind.PCA;
    }

    @Override
    public void write(ModelWriter writer) {
        if (components == null) {
            throw new IllegalStateException("PCA must be fitted before serialization");
        }
        writer.putInt(componentCount);
        writer.putInt(featureCount);
        writer.putByte(whiten ? 1 : 0);
        writer.putDouble(totalVariance);
        writer.putDoubles(means);
        writer.putDoubles(eigenvalues);
        writer.putDoubles(components);
    }

    public static PCA read(ModelReader reader) {
        int componentCount = reader.getInt();
        int features = reader.getInt();
        int whitenFlag = reader.getByte();
        double totalVariance = reader.getDouble();
        if (componentCount <= 0 || features <= 0 || (whitenFlag != 0 && whitenFlag != 1)) {
            throw new ModelFormatException("Invalid PCA header");
        }
        PCA model = new PCA(new Builder().components(componentCount).whiten(whitenFlag == 1));
        model.featureCount = features;
        model.totalVariance = totalVariance;
        model.means = reader.getDoubles(features);
        model.eigenvalues = reader.getDoubles(componentCount);
        model.components = reader.getDoubles(Math.multiplyExact(componentCount, features));
        model.resolvedSolver = EigenSolver.JACOBI;
        return model;
    }

    private record EigenDecomposition(double[] eigenvalues, double[] eigenvectors) {
    }

    public static final class Builder {
        private int components = 2;
        private EigenSolver solver = EigenSolver.AUTO;
        private int maxIterations = 60;
        private double tolerance = 1e-12;
        private boolean whiten;

        public Builder components(int components) {
            this.components = components;
            return this;
        }

        public Builder solver(EigenSolver solver) {
            this.solver = Objects.requireNonNull(solver, "solver");
            return this;
        }

        public Builder maxIterations(int maxIterations) {
            this.maxIterations = maxIterations;
            return this;
        }

        public Builder tolerance(double tolerance) {
            this.tolerance = tolerance;
            return this;
        }

        public Builder whiten(boolean whiten) {
            this.whiten = whiten;
            return this;
        }

        public PCA build() {
            if (components <= 0 || maxIterations <= 0) {
                throw new IllegalArgumentException("components and maxIterations must be positive");
            }
            MlChecks.positiveFinite(tolerance, "tolerance");
            return new PCA(this);
        }
    }
}
