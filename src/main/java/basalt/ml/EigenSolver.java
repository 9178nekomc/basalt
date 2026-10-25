package basalt.ml;

/** Available symmetric eigensolvers for PCA. */
public enum EigenSolver {
    /** Picks {@link #GRAM} when {@code features > samples}, otherwise {@link #JACOBI}. */
    AUTO,
    /** Diagonalizes the {@code p x p} covariance matrix; {@code O(n p^2)}. */
    JACOBI,
    /** Diagonalizes the {@code n x n} Gram matrix; {@code O(n^2 p)}, for wide data. */
    GRAM
}
