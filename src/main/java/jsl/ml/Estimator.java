package jsl.ml;

import jsl.core.Matrix;

/** Common contract for classical estimators. Implementations are mutable during fit. */
public interface Estimator {
    Estimator fit(Matrix features, double[] targets);

    double[] predict(Matrix features);
}
