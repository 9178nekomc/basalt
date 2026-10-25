package jsl.tests;

import jsl.core.Matrix;
import jsl.ml.Estimator;
import jsl.ml.PCA;

/** This file must not compile: a PCA model is not a LinearRegression. */
final class InvalidEstimatorUsage {
    void reject() {
        PCA pca = PCA.builder().components(1).build().fit(Matrix.zeros(2, 2));
        jsl.ml.LinearRegression regression = pca;
        Estimator estimator = regression;
        estimator.predict(Matrix.zeros(1, 1));
    }
}
