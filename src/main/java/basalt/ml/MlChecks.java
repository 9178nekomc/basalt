package basalt.ml;

import basalt.core.Matrix;
import java.util.Objects;

final class MlChecks {
    private MlChecks() {
    }

    static void features(Matrix features) {
        Objects.requireNonNull(features, "features");
        finite(features.data(), "features");
    }

    static void trainingData(Matrix features, double[] targets) {
        Objects.requireNonNull(features, "features");
        Objects.requireNonNull(targets, "targets");
        if (features.rows() != targets.length) {
            throw new IllegalArgumentException("Target count must equal training row count");
        }
        finite(features.data(), "features");
        finite(targets, "targets");
    }

    static void predictionData(Matrix features, int expectedFeatures, boolean fitted) {
        Objects.requireNonNull(features, "features");
        if (!fitted) {
            throw new IllegalStateException("Estimator must be fitted before prediction");
        }
        if (features.columns() != expectedFeatures) {
            throw new IllegalArgumentException(
                    "Expected " + expectedFeatures + " features but got " + features.columns());
        }
        finite(features.data(), "features");
    }

    static void finite(double[] values, String name) {
        for (int i = 0; i < values.length; i++) {
            if (!Double.isFinite(values[i])) {
                throw new IllegalArgumentException(name + "[" + i + "] must be finite");
            }
        }
    }

    static void positiveFinite(double value, String name) {
        if (!Double.isFinite(value) || value <= 0.0) {
            throw new IllegalArgumentException(name + " must be positive and finite");
        }
    }

    static void nonNegativeFinite(double value, String name) {
        if (!Double.isFinite(value) || value < 0.0) {
            throw new IllegalArgumentException(name + " must be non-negative and finite");
        }
    }
}
