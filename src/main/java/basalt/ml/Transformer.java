package basalt.ml;

import basalt.core.Matrix;
import basalt.core.Persistable;

/**
 * Common contract for stateful feature preprocessors and feature-engineering steps.
 *
 * <p>Every {@code Transformer} is {@link Persistable} so training-time statistics (means, scales,
 * category vocabularies, imputation constants, bin edges, selected feature masks) can be serialized
 * into a {@code .bsl} container or inside a {@link Pipeline}, guaranteeing bit-identical behavior
 * between training and production inference.</p>
 */
public interface Transformer extends Persistable {
    default Transformer fit(Matrix features) {
        return fit(features, null);
    }

    Transformer fit(Matrix features, double[] targets);

    Matrix transform(Matrix features);

    default Matrix fitTransform(Matrix features) {
        return fit(features, null).transform(features);
    }

    default Matrix fitTransform(Matrix features, double[] targets) {
        return fit(features, targets).transform(features);
    }

    default Matrix inverseTransform(Matrix transformed) {
        throw new basalt.BasaltUnsupportedOperationException(
                getClass().getSimpleName() + " does not support inverseTransform");
    }
}
