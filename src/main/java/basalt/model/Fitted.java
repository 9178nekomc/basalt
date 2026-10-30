package basalt.model;

import basalt.core.Matrix;

/**
 * A trained model, whatever family it came from.
 *
 * <p>Two capabilities, and a model only has the ones its task needs: tabular models predict one
 * value per row, sequence models generate a continuation as token ids — mapping ids to glyphs is the
 * caller's business, because the library never assumes a charset. The unused one throws
 * {@link UnsupportedOperationException} with a message that says which kind of model you are
 * holding, so a mistake surfaces immediately instead of returning garbage.</p>
 */
public interface Fitted extends AutoCloseable {

    /** One prediction per row. Tabular models only. */
    default double[] predict(Matrix features) {
        throw new UnsupportedOperationException(
                "this model is a " + task() + " model and does not predict rows; "
                        + "use generate(...) instead");
    }

    /** Continues {@code seed} by {@code steps} tokens. Sequence models only. */
    default int[] generate(int[] seed, int steps) {
        throw new UnsupportedOperationException(
                "this model is a " + task() + " model and does not generate tokens; "
                        + "use predict(...) instead");
    }

    /** Task the model was trained for. */
    Task task();

    @Override
    default void close() {
    }
}
