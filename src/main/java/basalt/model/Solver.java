package basalt.model;

/**
 * One trainable candidate: a named recipe that can score itself on a dataset and then be fitted on
 * all of it.
 *
 * <p>This is the seam that lets one screening loop compare a decision forest with a neural network
 * and a character-level language model. A solver is a stateless descriptor — it holds configuration,
 * not weights — so the same instance can be scored and fitted any number of times.</p>
 *
 * <p>Two contracts matter:</p>
 * <ul>
 *   <li>{@link #score} must return a number where larger is better, computed without touching the
 *       rows it reports on. Classical solvers cross-validate; neural solvers hold out a slice and
 *       train once, because a full k-fold sweep of a network is far more expensive than the model it
 *       is ranking.</li>
 *   <li>{@link #fit} trains on everything and returns something that can predict or generate.</li>
 * </ul>
 */
public interface Solver {

    /** Stable identifier, unique within {@link Solvers}. */
    String name();

    /** The task this solver can handle; screening only pairs matching tasks. */
    Task task();

    /**
     * Estimates how well this solver would do on {@code data}.
     *
     * @param folds requested cross-validation depth; classical solvers honour it (clamped to what
     *              the data allows), neural solvers hold out a single slice and ignore it
     * @return a score where larger is better
     * @throws IllegalArgumentException if the dataset poses a different task
     */
    double score(Dataset data, int folds, long seed);

    /**
     * Trains on all rows of {@code data}.
     *
     * @throws IllegalArgumentException if the dataset poses a different task
     */
    Fitted fit(Dataset data, long seed);
}
