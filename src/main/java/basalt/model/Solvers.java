package basalt.model;

import basalt.ml.Estimators;
import basalt.ml.Registry;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Catalogue of everything that can be screened, across every model family.
 *
 * <p>Classical solvers are <em>derived</em>, not copied: every supervised algorithm in {@link
 * Registry} appears here as a {@link ClassicalSolver}, built on demand. That keeps the two
 * extension points connected — registering an {@link basalt.ml.Estimator} with {@code Registry} is
 * enough for it to be screened, without also having to wrap it. Solvers that are not classical
 * (networks, language models) have no estimator behind them, so they are registered explicitly.</p>
 *
 * <p>Names are case-insensitive and may only be used once; replacing silently would make a
 * screening report impossible to reproduce.</p>
 */
public final class Solvers {

    private static final Map<String, Solver> REGISTERED = new LinkedHashMap<>();

    static {
        register(NeuralSolver.tabular("neural_mlp", Task.TABULAR_CLASSIFICATION,
                new int[]{64, 32}, 200, 32, 1e-2f));
        register(NeuralSolver.tabular("neural_mlp_regressor", Task.TABULAR_REGRESSION,
                new int[]{64, 32}, 200, 32, 5e-3f));
        register(NeuralSolver.sequence("char_lm", 6, 32, 256, 300, 8, 3e-3f));
    }

    private Solvers() {
    }

    /**
     * Adds a non-derived solver to the catalogue.
     *
     * @throws IllegalArgumentException if the solver is null, unnamed, or the name collides with a
     *                                  registered estimator or another solver
     */
    public static synchronized void register(Solver solver) {
        if (solver == null) {
            throw new IllegalArgumentException("solver must not be null");
        }
        String key = key(solver.name());
        if (REGISTERED.containsKey(key)) {
            throw new IllegalArgumentException("solver '" + key + "' is already registered");
        }
        if (supervisedEstimator(key)) {
            throw new IllegalArgumentException("'" + key
                    + "' is already an estimator in basalt.ml.Registry; it is screened automatically");
        }
        REGISTERED.put(key, solver);
    }

    /**
     * Looks a solver up by name, deriving a classical one if the name belongs to an estimator.
     *
     * @throws IllegalArgumentException if the name is unknown or names an unsupervised algorithm
     */
    public static synchronized Solver get(String name) {
        String key = key(name);
        Solver registered = REGISTERED.get(key);
        if (registered != null) {
            return registered;
        }
        if (supervisedEstimator(key)) {
            return new ClassicalSolver(key);
        }
        throw new IllegalArgumentException("unknown solver '" + name + "'; registered: "
                + String.join(", ", names()));
    }

    /** Whether {@code name} resolves to a solver. */
    public static synchronized boolean contains(String name) {
        String key = key(name);
        return REGISTERED.containsKey(key) || supervisedEstimator(key);
    }

    /** Every solver: derived classical ones first, then explicitly registered ones. */
    public static synchronized List<Solver> all() {
        List<Solver> result = new ArrayList<>();
        Set<String> seen = new LinkedHashSet<>();
        for (String name : estimatorNames()) {
            if (!seen.add(name)) {
                continue;
            }
            Solver override = REGISTERED.get(name);
            result.add(override != null ? override : new ClassicalSolver(name));
        }
        for (Map.Entry<String, Solver> entry : REGISTERED.entrySet()) {
            if (seen.add(entry.getKey())) {
                result.add(entry.getValue());
            }
        }
        return Collections.unmodifiableList(result);
    }

    /** Solvers that declare {@code task}. */
    public static synchronized List<Solver> byTask(Task task) {
        if (task == null) {
            throw new IllegalArgumentException("task must not be null");
        }
        List<Solver> matching = new ArrayList<>();
        for (Solver solver : all()) {
            if (solver.task() == task) {
                matching.add(solver);
            }
        }
        return Collections.unmodifiableList(matching);
    }

    /** Number of distinct solver names. */
    public static synchronized int size() {
        return all().size();
    }

    private static List<String> names() {
        List<String> names = new ArrayList<>();
        for (Solver solver : all()) {
            names.add(solver.name());
        }
        return names;
    }

    private static List<String> estimatorNames() {
        List<String> names = new ArrayList<>();
        for (Registry.Entry entry : Registry.all()) {
            if (entry.task() != Estimators.Task.UNSUPERVISED) {
                names.add(entry.name());
            }
        }
        return names;
    }

    private static boolean supervisedEstimator(String name) {
        if (!Registry.contains(name)) {
            return false;
        }
        for (Registry.Entry entry : Registry.all()) {
            if (entry.name().equals(name)) {
                return entry.task() != Estimators.Task.UNSUPERVISED;
            }
        }
        return false;
    }

    private static String key(String name) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("name must not be blank");
        }
        return name.toLowerCase(Locale.ROOT);
    }
}
