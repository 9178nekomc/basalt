package basalt.ml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Supplier;

/**
 * Named catalogue of estimators, open for third-party extension.
 *
 * <p>The built-in catalogue is derived from {@link Estimators.Algorithm}, so the two can never
 * drift apart. Additional estimators can be registered at runtime and immediately become visible
 * to {@link basalt.auto.Auto}, which screens every entry for the detected task. That is the
 * extension point for a new classical algorithm: implement {@link Estimator}, register a factory,
 * and it competes with the built-ins without touching library code.</p>
 *
 * <p>Names are lower-case snake case ({@code linear_regression}, {@code random_forest}). A name may
 * only be registered once; registering a duplicate is rejected instead of silently replacing the
 * earlier factory, because screening order and report labels depend on it.</p>
 *
 * <p>All methods are synchronized on the registry: registration usually happens once at start-up,
 * but {@link basalt.auto.Auto} reads the catalogue while other threads may still be registering.</p>
 */
public final class Registry {

    /** One catalogue entry: its public name, the task it solves, and whether it ships built in. */
    public record Entry(String name, Estimators.Task task, boolean builtIn) {
    }

    private static final Map<String, Estimators.Task> TASKS = new LinkedHashMap<>();
    private static final Map<String, Supplier<Estimator>> FACTORIES = new LinkedHashMap<>();
    private static final Set<String> BUILT_IN = new LinkedHashSet<>();

    static {
        for (Estimators.Algorithm algorithm : Estimators.Algorithm.values()) {
            String name = algorithm.name().toLowerCase(Locale.ROOT);
            TASKS.put(name, algorithm.task());
            FACTORIES.put(name, () -> Estimators.create(algorithm));
            BUILT_IN.add(name);
        }
    }

    private Registry() {
    }

    /**
     * Registers a factory under {@code name}.
     *
     * @throws IllegalArgumentException if the name is blank, already taken, or the factory is null
     */
    public static synchronized void register(String name, Estimators.Task task,
                                             Supplier<Estimator> factory) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("name must not be blank");
        }
        if (task == null) {
            throw new IllegalArgumentException("task must not be null");
        }
        if (factory == null) {
            throw new IllegalArgumentException("factory must not be null");
        }
        String key = name.toLowerCase(Locale.ROOT);
        if (TASKS.containsKey(key)) {
            throw new IllegalArgumentException("algorithm '" + key + "' is already registered");
        }
        TASKS.put(key, task);
        FACTORIES.put(key, factory);
    }

    /**
     * Builds a fresh, unfitted estimator for {@code name}.
     *
     * @throws IllegalArgumentException if the name is unknown
     */
    public static synchronized Estimator create(String name) {
        Supplier<Estimator> factory = FACTORIES.get(key(name));
        if (factory == null) {
            throw new IllegalArgumentException("unknown algorithm '" + name + "'; registered: "
                    + String.join(", ", TASKS.keySet()));
        }
        Estimator estimator = factory.get();
        if (estimator == null) {
            throw new IllegalStateException("factory for '" + name + "' returned null");
        }
        return estimator;
    }

    /** Every registered entry, built-ins first, in registration order. */
    public static synchronized List<Entry> all() {
        List<Entry> entries = new ArrayList<>(TASKS.size());
        for (Map.Entry<String, Estimators.Task> entry : TASKS.entrySet()) {
            entries.add(new Entry(entry.getKey(), entry.getValue(),
                    BUILT_IN.contains(entry.getKey())));
        }
        return Collections.unmodifiableList(entries);
    }

    /** Entries that solve {@code task}, in registration order. */
    public static synchronized List<Entry> byTask(Estimators.Task task) {
        if (task == null) {
            throw new IllegalArgumentException("task must not be null");
        }
        List<Entry> entries = new ArrayList<>();
        for (Entry entry : all()) {
            if (entry.task() == task) {
                entries.add(entry);
            }
        }
        return entries;
    }

    /** Whether {@code name} resolves to a factory. */
    public static synchronized boolean contains(String name) {
        return TASKS.containsKey(key(name));
    }

    /** Number of registered algorithms. */
    public static synchronized int size() {
        return TASKS.size();
    }

    private static String key(String name) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("name must not be blank");
        }
        return name.toLowerCase(Locale.ROOT);
    }
}
