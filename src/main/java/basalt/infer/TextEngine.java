package basalt.infer;

import basalt.Basalt;
import basalt.BasaltRuntimeException;
import basalt.core.BC;
import basalt.torch.Module;
import basalt.torch.Tensor;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Decoding loop for a windowed language model.
 *
 * <p>The model is an ordinary {@link Module}; what this class adds is everything around it that a
 * caller would otherwise rewrite for every project: keeping the window, running several contexts
 * through one native call, turning logits into a token, and saving the weights next to their shapes
 * so a chat session does not have to retrain.</p>
 *
 * <p>Where the work happens: every multiply, add and activation is executed by the Rust backend.
 * The loop itself is Java because the architecture is expressed in Java — the module decides which
 * ops run, so a replayable native program would have to duplicate that knowledge in Rust and drift
 * from it. The optimisation that does pay off is batching: {@link #scoresBatch} feeds N contexts
 * through a single {@code forward}, which turns N×ops JNI crossings per step into one.</p>
 *
 * <p>The engine borrows the runtime and the module; it never closes them. Closing an engine only
 * drops its own state, so a caller can keep using the model afterwards.</p>
 */
public final class TextEngine implements AutoCloseable {

    private final Basalt runtime;
    private final Module<BC, BC> model;
    private final int vocabulary;
    private final int window;
    private final Random random = new Random();

    private TextEngine(Basalt runtime, Module<BC, BC> model, int vocabulary, int window) {
        this.runtime = runtime;
        this.model = model;
        this.vocabulary = vocabulary;
        this.window = window;
    }

    /**
     * Binds a language model.
     *
     * @param runtime    runtime the model's tensors live in
     * @param model      module mapping {@code [N, window]} token ids to {@code [N, vocabulary]} logits
     * @param vocabulary number of distinct tokens
     * @param window     context length the module expects
     * @throws IllegalArgumentException if any width is not positive
     */
    public static TextEngine of(Basalt runtime, Module<BC, BC> model, int vocabulary, int window) {
        BasaltRuntimeException.requireNonNull(runtime, "runtime");
        BasaltRuntimeException.requireNonNull(model, "model");
        if (vocabulary < 2) {
            throw new IllegalArgumentException("vocabulary must be at least 2, got " + vocabulary);
        }
        if (window < 1) {
            throw new IllegalArgumentException("window must be at least 1, got " + window);
        }
        if (model.parameters().isEmpty()) {
            throw new IllegalArgumentException("model exposes no parameters; was it built empty?");
        }
        return new TextEngine(runtime, model, vocabulary, window);
    }

    /** Number of distinct tokens the model can emit. */
    public int vocabulary() {
        return vocabulary;
    }

    /** Context length the model was built for. */
    public int window() {
        return window;
    }

    /** The underlying module, for callers that need to inspect or extend it. */
    public Module<BC, BC> model() {
        return model;
    }

    /**
     * Logits for the next token after {@code context}.
     *
     * @param context at least {@link #window()} tokens; only the tail is used
     */
    public double[] scores(int[] context) {
        requireContext(context, 1);
        int[] tail = tail(context);
        float[] row = new float[window];
        for (int index = 0; index < window; index++) {
            row[index] = tail[index];
        }
        Tensor logits = model.forward(runtime.tensor(row, 1, window));
        float[] data = logits.data();
        double[] scores = new double[vocabulary];
        for (int index = 0; index < vocabulary; index++) {
            scores[index] = data[index];
        }
        return scores;
    }

    /**
     * Logits for several contexts at once, in a single forward pass.
     *
     * <p>This is the reason to batch: the native boundary is crossed once for the whole set instead
     * of once per row, and every row after the first is nearly free.</p>
     *
     * @param contexts rows of at least {@link #window()} tokens each; only the tails are used
     * @return one score row per context, in input order
     */
    public double[][] scoresBatch(int[][] contexts) {
        BasaltRuntimeException.requireNonNull(contexts, "contexts");
        if (contexts.length == 0) {
            throw new IllegalArgumentException("contexts must not be empty");
        }
        requireContext(contexts[0], contexts.length);
        int rows = contexts.length;
        float[] data = new float[rows * window];
        for (int row = 0; row < rows; row++) {
            int[] context = contexts[row];
            if (context == null || context.length < window) {
                throw new IllegalArgumentException("contexts[" + row + "] needs at least " + window
                        + " tokens, got " + (context == null ? "null" : context.length));
            }
            int offset = context.length - window;
            for (int index = 0; index < window; index++) {
                data[row * window + index] = context[offset + index];
            }
        }
        float[] logits = model.forward(runtime.tensor(data, rows, window)).data();
        double[][] scores = new double[rows][vocabulary];
        for (int row = 0; row < rows; row++) {
            for (int token = 0; token < vocabulary; token++) {
                scores[row][token] = logits[row * vocabulary + token];
            }
        }
        return scores;
    }

    /**
     * Greedy continuation: always take the most likely token.
     *
     * <p>Deterministic, so the same context always produces the same text — which is what you want
     * when checking whether a model actually learned something.</p>
     *
     * @return exactly {@code steps} tokens, excluding the context
     */
    public int[] generate(int[] context, int steps) {
        requireSteps(steps);
        int[] carried = tailAfter(context);
        int[] produced = new int[steps];
        for (int step = 0; step < steps; step++) {
            int next = argmax(scores(carried));
            produced[step] = next;
            push(carried, next);
        }
        return produced;
    }

    /**
     * Greedy continuation for several prompts at once.
     *
     * <p>Each step runs one batched forward over every prompt, so decoding N conversations costs
     * roughly one conversation's worth of native crossings per step instead of N.</p>
     *
     * @param contexts rows of at least {@link #window()} tokens
     * @return one token array per context, each exactly {@code steps} long
     */
    public int[][] generateBatch(int[][] contexts, int steps) {
        requireSteps(steps);
        BasaltRuntimeException.requireNonNull(contexts, "contexts");
        if (contexts.length == 0) {
            throw new IllegalArgumentException("contexts must not be empty");
        }
        int rows = contexts.length;
        int[] single = new int[1];
        int[][] carried = new int[rows][];
        for (int row = 0; row < rows; row++) {
            carried[row] = tailAfter(contexts[row]);
        }
        int[][] produced = new int[rows][steps];
        for (int step = 0; step < steps; step++) {
            double[][] batch = scoresBatch(carried);
            for (int row = 0; row < rows; row++) {
                int next = argmax(batch[row]);
                produced[row][step] = next;
                push(carried[row], next);
            }
        }
        return produced;
    }

    /**
     * Sampled continuation.
     *
     * <p>Temperature scales the logits before the softmax: below 1 sharpens towards the argmax,
     * above 1 flattens. {@code topK} keeps only the k most likely tokens before drawing, which is
     * what stops a small model from emitting a random rare character at every step. The engine's
     * own RNG is used, so {@link #seed} makes a run reproducible.</p>
     *
     * @param temperature positive; 1.0 leaves the distribution unchanged
     * @param topK        at least 1; values above the vocabulary are clamped
     */
    public int[] sample(int[] context, int steps, float temperature, int topK) {
        requireSteps(steps);
        if (!(temperature > 0f) || !Float.isFinite(temperature)) {
            throw new IllegalArgumentException("temperature must be positive and finite, got "
                    + temperature);
        }
        if (topK < 1) {
            throw new IllegalArgumentException("topK must be at least 1, got " + topK);
        }
        int[] carried = tailAfter(context);
        int[] produced = new int[steps];
        for (int step = 0; step < steps; step++) {
            double[] scores = scores(carried);
            int next = draw(scores, temperature, Math.min(topK, vocabulary));
            produced[step] = next;
            push(carried, next);
        }
        return produced;
    }

    /** Seeds the sampler, so a sampled run can be replayed. */
    public void seed(long seed) {
        random.setSeed(seed);
    }

    /**
     * Writes the weights and the shapes they belong to.
     *
     * <p>Stored by name rather than by position: loading into a module built with a different width
     * fails with the offending name instead of silently reshaping.</p>
     */
    public void save(Path path) {
        BasaltRuntimeException.requireNonNull(path, "path");
        Map<String, Tensor> parameters = new LinkedHashMap<>();
        for (Map.Entry<String, Tensor> entry : model.namedParameters()) {
            parameters.put(entry.getKey(), entry.getValue());
        }
        try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(path))) {
            out.writeInt(vocabulary);
            out.writeInt(window);
            out.writeInt(parameters.size());
            for (Map.Entry<String, Tensor> entry : parameters.entrySet()) {
                out.writeUTF(entry.getKey());
                float[] values = entry.getValue().data();
                out.writeInt(values.length);
                for (float value : values) {
                    out.writeFloat(value);
                }
            }
        } catch (IOException failure) {
            throw new UncheckedIOException("cannot write model to " + path, failure);
        }
    }

    /**
     * Reads weights written by {@link #save}.
     *
     * @throws IllegalStateException if the file was written for a different vocabulary or window,
     *                               or its parameter names do not match this model
     */
    public void load(Path path) {
        BasaltRuntimeException.requireNonNull(path, "path");
        try (DataInputStream in = new DataInputStream(Files.newInputStream(path))) {
            int storedVocabulary = in.readInt();
            int storedWindow = in.readInt();
            if (storedVocabulary != vocabulary || storedWindow != window) {
                throw new IllegalStateException("model was saved with vocabulary=" + storedVocabulary
                        + " window=" + storedWindow + ", this engine is vocabulary=" + vocabulary
                        + " window=" + window);
            }
            int count = in.readInt();
            Map<String, float[]> stored = new LinkedHashMap<>();
            for (int index = 0; index < count; index++) {
                String name = in.readUTF();
                float[] values = new float[in.readInt()];
                for (int value = 0; value < values.length; value++) {
                    values[value] = in.readFloat();
                }
                stored.put(name, values);
            }
            model.loadStateDict(stored);
        } catch (IOException failure) {
            throw new UncheckedIOException("cannot read model from " + path, failure);
        }
    }

    /** Drops engine state. The runtime and the module stay open. */
    @Override
    public void close() {
        // nothing owned, nothing to release; kept so the engine can be used in try-with-resources
    }

    private void requireContext(int[] context, int rows) {
        BasaltRuntimeException.requireNonNull(context, "context");
        if (context.length < window) {
            throw new IllegalArgumentException("context needs at least " + window
                    + " tokens, got " + context.length);
        }
        if (rows < 1) {
            throw new IllegalArgumentException("rows must be positive");
        }
    }

    private void requireSteps(int steps) {
        if (steps < 1) {
            throw new IllegalArgumentException("steps must be at least 1, got " + steps);
        }
    }

    private int[] tail(int[] context) {
        int[] tail = new int[window];
        System.arraycopy(context, context.length - window, tail, 0, window);
        return tail;
    }

    private int[] tailAfter(int[] context) {
        requireContext(context, 1);
        return tail(context);
    }

    private static void push(int[] carried, int next) {
        System.arraycopy(carried, 1, carried, 0, carried.length - 1);
        carried[carried.length - 1] = next;
    }

    private static int argmax(double[] scores) {
        int best = 0;
        for (int candidate = 1; candidate < scores.length; candidate++) {
            if (scores[candidate] > scores[best]) {
                best = candidate;
            }
        }
        return best;
    }

    /** Temperature softmax over the top {@code topK} logits, then one draw. */
    private int draw(double[] scores, float temperature, int topK) {
        List<Integer> order = new ArrayList<>(scores.length);
        for (int token = 0; token < scores.length; token++) {
            order.add(token);
        }
        order.sort((left, right) -> Double.compare(scores[right], scores[left]));

        double max = scores[order.get(0)];
        double total = 0.0;
        double[] weights = new double[topK];
        for (int index = 0; index < topK; index++) {
            weights[index] = Math.exp((scores[order.get(index)] - max) / temperature);
            total += weights[index];
        }
        double target = random.nextDouble() * total;
        double running = 0.0;
        for (int index = 0; index < topK; index++) {
            running += weights[index];
            if (target <= running) {
                return order.get(index);
            }
        }
        return order.get(topK - 1);
    }
}
