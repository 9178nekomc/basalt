package basalt.tests;

import basalt.Basalt;
import basalt.core.BC;
import basalt.infer.TextEngine;
import basalt.nativebridge.NativeContext;
import basalt.torch.Linear;
import basalt.torch.Module;
import basalt.torch.Tensor;
import basalt.zoo.Trainer;
import basalt.zoo.Zoo;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The decoding engine: batching, greedy and sampled decoding, and the weight round-trip. */
class TextEngineTest {

    private static final int VOCABULARY = 5;
    private static final int WINDOW = 3;

    @Test
    void batchingMatchesSingleRowScoring() {
        try (Basalt engine = Basalt.create()) {
            Module<BC, BC> model = new LastTokenModel(engine.context(), VOCABULARY);
            TextEngine text = TextEngine.of(engine, model, VOCABULARY, WINDOW);

            int[][] contexts = {{0, 1, 2}, {4, 4, 4}, {2, 0, 1}};
            double[][] batch = text.scoresBatch(contexts);
            assertEquals(3, batch.length);
            for (int row = 0; row < contexts.length; row++) {
                assertArrayEquals(text.scores(contexts[row]), batch[row], 1e-6,
                        "batched row " + row + " must equal the single-row score");
            }
            model.close();
        }
    }

    @Test
    void greedyDecodingIsDeterministicAndUsesOnlyTheTail() {
        try (Basalt engine = Basalt.create()) {
            Module<BC, BC> model = new LastTokenModel(engine.context(), VOCABULARY);
            TextEngine text = TextEngine.of(engine, model, VOCABULARY, WINDOW);

            int[] first = text.generate(new int[]{0, 0, 0}, 4);
            int[] second = text.generate(new int[]{0, 0, 0}, 4);
            assertArrayEquals(first, second, "greedy decoding must be deterministic");
            assertArrayEquals(new int[]{1, 2, 3, 4}, first,
                    "the toy model maps token t to (t + 1) mod vocabulary");

            assertArrayEquals(new int[]{4}, text.generate(new int[]{9, 9, 3}, 1),
                    "a long context must be truncated to its tail");
            model.close();
        }
    }

    @Test
    void batchedGenerationAgreesWithSequential() {
        try (Basalt engine = Basalt.create()) {
            Module<BC, BC> model = new LastTokenModel(engine.context(), VOCABULARY);
            TextEngine text = TextEngine.of(engine, model, VOCABULARY, WINDOW);

            int[][] contexts = {{0, 0, 0}, {1, 1, 1}, {3, 3, 3}};
            int[][] batched = text.generateBatch(contexts, 3);
            for (int row = 0; row < contexts.length; row++) {
                assertArrayEquals(text.generate(contexts[row], 3), batched[row],
                        "batch row " + row + " must match sequential decoding");
            }
            model.close();
        }
    }

    @Test
    void samplingIsSeededAndHonoursTopK() {
        try (Basalt engine = Basalt.create()) {
            Module<BC, BC> model = new LastTokenModel(engine.context(), VOCABULARY);
            TextEngine text = TextEngine.of(engine, model, VOCABULARY, WINDOW);

            text.seed(7L);
            int[] first = text.sample(new int[]{0, 0, 0}, 6, 1.0f, 2);
            text.seed(7L);
            assertArrayEquals(first, text.sample(new int[]{0, 0, 0}, 6, 1.0f, 2),
                    "same seed must replay the same draw");

            // topK=1 keeps a single candidate, so sampling must reproduce greedy decoding --
            // note the context shifts every step, so the tokens are not all the same value
            text.seed(11L);
            assertArrayEquals(text.generate(new int[]{0, 0, 0}, 40),
                    text.sample(new int[]{0, 0, 0}, 40, 1.0f, 1),
                    "topK=1 degenerates to the argmax at every step");
            model.close();
        }
    }

    @Test
    void weightsRoundTripThroughDisk() throws Exception {
        Path file = Files.createTempFile("text-engine", ".bin");
        try (Basalt engine = Basalt.create()) {
            Module<BC, BC> model = new LastTokenModel(engine.context(), VOCABULARY);
            TextEngine text = TextEngine.of(engine, model, VOCABULARY, WINDOW);
            double[] before = text.scores(new int[]{0, 1, 2});
            text.save(file);

            Module<BC, BC> restored = new LastTokenModel(engine.context(), VOCABULARY);
            TextEngine reloaded = TextEngine.of(engine, restored, VOCABULARY, WINDOW);
            reloaded.load(file);
            assertArrayEquals(before, reloaded.scores(new int[]{0, 1, 2}), 1e-9,
                    "reloaded weights must produce identical logits");
            model.close();
            restored.close();
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test
    void streamingEmitsTheSameTokensInOrder() {
        try (Basalt engine = Basalt.create()) {
            Module<BC, BC> model = new LastTokenModel(engine.context(), VOCABULARY);
            TextEngine text = TextEngine.of(engine, model, VOCABULARY, WINDOW);

            java.util.List<Integer> seen = new java.util.ArrayList<>();
            int[] streamed = text.stream(new int[]{0, 0, 0}, 5, seen::add);
            assertArrayEquals(text.generate(new int[]{0, 0, 0}, 5), streamed,
                    "streaming must not change the chosen tokens");
            assertEquals(5, seen.size(), "the sink must be called once per step");
            for (int index = 0; index < seen.size(); index++) {
                assertEquals(streamed[index], seen.get(index), "tokens must arrive in order");
            }

            java.util.List<Integer> sampledSeen = new java.util.ArrayList<>();
            text.seed(3L);
            int[] sampled = text.streamSampled(new int[]{0, 0, 0}, 4, 1.0f, 2, sampledSeen::add);
            text.seed(3L);
            assertArrayEquals(text.sample(new int[]{0, 0, 0}, 4, 1.0f, 2), sampled,
                    "streamed sampling must match sample() under the same seed");
            assertEquals(4, sampledSeen.size());
            model.close();
        }
    }

    @Test
    void rejectsShapesItCannotServe() {
        try (Basalt engine = Basalt.create()) {
            Module<BC, BC> model = new LastTokenModel(engine.context(), VOCABULARY);
            TextEngine text = TextEngine.of(engine, model, VOCABULARY, WINDOW);
            assertThrows(IllegalArgumentException.class, () -> text.scores(new int[]{0, 1}));
            assertThrows(IllegalArgumentException.class, () -> text.generate(new int[]{0, 1, 2}, 0));
            assertThrows(IllegalArgumentException.class,
                    () -> text.sample(new int[]{0, 1, 2}, 3, 0f, 2));
            assertThrows(IllegalArgumentException.class,
                    () -> text.sample(new int[]{0, 1, 2}, 3, 1f, 0));
            assertThrows(IllegalArgumentException.class,
                    () -> text.scoresBatch(new int[][]{{0, 1}, {0, 1, 2}}));
            assertThrows(IllegalArgumentException.class,
                    () -> TextEngine.of(engine, model, 1, WINDOW));
            model.close();
        }
    }

    @Test
    void decodingATrainedModelReproducesItsCorpus() {
        String corpus = "滚木，你好";
        StringBuilder builder = new StringBuilder();
        corpus.chars().distinct().sorted().forEach(builder::appendCodePoint);
        String alphabet = builder.toString();
        int[] ids = new int[corpus.length()];
        for (int index = 0; index < ids.length; index++) {
            ids[index] = alphabet.indexOf(corpus.charAt(index));
        }

        try (Basalt engine = Basalt.create()) {
            Module<BC, BC> model = Zoo.charLm(engine, alphabet.length(), 3, 8, 32);
            int samples = ids.length - 3;
            float[] windows = new float[samples * 3];
            float[] targets = new float[samples];
            for (int sample = 0; sample < samples; sample++) {
                for (int offset = 0; offset < 3; offset++) {
                    windows[sample * 3 + offset] = ids[sample + offset];
                }
                targets[sample] = ids[sample + 3];
            }
            Trainer.of(engine.context(), model,
                            engine.adamw(model.parameters(), 1e-2f), engine.crossEntropy())
                    .epochs(400)
                    .batchSize(samples)
                    .shuffle(false)
                    .fit(engine.tensor(windows, samples, 3), engine.tensor(targets, samples));

            TextEngine text = TextEngine.of(engine, model, alphabet.length(), 3);
            int[] produced = text.generate(ids, ids.length - 3);
            int matched = 0;
            for (int index = 0; index < produced.length; index++) {
                if (produced[index] == ids[3 + index]) {
                    matched++;
                }
            }
            assertTrue(matched == produced.length, "a memorised corpus must decode exactly, matched "
                    + matched + " of " + produced.length);
            model.close();
        }
    }

    /** Logits that depend only on the last token: argmax is {@code (last + 1) mod vocabulary}. */
    private static final class LastTokenModel extends Module<BC, BC> {
        private final Linear head;
        private final int vocabulary;

        LastTokenModel(NativeContext context, int vocabulary) {
            this.vocabulary = vocabulary;
            this.head = new Linear(context, vocabulary, vocabulary);
            float[] identity = new float[vocabulary * vocabulary];
            for (int token = 0; token < vocabulary; token++) {
                identity[token * vocabulary + (token + 1) % vocabulary] = 6.0f;
            }
            head.weight().setData(identity);
            head.bias().fill(0.0f);
        }

        @Override
        public Tensor forward(Tensor input) {
            int rows = input.shape()[0];
            int columns = input.shape()[1];
            float[] data = input.data();
            float[] oneHot = new float[rows * vocabulary];
            for (int row = 0; row < rows; row++) {
                int token = (int) data[row * columns + columns - 1];
                oneHot[row * vocabulary + Math.floorMod(token, vocabulary)] = 1.0f;
            }
            return head.forward(Tensor.of(input.context(), oneHot, rows, vocabulary));
        }

        @Override
        public List<Tensor> parameters() {
            return List.of(head.weight(), head.bias());
        }
    }
}
