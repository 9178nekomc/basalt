package jsl.torch;

import jsl.core.ModelFormatException;
import jsl.core.ModelReader;
import jsl.core.ModelWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Training checkpoint persistence for {@link Module} and {@link Optimizer}: saves and restores
 * the complete model {@code stateDict}, epoch index, optimizer learning rate, Adam step count,
 * and last recorded loss to enable seamless resume-from-checkpoint training.
 */
public final class Checkpoint {
    private static final int MAGIC = 0x4A534C4B; // "JSLK"
    private static final int VERSION = 1;

    private final int epoch;
    private final float loss;
    private final float learningRate;
    private final long optimizerStep;
    private final Map<String, float[]> stateDict;

    public Checkpoint(int epoch, float loss, float learningRate, long optimizerStep,
                      Map<String, float[]> stateDict) {
        this.epoch = epoch;
        this.loss = loss;
        this.learningRate = learningRate;
        this.optimizerStep = optimizerStep;
        Map<String, float[]> copy = new LinkedHashMap<>();
        for (Map.Entry<String, float[]> e : stateDict.entrySet()) {
            copy.put(e.getKey(), e.getValue().clone());
        }
        this.stateDict = Collections.unmodifiableMap(copy);
    }

    public static void save(Path target, Module<?, ?> module, Optimizer optimizer,
                            int epoch, float loss) {
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(module, "module");
        float lr = optimizer == null ? 0.0f : optimizer.learningRate();
        long optStep = (optimizer instanceof AdamW adam) ? adam.stepCount() : 0L;
        Map<String, float[]> state = module.stateDict();

        ModelWriter writer = new ModelWriter(1024);
        writer.putInt(MAGIC);
        writer.putInt(VERSION);
        writer.putInt(epoch);
        writer.putDouble(loss);
        writer.putDouble(lr);
        writer.putLong(optStep);
        writer.putInt(state.size());
        for (Map.Entry<String, float[]> entry : state.entrySet()) {
            byte[] keyBytes = entry.getKey().getBytes(StandardCharsets.UTF_8);
            writer.putInt(keyBytes.length);
            writer.putBytes(keyBytes);
            float[] vals = entry.getValue();
            writer.putInt(vals.length);
            double[] dvals = new double[vals.length];
            for (int i = 0; i < vals.length; i++) dvals[i] = vals[i];
            writer.putDoubles(dvals);
        }
        try {
            Files.write(target, writer.toByteArray());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public static Checkpoint load(Path source) {
        Objects.requireNonNull(source, "source");
        try {
            byte[] bytes = Files.readAllBytes(source);
            ModelReader reader = new ModelReader(bytes);
            if (reader.getInt() != MAGIC) {
                throw new ModelFormatException("Invalid JSL checkpoint magic");
            }
            if (reader.getInt() != VERSION) {
                throw new ModelFormatException("Unsupported JSL checkpoint version");
            }
            int epoch = reader.getInt();
            float loss = (float) reader.getDouble();
            float lr = (float) reader.getDouble();
            long optStep = reader.getLong();
            int entries = reader.getInt();
            Map<String, float[]> state = new LinkedHashMap<>();
            for (int i = 0; i < entries; i++) {
                int kLen = reader.getInt();
                byte[] kBytes = new byte[kLen];
                for (int b = 0; b < kLen; b++) kBytes[b] = (byte) reader.getByte();
                String key = new String(kBytes, StandardCharsets.UTF_8);
                int vLen = reader.getInt();
                double[] dvals = reader.getDoubles(vLen);
                float[] fvals = new float[vLen];
                for (int j = 0; j < vLen; j++) fvals[j] = (float) dvals[j];
                state.put(key, fvals);
            }
            reader.requireFullyConsumed();
            return new Checkpoint(epoch, loss, lr, optStep, state);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public static Checkpoint restore(Path source, Module<?, ?> module, Optimizer optimizer) {
        Checkpoint ckpt = load(source);
        module.loadStateDict(ckpt.stateDict());
        if (optimizer != null) {
            if (ckpt.learningRate() > 0.0f) {
                optimizer.setLearningRate(ckpt.learningRate());
            }
            if (optimizer instanceof AdamW adam) {
                adam.setStepCount(ckpt.optimizerStep());
            }
        }
        return ckpt;
    }

    public int epoch() {
        return epoch;
    }

    public float loss() {
        return loss;
    }

    public float learningRate() {
        return learningRate;
    }

    public long optimizerStep() {
        return optimizerStep;
    }

    public Map<String, float[]> stateDict() {
        return stateDict;
    }
}
