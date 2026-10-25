package jsl.ml;

import jsl.core.Matrix;
import jsl.core.ModelFormatException;
import jsl.core.ModelKind;
import jsl.core.ModelReader;
import jsl.core.ModelWriter;
import jsl.core.Persistable;
import jsl.io.ModelCodec;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Sequential pipeline of {@link Transformer} preprocessing / feature-engineering steps followed
 * by an optional final {@link Estimator}, matching {@code sklearn.pipeline.Pipeline}.
 *
 * <p>Implements both {@link Estimator} (when an estimator head is attached or the last step is an
 * estimator) and {@link Transformer} (when composed purely of transformers), plus full {@code .jsl}
 * binary persistence via {@link ModelCodec} so fitted preprocessing parameters and estimator weights
 * are saved and loaded as a single atomic unit.</p>
 */
public final class Pipeline implements Estimator, Transformer, Persistable {
    public record Step(String name, Transformer transformer) {
        public Step {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(transformer, "transformer");
        }
    }

    private final List<Step> steps;
    private final String estimatorName;
    private final Estimator estimator;

    private Pipeline(List<Step> steps, String estimatorName, Estimator estimator) {
        this.steps = List.copyOf(steps);
        this.estimatorName = estimatorName;
        this.estimator = estimator;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Pipeline fit(Matrix features) {
        return fit(features, null);
    }

    @Override
    public Pipeline fit(Matrix features, double[] targets) {
        Objects.requireNonNull(features, "features");
        Matrix current = features;
        for (Step step : steps) {
            current = step.transformer().fitTransform(current, targets);
        }
        if (estimator != null) {
            estimator.fit(current, targets);
        }
        return this;
    }

    @Override
    public Matrix transform(Matrix features) {
        Objects.requireNonNull(features, "features");
        Matrix current = features;
        for (Step step : steps) {
            current = step.transformer().transform(current);
        }
        if (estimator instanceof Transformer finalTransformer) {
            current = finalTransformer.transform(current);
        }
        return current;
    }

    @Override
    public Matrix inverseTransform(Matrix transformed) {
        Objects.requireNonNull(transformed, "transformed");
        Matrix current = transformed;
        if (estimator instanceof Transformer finalTransformer) {
            current = finalTransformer.inverseTransform(current);
        }
        for (int i = steps.size() - 1; i >= 0; i--) {
            current = steps.get(i).transformer().inverseTransform(current);
        }
        return current;
    }

    @Override
    public double[] predict(Matrix features) {
        Objects.requireNonNull(features, "features");
        Matrix current = features;
        for (Step step : steps) {
            current = step.transformer().transform(current);
        }
        if (estimator != null) {
            return estimator.predict(current);
        }
        if (!steps.isEmpty() && steps.get(steps.size() - 1).transformer() instanceof Estimator est) {
            return est.predict(features);
        }
        return current.toArray();
    }

    public List<Step> steps() {
        return steps;
    }

    public Estimator estimator() {
        return estimator;
    }

    public String estimatorName() {
        return estimatorName;
    }

    public Map<String, Object> namedSteps() {
        Map<String, Object> map = new LinkedHashMap<>();
        for (Step step : steps) {
            map.put(step.name(), step.transformer());
        }
        if (estimator != null && estimatorName != null) {
            map.put(estimatorName, estimator);
        }
        return Collections.unmodifiableMap(map);
    }

    @Override
    public ModelKind kind() {
        return ModelKind.PIPELINE;
    }

    @Override
    public void write(ModelWriter writer) {
        writer.putInt(steps.size());
        for (Step step : steps) {
            writeString(writer, step.name());
            writer.putInt(step.transformer().kind().code());
            ModelWriter sub = new ModelWriter(256);
            step.transformer().write(sub);
            byte[] payload = sub.toByteArray();
            writer.putInt(payload.length);
            writer.putBytes(payload);
        }
        if (estimator == null) {
            writer.putByte(0);
        } else {
            if (!(estimator instanceof Persistable persistable)) {
                throw new IllegalStateException(
                        "Pipeline final estimator " + estimator.getClass().getSimpleName()
                                + " does not implement Persistable");
            }
            writer.putByte(1);
            writeString(writer, estimatorName == null ? "estimator" : estimatorName);
            writer.putInt(persistable.kind().code());
            ModelWriter sub = new ModelWriter(512);
            persistable.write(sub);
            byte[] payload = sub.toByteArray();
            writer.putInt(payload.length);
            writer.putBytes(payload);
        }
    }

    public static Pipeline read(ModelReader reader) {
        int stepCount = reader.getInt();
        if (stepCount < 0 || stepCount > 10_000) {
            throw new ModelFormatException("Invalid Pipeline step count " + stepCount);
        }
        List<Step> steps = new ArrayList<>(stepCount);
        for (int i = 0; i < stepCount; i++) {
            String name = readString(reader);
            ModelKind kind = ModelKind.fromCode(reader.getInt());
            int payloadLen = reader.getInt();
            byte[] payload = readBytes(reader, payloadLen);
            Persistable loaded = ModelCodec.readPayload(kind, new ModelReader(payload));
            if (!(loaded instanceof Transformer transformer)) {
                throw new ModelFormatException("Pipeline step " + name + " is not a Transformer");
            }
            steps.add(new Step(name, transformer));
        }
        int hasEstimator = reader.getByte();
        String estName = null;
        Estimator est = null;
        if (hasEstimator != 0) {
            estName = readString(reader);
            ModelKind kind = ModelKind.fromCode(reader.getInt());
            int payloadLen = reader.getInt();
            byte[] payload = readBytes(reader, payloadLen);
            Persistable loaded = ModelCodec.readPayload(kind, new ModelReader(payload));
            if (!(loaded instanceof Estimator loadedEstimator)) {
                throw new ModelFormatException("Pipeline final model is not an Estimator");
            }
            est = loadedEstimator;
        }
        return new Pipeline(steps, estName, est);
    }

    private static void writeString(ModelWriter writer, String s) {
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
        writer.putInt(bytes.length);
        writer.putBytes(bytes);
    }

    private static String readString(ModelReader reader) {
        int len = reader.getInt();
        if (len < 0 || len > 65_536) {
            throw new ModelFormatException("Invalid string length " + len);
        }
        return new String(readBytes(reader, len), StandardCharsets.UTF_8);
    }

    private static byte[] readBytes(ModelReader reader, int length) {
        if (length < 0) throw new ModelFormatException("Negative byte length");
        byte[] bytes = new byte[length];
        for (int i = 0; i < length; i++) {
            bytes[i] = (byte) reader.getByte();
        }
        return bytes;
    }

    public static final class Builder {
        private final List<Step> steps = new ArrayList<>();
        private String estimatorName;
        private Estimator estimator;

        public Builder addStep(String name, Transformer transformer) {
            steps.add(new Step(name, transformer));
            return this;
        }

        public Builder addStep(Transformer transformer) {
            Objects.requireNonNull(transformer, "transformer");
            steps.add(new Step("step" + steps.size(), transformer));
            return this;
        }

        public Builder estimator(String name, Estimator estimator) {
            this.estimatorName = Objects.requireNonNull(name, "name");
            this.estimator = Objects.requireNonNull(estimator, "estimator");
            return this;
        }

        public Builder estimator(Estimator estimator) {
            this.estimatorName = "estimator";
            this.estimator = Objects.requireNonNull(estimator, "estimator");
            return this;
        }

        public Pipeline build() {
            if (steps.isEmpty() && estimator == null) {
                throw new IllegalArgumentException("Pipeline requires at least one step or estimator");
            }
            return new Pipeline(steps, estimatorName, estimator);
        }
    }
}
