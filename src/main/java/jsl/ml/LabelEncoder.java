package jsl.ml;

import jsl.core.Matrix;
import jsl.core.ModelFormatException;
import jsl.core.ModelKind;
import jsl.core.ModelReader;
import jsl.core.ModelWriter;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.TreeSet;

/**
 * Encodes target labels (either {@code double[]} or {@code String[]}) with values between
 * {@code 0} and {@code n_classes - 1}, matching {@code sklearn.preprocessing.LabelEncoder}.
 */
public final class LabelEncoder implements Transformer {
    private double[] classes;
    private String[] stringClasses;
    private Map<String, Integer> stringToIndex;

    public LabelEncoder() {
    }

    public LabelEncoder fit(double[] targets) {
        Objects.requireNonNull(targets, "targets");
        if (targets.length == 0) {
            throw new IllegalArgumentException("targets must not be empty");
        }
        MlChecks.finite(targets, "targets");
        TreeSet<Double> unique = new TreeSet<>();
        for (double v : targets) {
            unique.add(v);
        }
        classes = new double[unique.size()];
        int idx = 0;
        for (double v : unique) {
            classes[idx++] = v;
        }
        stringClasses = null;
        stringToIndex = null;
        return this;
    }

    public LabelEncoder fit(String[] labels) {
        Objects.requireNonNull(labels, "labels");
        if (labels.length == 0) {
            throw new IllegalArgumentException("labels must not be empty");
        }
        TreeSet<String> unique = new TreeSet<>();
        for (String s : labels) {
            Objects.requireNonNull(s, "label element");
            unique.add(s);
        }
        stringClasses = unique.toArray(new String[0]);
        stringToIndex = new LinkedHashMap<>();
        classes = new double[stringClasses.length];
        for (int i = 0; i < stringClasses.length; i++) {
            stringToIndex.put(stringClasses[i], i);
            classes[i] = i;
        }
        return this;
    }

    public int[] transformToInt(double[] targets) {
        if (classes == null) {
            throw new IllegalStateException("LabelEncoder must be fitted first");
        }
        Objects.requireNonNull(targets, "targets");
        int[] encoded = new int[targets.length];
        for (int i = 0; i < targets.length; i++) {
            int pos = Arrays.binarySearch(classes, targets[i]);
            if (pos < 0) {
                throw new IllegalArgumentException("Unseen label in transform: " + targets[i]);
            }
            encoded[i] = pos;
        }
        return encoded;
    }

    public double[] transform(double[] targets) {
        int[] ints = transformToInt(targets);
        double[] out = new double[ints.length];
        for (int i = 0; i < ints.length; i++) {
            out[i] = ints[i];
        }
        return out;
    }

    public double[] fitTransform(double[] targets) {
        return fit(targets).transform(targets);
    }

    public double[] inverseTransform(double[] encoded) {
        if (classes == null) {
            throw new IllegalStateException("LabelEncoder must be fitted first");
        }
        Objects.requireNonNull(encoded, "encoded");
        double[] out = new double[encoded.length];
        for (int i = 0; i < encoded.length; i++) {
            int idx = (int) Math.round(encoded[i]);
            if (idx < 0 || idx >= classes.length || Math.abs(encoded[i] - idx) > 1e-9) {
                throw new IllegalArgumentException("Encoded label index out of range: " + encoded[i]);
            }
            out[i] = classes[idx];
        }
        return out;
    }

    public double[] inverseTransform(int[] encoded) {
        if (classes == null) {
            throw new IllegalStateException("LabelEncoder must be fitted first");
        }
        Objects.requireNonNull(encoded, "encoded");
        double[] out = new double[encoded.length];
        for (int i = 0; i < encoded.length; i++) {
            int idx = encoded[i];
            if (idx < 0 || idx >= classes.length) {
                throw new IllegalArgumentException("Encoded label index out of range: " + idx);
            }
            out[i] = classes[idx];
        }
        return out;
    }

    public int[] transform(String[] labels) {
        if (stringToIndex == null) {
            throw new IllegalStateException("LabelEncoder was not fitted on String labels");
        }
        Objects.requireNonNull(labels, "labels");
        int[] out = new int[labels.length];
        for (int i = 0; i < labels.length; i++) {
            Integer idx = stringToIndex.get(labels[i]);
            if (idx == null) {
                throw new IllegalArgumentException("Unseen string label: " + labels[i]);
            }
            out[i] = idx;
        }
        return out;
    }

    public int[] fitTransform(String[] labels) {
        return fit(labels).transform(labels);
    }

    public String[] inverseTransformStrings(int[] encoded) {
        if (stringClasses == null) {
            throw new IllegalStateException("LabelEncoder was not fitted on String labels");
        }
        Objects.requireNonNull(encoded, "encoded");
        String[] out = new String[encoded.length];
        for (int i = 0; i < encoded.length; i++) {
            int idx = encoded[i];
            if (idx < 0 || idx >= stringClasses.length) {
                throw new IllegalArgumentException("Encoded label index out of range: " + idx);
            }
            out[i] = stringClasses[idx];
        }
        return out;
    }

    @Override
    public LabelEncoder fit(Matrix features, double[] ignoredTargets) {
        Objects.requireNonNull(features, "features");
        if (features.columns() != 1) {
            throw new IllegalArgumentException("LabelEncoder Matrix adapter expects a 1-column Matrix");
        }
        return fit(features.data());
    }

    @Override
    public Matrix transform(Matrix features) {
        Objects.requireNonNull(features, "features");
        if (features.columns() != 1) {
            throw new IllegalArgumentException("LabelEncoder Matrix adapter expects a 1-column Matrix");
        }
        return new Matrix(features.rows(), 1, transform(features.data()));
    }

    @Override
    public Matrix inverseTransform(Matrix transformed) {
        Objects.requireNonNull(transformed, "transformed");
        if (transformed.columns() != 1) {
            throw new IllegalArgumentException("LabelEncoder Matrix adapter expects a 1-column Matrix");
        }
        return new Matrix(transformed.rows(), 1, inverseTransform(transformed.data()));
    }

    public double[] classes() {
        if (classes == null) throw new IllegalStateException("LabelEncoder is not fitted");
        return classes.clone();
    }

    public String[] stringClasses() {
        return stringClasses == null ? null : stringClasses.clone();
    }

    @Override
    public ModelKind kind() {
        return ModelKind.LABEL_ENCODER;
    }

    @Override
    public void write(ModelWriter writer) {
        if (classes == null) {
            throw new IllegalStateException("LabelEncoder must be fitted before serialization");
        }
        writer.putInt(classes.length);
        writer.putDoubles(classes);
        if (stringClasses == null) {
            writer.putInt(0);
        } else {
            writer.putInt(stringClasses.length);
            for (String s : stringClasses) {
                byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
                writer.putInt(bytes.length);
                writer.putBytes(bytes);
            }
        }
    }

    public static LabelEncoder read(ModelReader reader) {
        int count = reader.getInt();
        if (count <= 0) {
            throw new ModelFormatException("Invalid LabelEncoder class count");
        }
        LabelEncoder encoder = new LabelEncoder();
        encoder.classes = reader.getDoubles(count);
        int strCount = reader.getInt();
        if (strCount > 0) {
            encoder.stringClasses = new String[strCount];
            encoder.stringToIndex = new LinkedHashMap<>();
            for (int i = 0; i < strCount; i++) {
                int byteLen = reader.getInt();
                byte[] bytes = new byte[byteLen];
                for (int b = 0; b < byteLen; b++) {
                    bytes[b] = (byte) reader.getByte();
                }
                String s = new String(bytes, StandardCharsets.UTF_8);
                encoder.stringClasses[i] = s;
                encoder.stringToIndex.put(s, i);
            }
        }
        return encoder;
    }
}
