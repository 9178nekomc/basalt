package jsl.ml;

import jsl.core.Matrix;
import jsl.core.ModelFormatException;
import jsl.core.ModelKind;
import jsl.core.ModelReader;
import jsl.core.ModelWriter;
import java.util.Arrays;

/**
 * Transforms features by scaling each feature to a given range {@code [featureMin, featureMax]}
 * (default {@code [0, 1]}), matching {@code sklearn.preprocessing.MinMaxScaler}.
 */
public final class MinMaxScaler implements Transformer, Estimator {
    private final double featureMin;
    private final double featureMax;
    private final boolean clip;
    private double[] dataMin;
    private double[] dataMax;
    private double[] dataRange;
    private double[] scale;
    private double[] minOffset;
    private int featureCount;

    private MinMaxScaler(Builder builder) {
        this.featureMin = builder.featureMin;
        this.featureMax = builder.featureMax;
        this.clip = builder.clip;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public MinMaxScaler fit(Matrix features) {
        return fit(features, null);
    }

    @Override
    public MinMaxScaler fit(Matrix features, double[] ignoredTargets) {
        MlChecks.features(features);
        int rows = features.rows();
        featureCount = features.columns();
        dataMin = new double[featureCount];
        dataMax = new double[featureCount];
        dataRange = new double[featureCount];
        scale = new double[featureCount];
        minOffset = new double[featureCount];
        Arrays.fill(dataMin, Double.POSITIVE_INFINITY);
        Arrays.fill(dataMax, Double.NEGATIVE_INFINITY);
        double[] data = features.data();

        for (int r = 0; r < rows; r++) {
            int offset = r * featureCount;
            for (int c = 0; c < featureCount; c++) {
                double v = data[offset + c];
                if (v < dataMin[c]) dataMin[c] = v;
                if (v > dataMax[c]) dataMax[c] = v;
            }
        }
        double targetSpan = featureMax - featureMin;
        for (int c = 0; c < featureCount; c++) {
            double range = dataMax[c] - dataMin[c];
            dataRange[c] = range;
            double safeRange = range > 1e-12 ? range : 1.0;
            scale[c] = targetSpan / safeRange;
            minOffset[c] = featureMin - dataMin[c] * scale[c];
        }
        return this;
    }

    @Override
    public Matrix transform(Matrix features) {
        MlChecks.predictionData(features, featureCount, scale != null);
        int rows = features.rows();
        double[] src = features.data();
        double[] dst = new double[rows * featureCount];
        for (int r = 0; r < rows; r++) {
            int offset = r * featureCount;
            for (int c = 0; c < featureCount; c++) {
                double v = src[offset + c] * scale[c] + minOffset[c];
                if (clip) {
                    v = Math.max(featureMin, Math.min(featureMax, v));
                }
                dst[offset + c] = v;
            }
        }
        return new Matrix(rows, featureCount, dst);
    }

    @Override
    public Matrix inverseTransform(Matrix transformed) {
        MlChecks.predictionData(transformed, featureCount, scale != null);
        int rows = transformed.rows();
        double[] src = transformed.data();
        double[] dst = new double[rows * featureCount];
        for (int r = 0; r < rows; r++) {
            int offset = r * featureCount;
            for (int c = 0; c < featureCount; c++) {
                dst[offset + c] = (src[offset + c] - minOffset[c]) / scale[c];
            }
        }
        return new Matrix(rows, featureCount, dst);
    }

    @Override
    public double[] predict(Matrix features) {
        return transform(features).toArray();
    }

    public double[] dataMin() {
        if (dataMin == null) throw new IllegalStateException("MinMaxScaler is not fitted");
        return dataMin.clone();
    }

    public double[] dataMax() {
        if (dataMax == null) throw new IllegalStateException("MinMaxScaler is not fitted");
        return dataMax.clone();
    }

    public double[] dataRange() {
        if (dataRange == null) throw new IllegalStateException("MinMaxScaler is not fitted");
        return dataRange.clone();
    }

    public double[] scale() {
        if (scale == null) throw new IllegalStateException("MinMaxScaler is not fitted");
        return scale.clone();
    }

    @Override
    public ModelKind kind() {
        return ModelKind.MIN_MAX_SCALER;
    }

    @Override
    public void write(ModelWriter writer) {
        if (scale == null) {
            throw new IllegalStateException("MinMaxScaler must be fitted before serialization");
        }
        writer.putDouble(featureMin);
        writer.putDouble(featureMax);
        writer.putByte(clip ? 1 : 0);
        writer.putInt(featureCount);
        writer.putDoubles(dataMin);
        writer.putDoubles(dataMax);
        writer.putDoubles(dataRange);
        writer.putDoubles(scale);
        writer.putDoubles(minOffset);
    }

    public static MinMaxScaler read(ModelReader reader) {
        double fMin = reader.getDouble();
        double fMax = reader.getDouble();
        boolean clip = reader.getByte() != 0;
        int features = reader.getInt();
        if (features <= 0 || !(fMax > fMin)) {
            throw new ModelFormatException("Invalid MinMaxScaler header");
        }
        MinMaxScaler scaler = new Builder().featureRange(fMin, fMax).clip(clip).build();
        scaler.featureCount = features;
        scaler.dataMin = reader.getDoubles(features);
        scaler.dataMax = reader.getDoubles(features);
        scaler.dataRange = reader.getDoubles(features);
        scaler.scale = reader.getDoubles(features);
        scaler.minOffset = reader.getDoubles(features);
        return scaler;
    }

    public static final class Builder {
        private double featureMin = 0.0;
        private double featureMax = 1.0;
        private boolean clip = false;

        public Builder featureRange(double min, double max) {
            if (!Double.isFinite(min) || !Double.isFinite(max) || !(max > min)) {
                throw new IllegalArgumentException("featureRange requires finite min < max");
            }
            this.featureMin = min;
            this.featureMax = max;
            return this;
        }

        public Builder clip(boolean clip) {
            this.clip = clip;
            return this;
        }

        public MinMaxScaler build() {
            return new MinMaxScaler(this);
        }
    }
}
