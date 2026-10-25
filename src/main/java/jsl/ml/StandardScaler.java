package jsl.ml;

import jsl.core.Matrix;
import jsl.core.ModelFormatException;
import jsl.core.ModelKind;
import jsl.core.ModelReader;
import jsl.core.ModelWriter;
import java.util.Arrays;

/**
 * Standardizes features by removing the mean and scaling to unit variance ({@code z = (x - u) / s}),
 * matching {@code sklearn.preprocessing.StandardScaler} (population variance ddof=0 by default).
 */
public final class StandardScaler implements Transformer, Estimator {
    private final boolean withMean;
    private final boolean withStd;
    private double[] mean;
    private double[] scale;
    private double[] var;
    private int featureCount;
    private int samplesSeen;

    private StandardScaler(Builder builder) {
        this.withMean = builder.withMean;
        this.withStd = builder.withStd;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public StandardScaler fit(Matrix features) {
        return fit(features, null);
    }

    @Override
    public StandardScaler fit(Matrix features, double[] ignoredTargets) {
        MlChecks.features(features);
        int rows = features.rows();
        featureCount = features.columns();
        samplesSeen = rows;
        mean = new double[featureCount];
        var = new double[featureCount];
        scale = new double[featureCount];
        double[] data = features.data();

        for (int r = 0; r < rows; r++) {
            int offset = r * featureCount;
            for (int c = 0; c < featureCount; c++) {
                mean[c] += data[offset + c];
            }
        }
        for (int c = 0; c < featureCount; c++) {
            mean[c] /= rows;
        }
        for (int r = 0; r < rows; r++) {
            int offset = r * featureCount;
            for (int c = 0; c < featureCount; c++) {
                double d = data[offset + c] - mean[c];
                var[c] += d * d;
            }
        }
        for (int c = 0; c < featureCount; c++) {
            var[c] /= rows;
            double s = Math.sqrt(var[c]);
            scale[c] = (withStd && s > 1e-12) ? s : 1.0;
        }
        return this;
    }

    @Override
    public Matrix transform(Matrix features) {
        MlChecks.predictionData(features, featureCount, mean != null);
        int rows = features.rows();
        double[] src = features.data();
        double[] dst = new double[rows * featureCount];
        for (int r = 0; r < rows; r++) {
            int offset = r * featureCount;
            for (int c = 0; c < featureCount; c++) {
                double v = src[offset + c];
                if (withMean) {
                    v -= mean[c];
                }
                if (withStd) {
                    v /= scale[c];
                }
                dst[offset + c] = v;
            }
        }
        return new Matrix(rows, featureCount, dst);
    }

    @Override
    public Matrix inverseTransform(Matrix transformed) {
        MlChecks.predictionData(transformed, featureCount, mean != null);
        int rows = transformed.rows();
        double[] src = transformed.data();
        double[] dst = new double[rows * featureCount];
        for (int r = 0; r < rows; r++) {
            int offset = r * featureCount;
            for (int c = 0; c < featureCount; c++) {
                double v = src[offset + c];
                if (withStd) {
                    v *= scale[c];
                }
                if (withMean) {
                    v += mean[c];
                }
                dst[offset + c] = v;
            }
        }
        return new Matrix(rows, featureCount, dst);
    }

    @Override
    public double[] predict(Matrix features) {
        return transform(features).toArray();
    }

    public double[] mean() {
        if (mean == null) throw new IllegalStateException("StandardScaler is not fitted");
        return mean.clone();
    }

    public double[] scale() {
        if (scale == null) throw new IllegalStateException("StandardScaler is not fitted");
        return scale.clone();
    }

    public double[] variance() {
        if (var == null) throw new IllegalStateException("StandardScaler is not fitted");
        return var.clone();
    }

    public int samplesSeen() {
        return samplesSeen;
    }

    public boolean withMean() {
        return withMean;
    }

    public boolean withStd() {
        return withStd;
    }

    @Override
    public ModelKind kind() {
        return ModelKind.STANDARD_SCALER;
    }

    @Override
    public void write(ModelWriter writer) {
        if (mean == null) {
            throw new IllegalStateException("StandardScaler must be fitted before serialization");
        }
        writer.putByte(withMean ? 1 : 0);
        writer.putByte(withStd ? 1 : 0);
        writer.putInt(featureCount);
        writer.putInt(samplesSeen);
        writer.putDoubles(mean);
        writer.putDoubles(scale);
        writer.putDoubles(var);
    }

    public static StandardScaler read(ModelReader reader) {
        boolean withMean = reader.getByte() != 0;
        boolean withStd = reader.getByte() != 0;
        int features = reader.getInt();
        int samplesSeen = reader.getInt();
        if (features <= 0 || samplesSeen <= 0) {
            throw new ModelFormatException("Invalid StandardScaler header");
        }
        StandardScaler scaler = new Builder().withMean(withMean).withStd(withStd).build();
        scaler.featureCount = features;
        scaler.samplesSeen = samplesSeen;
        scaler.mean = reader.getDoubles(features);
        scaler.scale = reader.getDoubles(features);
        scaler.var = reader.getDoubles(features);
        return scaler;
    }

    public static final class Builder {
        private boolean withMean = true;
        private boolean withStd = true;

        public Builder withMean(boolean withMean) {
            this.withMean = withMean;
            return this;
        }

        public Builder withStd(boolean withStd) {
            this.withStd = withStd;
            return this;
        }

        public StandardScaler build() {
            return new StandardScaler(this);
        }
    }
}
