package basalt.ml;

import basalt.core.Matrix;
import basalt.core.ModelFormatException;
import basalt.core.ModelKind;
import basalt.core.ModelReader;
import basalt.core.ModelWriter;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * Univariate imputer for completing missing values ({@link Double#NaN}) with {@code MEAN},
 * {@code MEDIAN}, {@code MOST_FREQUENT}, or {@code CONSTANT} strategies.
 */
public final class SimpleImputer implements Transformer, Estimator {
    public enum Strategy {
        MEAN,
        MEDIAN,
        MOST_FREQUENT,
        CONSTANT
    }

    private final Strategy strategy;
    private final double fillValue;
    private double[] statistics;
    private int featureCount;

    private SimpleImputer(Builder builder) {
        this.strategy = builder.strategy;
        this.fillValue = builder.fillValue;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public SimpleImputer fit(Matrix features) {
        return fit(features, null);
    }

    @Override
    public SimpleImputer fit(Matrix features, double[] ignoredTargets) {
        Objects.requireNonNull(features, "features");
        int rows = features.rows();
        featureCount = features.columns();
        statistics = new double[featureCount];
        double[] data = features.data();

        for (int c = 0; c < featureCount; c++) {
            double[] valid = new double[rows];
            int count = 0;
            for (int r = 0; r < rows; r++) {
                double v = data[r * featureCount + c];
                if (!Double.isNaN(v)) {
                    if (!Double.isFinite(v)) {
                        throw new IllegalArgumentException("SimpleImputer encountered infinite value at column " + c);
                    }
                    valid[count++] = v;
                }
            }
            if (strategy != Strategy.CONSTANT && count == 0) {
                throw new IllegalArgumentException("Column " + c + " contains only NaN values");
            }
            switch (strategy) {
                case CONSTANT -> statistics[c] = fillValue;
                case MEAN -> {
                    double sum = 0.0;
                    for (int i = 0; i < count; i++) sum += valid[i];
                    statistics[c] = sum / count;
                }
                case MEDIAN -> {
                    Arrays.sort(valid, 0, count);
                    if (count % 2 == 1) {
                        statistics[c] = valid[count / 2];
                    } else {
                        statistics[c] = 0.5 * (valid[count / 2 - 1] + valid[count / 2]);
                    }
                }
                case MOST_FREQUENT -> {
                    TreeMap<Double, Integer> freq = new TreeMap<>();
                    for (int i = 0; i < count; i++) {
                        freq.merge(valid[i], 1, Integer::sum);
                    }
                    double bestVal = valid[0];
                    int bestCount = -1;
                    for (Map.Entry<Double, Integer> entry : freq.entrySet()) {
                        if (entry.getValue() > bestCount) {
                            bestCount = entry.getValue();
                            bestVal = entry.getKey();
                        }
                    }
                    statistics[c] = bestVal;
                }
            }
        }
        return this;
    }

    @Override
    public Matrix transform(Matrix features) {
        Objects.requireNonNull(features, "features");
        if (statistics == null) {
            throw new IllegalStateException("SimpleImputer must be fitted before transform");
        }
        if (features.columns() != featureCount) {
            throw new IllegalArgumentException(
                    "Expected " + featureCount + " features but got " + features.columns());
        }
        int rows = features.rows();
        double[] src = features.data();
        double[] dst = new double[rows * featureCount];
        for (int r = 0; r < rows; r++) {
            int offset = r * featureCount;
            for (int c = 0; c < featureCount; c++) {
                double v = src[offset + c];
                if (Double.isNaN(v)) {
                    dst[offset + c] = statistics[c];
                } else if (!Double.isFinite(v)) {
                    throw new IllegalArgumentException("SimpleImputer encountered infinite value");
                } else {
                    dst[offset + c] = v;
                }
            }
        }
        return new Matrix(rows, featureCount, dst);
    }

    @Override
    public double[] predict(Matrix features) {
        return transform(features).toArray();
    }

    public double[] statistics() {
        if (statistics == null) throw new IllegalStateException("SimpleImputer is not fitted");
        return statistics.clone();
    }

    public Strategy strategy() {
        return strategy;
    }

    @Override
    public ModelKind kind() {
        return ModelKind.SIMPLE_IMPUTER;
    }

    @Override
    public void write(ModelWriter writer) {
        if (statistics == null) {
            throw new IllegalStateException("SimpleImputer must be fitted before serialization");
        }
        writer.putInt(strategy.ordinal());
        writer.putDouble(fillValue);
        writer.putInt(featureCount);
        writer.putDoubles(statistics);
    }

    public static SimpleImputer read(ModelReader reader) {
        int stratOrd = reader.getInt();
        if (stratOrd < 0 || stratOrd >= Strategy.values().length) {
            throw new ModelFormatException("Invalid SimpleImputer strategy ordinal");
        }
        double fillVal = reader.getDouble();
        int features = reader.getInt();
        if (features <= 0) {
            throw new ModelFormatException("Invalid SimpleImputer feature count");
        }
        SimpleImputer imputer = new Builder()
                .strategy(Strategy.values()[stratOrd])
                .fillValue(fillVal)
                .build();
        imputer.featureCount = features;
        imputer.statistics = reader.getDoubles(features);
        return imputer;
    }

    public static final class Builder {
        private Strategy strategy = Strategy.MEAN;
        private double fillValue = 0.0;

        public Builder strategy(Strategy strategy) {
            this.strategy = Objects.requireNonNull(strategy, "strategy");
            return this;
        }

        public Builder fillValue(double fillValue) {
            if (!Double.isFinite(fillValue)) {
                throw new IllegalArgumentException("fillValue must be finite");
            }
            this.fillValue = fillValue;
            return this;
        }

        public SimpleImputer build() {
            return new SimpleImputer(this);
        }
    }
}
