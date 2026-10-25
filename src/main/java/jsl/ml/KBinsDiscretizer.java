package jsl.ml;

import jsl.core.Matrix;
import jsl.core.ModelFormatException;
import jsl.core.ModelKind;
import jsl.core.ModelReader;
import jsl.core.ModelWriter;
import java.util.Arrays;
import java.util.Objects;

/**
 * Bins continuous data into intervals using {@code QUANTILE}, {@code UNIFORM}, or {@code KMEANS}
 * strategies and outputs {@code ORDINAL} or {@code ONEHOT_DENSE} encodings, matching
 * {@code sklearn.preprocessing.KBinsDiscretizer}.
 */
public final class KBinsDiscretizer implements Transformer, Estimator {
    public enum Strategy {
        UNIFORM,
        QUANTILE,
        KMEANS
    }

    public enum Encode {
        ORDINAL,
        ONEHOT_DENSE
    }

    private final int nBins;
    private final Strategy strategy;
    private final Encode encode;

    private int inputFeatures;
    private double[][] binEdges; // [feature][nBins + 1]

    private KBinsDiscretizer(Builder builder) {
        this.nBins = builder.nBins;
        this.strategy = builder.strategy;
        this.encode = builder.encode;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public KBinsDiscretizer fit(Matrix features) {
        return fit(features, null);
    }

    @Override
    public KBinsDiscretizer fit(Matrix features, double[] ignoredTargets) {
        MlChecks.features(features);
        int rows = features.rows();
        inputFeatures = features.columns();
        binEdges = new double[inputFeatures][nBins + 1];

        for (int c = 0; c < inputFeatures; c++) {
            double[] col = new double[rows];
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            for (int r = 0; r < rows; r++) {
                double v = features.get(r, c);
                col[r] = v;
                if (v < min) min = v;
                if (v > max) max = v;
            }
            double[] edges = binEdges[c];
            edges[0] = min;
            edges[nBins] = max;
            if (max == min) {
                for (int b = 1; b < nBins; b++) edges[b] = min;
                continue;
            }
            switch (strategy) {
                case UNIFORM -> {
                    double step = (max - min) / nBins;
                    for (int b = 1; b < nBins; b++) {
                        edges[b] = min + b * step;
                    }
                }
                case QUANTILE -> {
                    Arrays.sort(col);
                    for (int b = 1; b < nBins; b++) {
                        double pos = (rows - 1.0) * b / nBins;
                        int lo = (int) Math.floor(pos);
                        int hi = (int) Math.ceil(pos);
                        double frac = pos - lo;
                        edges[b] = col[lo] * (1.0 - frac) + col[hi] * frac;
                    }
                }
                case KMEANS -> {
                    KMeans km = KMeans.builder().clusters(nBins).maxIterations(50).seed(42L).build()
                            .fit(new Matrix(rows, 1, col));
                    double[] centers = km.centroids().toArray();
                    Arrays.sort(centers);
                    for (int b = 1; b < nBins; b++) {
                        edges[b] = 0.5 * (centers[b - 1] + centers[b]);
                    }
                }
            }
        }
        return this;
    }

    private int findBin(int feature, double value) {
        double[] edges = binEdges[feature];
        for (int b = 1; b < nBins; b++) {
            if (value < edges[b]) {
                return b - 1;
            }
        }
        return nBins - 1;
    }

    @Override
    public Matrix transform(Matrix features) {
        MlChecks.predictionData(features, inputFeatures, binEdges != null);
        int rows = features.rows();
        if (encode == Encode.ORDINAL) {
            double[] dst = new double[rows * inputFeatures];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < inputFeatures; c++) {
                    dst[r * inputFeatures + c] = findBin(c, features.get(r, c));
                }
            }
            return new Matrix(rows, inputFeatures, dst);
        }
        int outCols = inputFeatures * nBins;
        double[] dst = new double[rows * outCols];
        for (int r = 0; r < rows; r++) {
            int rowBase = r * outCols;
            for (int c = 0; c < inputFeatures; c++) {
                int b = findBin(c, features.get(r, c));
                dst[rowBase + c * nBins + b] = 1.0;
            }
        }
        return new Matrix(rows, outCols, dst);
    }

    @Override
    public Matrix inverseTransform(Matrix transformed) {
        if (binEdges == null) throw new IllegalStateException("KBinsDiscretizer is not fitted");
        int rows = transformed.rows();
        double[] dst = new double[rows * inputFeatures];
        if (encode == Encode.ORDINAL) {
            MlChecks.predictionData(transformed, inputFeatures, true);
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < inputFeatures; c++) {
                    int b = Math.max(0, Math.min(nBins - 1, (int) Math.round(transformed.get(r, c))));
                    dst[r * inputFeatures + c] = 0.5 * (binEdges[c][b] + binEdges[c][b + 1]);
                }
            }
            return new Matrix(rows, inputFeatures, dst);
        }
        MlChecks.predictionData(transformed, inputFeatures * nBins, true);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < inputFeatures; c++) {
                int bestBin = 0;
                double bestVal = Double.NEGATIVE_INFINITY;
                for (int b = 0; b < nBins; b++) {
                    double v = transformed.get(r, c * nBins + b);
                    if (v > bestVal) {
                        bestVal = v;
                        bestBin = b;
                    }
                }
                dst[r * inputFeatures + c] = 0.5 * (binEdges[c][bestBin] + binEdges[c][bestBin + 1]);
            }
        }
        return new Matrix(rows, inputFeatures, dst);
    }

    @Override
    public double[] predict(Matrix features) {
        return transform(features).toArray();
    }

    public double[][] binEdges() {
        if (binEdges == null) throw new IllegalStateException("KBinsDiscretizer is not fitted");
        double[][] copy = new double[binEdges.length][];
        for (int i = 0; i < binEdges.length; i++) copy[i] = binEdges[i].clone();
        return copy;
    }

    @Override
    public ModelKind kind() {
        return ModelKind.KBINS_DISCRETIZER;
    }

    @Override
    public void write(ModelWriter writer) {
        if (binEdges == null) {
            throw new IllegalStateException("KBinsDiscretizer must be fitted before serialization");
        }
        writer.putInt(nBins);
        writer.putInt(strategy.ordinal());
        writer.putInt(encode.ordinal());
        writer.putInt(inputFeatures);
        for (int c = 0; c < inputFeatures; c++) {
            writer.putDoubles(binEdges[c]);
        }
    }

    public static KBinsDiscretizer read(ModelReader reader) {
        int bins = reader.getInt();
        int stratOrd = reader.getInt();
        int encOrd = reader.getInt();
        int inFeatures = reader.getInt();
        if (bins < 2 || inFeatures <= 0
                || stratOrd < 0 || stratOrd >= Strategy.values().length
                || encOrd < 0 || encOrd >= Encode.values().length) {
            throw new ModelFormatException("Invalid KBinsDiscretizer header");
        }
        KBinsDiscretizer disc = new Builder()
                .nBins(bins)
                .strategy(Strategy.values()[stratOrd])
                .encode(Encode.values()[encOrd])
                .build();
        disc.inputFeatures = inFeatures;
        disc.binEdges = new double[inFeatures][];
        for (int c = 0; c < inFeatures; c++) {
            disc.binEdges[c] = reader.getDoubles(bins + 1);
        }
        return disc;
    }

    public static final class Builder {
        private int nBins = 5;
        private Strategy strategy = Strategy.QUANTILE;
        private Encode encode = Encode.ORDINAL;

        public Builder nBins(int nBins) {
            if (nBins < 2) throw new IllegalArgumentException("nBins must be >= 2");
            this.nBins = nBins;
            return this;
        }

        public Builder strategy(Strategy strategy) {
            this.strategy = Objects.requireNonNull(strategy, "strategy");
            return this;
        }

        public Builder encode(Encode encode) {
            this.encode = Objects.requireNonNull(encode, "encode");
            return this;
        }

        public KBinsDiscretizer build() {
            return new KBinsDiscretizer(this);
        }
    }
}
