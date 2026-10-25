package basalt.ml;

import basalt.core.Matrix;
import basalt.core.ModelFormatException;
import basalt.core.ModelKind;
import basalt.core.ModelReader;
import basalt.core.ModelWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Generates polynomial and interaction features up to degree {@code degree}, matching
 * {@code sklearn.preprocessing.PolynomialFeatures}.
 */
public final class PolynomialFeatures implements Transformer, Estimator {
    private final int degree;
    private final boolean interactionOnly;
    private final boolean includeBias;

    private int inputFeatures;
    private List<int[]> combinations;

    private PolynomialFeatures(Builder builder) {
        this.degree = builder.degree;
        this.interactionOnly = builder.interactionOnly;
        this.includeBias = builder.includeBias;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public PolynomialFeatures fit(Matrix features) {
        return fit(features, null);
    }

    @Override
    public PolynomialFeatures fit(Matrix features, double[] ignoredTargets) {
        MlChecks.features(features);
        inputFeatures = features.columns();
        combinations = buildCombinations(inputFeatures, degree, interactionOnly, includeBias);
        return this;
    }

    private static List<int[]> buildCombinations(int nFeatures, int maxDeg,
                                                 boolean interactionOnly, boolean includeBias) {
        List<int[]> list = new ArrayList<>();
        if (includeBias) {
            list.add(new int[0]);
        }
        for (int deg = 1; deg <= maxDeg; deg++) {
            generateCombos(list, new int[deg], 0, 0, nFeatures, interactionOnly);
        }
        return list;
    }

    private static void generateCombos(List<int[]> out, int[] current, int pos, int startFeature,
                                       int nFeatures, boolean interactionOnly) {
        if (pos == current.length) {
            out.add(current.clone());
            return;
        }
        for (int f = startFeature; f < nFeatures; f++) {
            current[pos] = f;
            generateCombos(out, current, pos + 1, interactionOnly ? f + 1 : f,
                    nFeatures, interactionOnly);
        }
    }

    @Override
    public Matrix transform(Matrix features) {
        MlChecks.predictionData(features, inputFeatures, combinations != null);
        int rows = features.rows();
        int outCols = combinations.size();
        double[] dst = new double[rows * outCols];
        double[] src = features.data();
        for (int r = 0; r < rows; r++) {
            int inBase = r * inputFeatures;
            int outBase = r * outCols;
            for (int c = 0; c < outCols; c++) {
                int[] combo = combinations.get(c);
                double prod = 1.0;
                for (int fIdx : combo) {
                    prod *= src[inBase + fIdx];
                }
                dst[outBase + c] = prod;
            }
        }
        return new Matrix(rows, outCols, dst);
    }

    @Override
    public double[] predict(Matrix features) {
        return transform(features).toArray();
    }

    public int outputFeatureCount() {
        if (combinations == null) throw new IllegalStateException("PolynomialFeatures is not fitted");
        return combinations.size();
    }

    @Override
    public ModelKind kind() {
        return ModelKind.POLYNOMIAL_FEATURES;
    }

    @Override
    public void write(ModelWriter writer) {
        if (combinations == null) {
            throw new IllegalStateException("PolynomialFeatures must be fitted before serialization");
        }
        writer.putInt(degree);
        writer.putByte(interactionOnly ? 1 : 0);
        writer.putByte(includeBias ? 1 : 0);
        writer.putInt(inputFeatures);
    }

    public static PolynomialFeatures read(ModelReader reader) {
        int deg = reader.getInt();
        boolean inter = reader.getByte() != 0;
        boolean bias = reader.getByte() != 0;
        int inFeatures = reader.getInt();
        if (deg < 1 || inFeatures <= 0) {
            throw new ModelFormatException("Invalid PolynomialFeatures header");
        }
        PolynomialFeatures pf = new Builder()
                .degree(deg)
                .interactionOnly(inter)
                .includeBias(bias)
                .build();
        pf.inputFeatures = inFeatures;
        pf.combinations = buildCombinations(inFeatures, deg, inter, bias);
        return pf;
    }

    public static final class Builder {
        private int degree = 2;
        private boolean interactionOnly = false;
        private boolean includeBias = true;

        public Builder degree(int degree) {
            if (degree < 1) throw new IllegalArgumentException("degree must be >= 1");
            this.degree = degree;
            return this;
        }

        public Builder interactionOnly(boolean interactionOnly) {
            this.interactionOnly = interactionOnly;
            return this;
        }

        public Builder includeBias(boolean includeBias) {
            this.includeBias = includeBias;
            return this;
        }

        public PolynomialFeatures build() {
            return new PolynomialFeatures(this);
        }
    }
}
