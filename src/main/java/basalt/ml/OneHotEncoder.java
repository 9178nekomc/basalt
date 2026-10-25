package basalt.ml;

import basalt.core.Matrix;
import basalt.core.ModelFormatException;
import basalt.core.ModelKind;
import basalt.core.ModelReader;
import basalt.core.ModelWriter;
import java.util.Arrays;
import java.util.Objects;
import java.util.TreeSet;

/**
 * One-hot encodes categorical columns into dense binary indicator features, matching
 * {@code sklearn.preprocessing.OneHotEncoder(sparse_output=False)}.
 *
 * <p>Supports encoding all columns or a specified subset of categorical column indices while
 * passing numeric columns through unchanged. Supports {@link HandleUnknown#ERROR} and
 * {@link HandleUnknown#IGNORE} (all-zeros indicator for unseen categories), plus
 * {@link #inverseTransform(Matrix)}.</p>
 */
public final class OneHotEncoder implements Transformer, Estimator {
    public enum HandleUnknown {
        ERROR,
        IGNORE
    }

    private final HandleUnknown handleUnknown;
    private final int[] categoricalColumns;
    private boolean[] isCategorical;
    private double[][] categories;
    private int[] outputOffsets;
    private int inputFeatures;
    private int totalOutputFeatures;

    private OneHotEncoder(Builder builder) {
        this.handleUnknown = builder.handleUnknown;
        this.categoricalColumns = builder.categoricalColumns == null
                ? null : builder.categoricalColumns.clone();
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public OneHotEncoder fit(Matrix features) {
        return fit(features, null);
    }

    @Override
    public OneHotEncoder fit(Matrix features, double[] ignoredTargets) {
        MlChecks.features(features);
        int rows = features.rows();
        inputFeatures = features.columns();
        isCategorical = new boolean[inputFeatures];
        if (categoricalColumns == null) {
            Arrays.fill(isCategorical, true);
        } else {
            for (int col : categoricalColumns) {
                if (col < 0 || col >= inputFeatures) {
                    throw new IllegalArgumentException("Categorical column index out of bounds: " + col);
                }
                isCategorical[col] = true;
            }
        }
        categories = new double[inputFeatures][];
        outputOffsets = new int[inputFeatures + 1];
        int currentOffset = 0;
        for (int c = 0; c < inputFeatures; c++) {
            outputOffsets[c] = currentOffset;
            if (isCategorical[c]) {
                TreeSet<Double> unique = new TreeSet<>();
                for (int r = 0; r < rows; r++) {
                    unique.add(features.get(r, c));
                }
                double[] cats = new double[unique.size()];
                int idx = 0;
                for (double v : unique) {
                    cats[idx++] = v;
                }
                categories[c] = cats;
                currentOffset += cats.length;
            } else {
                categories[c] = new double[0];
                currentOffset += 1;
            }
        }
        outputOffsets[inputFeatures] = currentOffset;
        totalOutputFeatures = currentOffset;
        return this;
    }

    @Override
    public Matrix transform(Matrix features) {
        MlChecks.predictionData(features, inputFeatures, categories != null);
        int rows = features.rows();
        double[] dst = new double[rows * totalOutputFeatures];
        for (int r = 0; r < rows; r++) {
            int rowBase = r * totalOutputFeatures;
            for (int c = 0; c < inputFeatures; c++) {
                double v = features.get(r, c);
                int colOutBase = rowBase + outputOffsets[c];
                if (isCategorical[c]) {
                    double[] cats = categories[c];
                    int found = -1;
                    for (int k = 0; k < cats.length; k++) {
                        if (Double.compare(cats[k], v) == 0) {
                            found = k;
                            break;
                        }
                    }
                    if (found >= 0) {
                        dst[colOutBase + found] = 1.0;
                    } else if (handleUnknown == HandleUnknown.ERROR) {
                        throw new IllegalArgumentException(
                                "Found unknown category " + v + " in column " + c + " during transform");
                    }
                } else {
                    dst[colOutBase] = v;
                }
            }
        }
        return new Matrix(rows, totalOutputFeatures, dst);
    }

    @Override
    public Matrix inverseTransform(Matrix transformed) {
        MlChecks.predictionData(transformed, totalOutputFeatures, categories != null);
        int rows = transformed.rows();
        double[] dst = new double[rows * inputFeatures];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < inputFeatures; c++) {
                int colOutBase = outputOffsets[c];
                if (isCategorical[c]) {
                    double[] cats = categories[c];
                    int bestIdx = -1;
                    double bestVal = 0.0;
                    for (int k = 0; k < cats.length; k++) {
                        double val = transformed.get(r, colOutBase + k);
                        if (val > bestVal) {
                            bestVal = val;
                            bestIdx = k;
                        }
                    }
                    dst[r * inputFeatures + c] = bestIdx >= 0 ? cats[bestIdx] : Double.NaN;
                } else {
                    dst[r * inputFeatures + c] = transformed.get(r, colOutBase);
                }
            }
        }
        return new Matrix(rows, inputFeatures, dst);
    }

    @Override
    public double[] predict(Matrix features) {
        return transform(features).toArray();
    }

    public double[][] categories() {
        if (categories == null) throw new IllegalStateException("OneHotEncoder is not fitted");
        double[][] copy = new double[categories.length][];
        for (int i = 0; i < categories.length; i++) {
            copy[i] = categories[i].clone();
        }
        return copy;
    }

    public int outputFeatureCount() {
        return totalOutputFeatures;
    }

    @Override
    public ModelKind kind() {
        return ModelKind.ONE_HOT_ENCODER;
    }

    @Override
    public void write(ModelWriter writer) {
        if (categories == null) {
            throw new IllegalStateException("OneHotEncoder must be fitted before serialization");
        }
        writer.putInt(handleUnknown.ordinal());
        writer.putInt(inputFeatures);
        writer.putInt(totalOutputFeatures);
        for (int c = 0; c < inputFeatures; c++) {
            writer.putByte(isCategorical[c] ? 1 : 0);
            writer.putInt(categories[c].length);
            writer.putDoubles(categories[c]);
        }
        writer.putInts(outputOffsets);
    }

    public static OneHotEncoder read(ModelReader reader) {
        int unknownOrd = reader.getInt();
        if (unknownOrd < 0 || unknownOrd >= HandleUnknown.values().length) {
            throw new ModelFormatException("Invalid HandleUnknown ordinal");
        }
        int inFeatures = reader.getInt();
        int outFeatures = reader.getInt();
        if (inFeatures <= 0 || outFeatures <= 0) {
            throw new ModelFormatException("Invalid OneHotEncoder feature counts");
        }
        OneHotEncoder encoder = new Builder().handleUnknown(HandleUnknown.values()[unknownOrd]).build();
        encoder.inputFeatures = inFeatures;
        encoder.totalOutputFeatures = outFeatures;
        encoder.isCategorical = new boolean[inFeatures];
        encoder.categories = new double[inFeatures][];
        for (int c = 0; c < inFeatures; c++) {
            encoder.isCategorical[c] = reader.getByte() != 0;
            int len = reader.getInt();
            encoder.categories[c] = reader.getDoubles(len);
        }
        encoder.outputOffsets = reader.getInts(inFeatures + 1);
        return encoder;
    }

    public static final class Builder {
        private HandleUnknown handleUnknown = HandleUnknown.ERROR;
        private int[] categoricalColumns;

        public Builder handleUnknown(HandleUnknown handleUnknown) {
            this.handleUnknown = Objects.requireNonNull(handleUnknown, "handleUnknown");
            return this;
        }

        public Builder categoricalColumns(int... columns) {
            this.categoricalColumns = columns == null ? null : columns.clone();
            return this;
        }

        public OneHotEncoder build() {
            return new OneHotEncoder(this);
        }
    }
}
