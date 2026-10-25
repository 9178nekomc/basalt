package basalt.data;

/** Immutable precomputed standardization parameters; application performs no allocation. */
public final class Normalization {
    private final double[] means;
    private final double[] standardDeviations;
    private final double[] inverseStandardDeviations;

    private Normalization(double[] means, double[] standardDeviations, double[] inverseStandardDeviations) {
        this.means = means;
        this.standardDeviations = standardDeviations;
        this.inverseStandardDeviations = inverseStandardDeviations;
    }

    public static Normalization standardize(double[] means, double[] standardDeviations) {
        if (means.length == 0 || means.length != standardDeviations.length) {
            throw new IllegalArgumentException("Normalization statistics must be non-empty and equally sized");
        }
        double[] meanCopy = means.clone();
        double[] inverse = new double[standardDeviations.length];
        for (int i = 0; i < inverse.length; i++) {
            if (!Double.isFinite(meanCopy[i]) || !Double.isFinite(standardDeviations[i])
                    || standardDeviations[i] < 0.0) {
                throw new IllegalArgumentException("Normalization statistics must be finite and std non-negative");
            }
            inverse[i] = standardDeviations[i] == 0.0 ? 1.0 : 1.0 / standardDeviations[i];
        }
        return new Normalization(meanCopy, standardDeviations.clone(), inverse);
    }

    void apply(double[] features, int offset) {
        for (int feature = 0; feature < means.length; feature++) {
            features[offset + feature] = (features[offset + feature] - means[feature])
                    * inverseStandardDeviations[feature];
        }
    }

    public int featureCount() {
        return means.length;
    }

    public double[] means() {
        return means.clone();
    }

    public double[] standardDeviations() {
        return standardDeviations.clone();
    }
}
