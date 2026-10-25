package jsl.ml;

import jsl.core.Matrix;
import jsl.core.ModelFormatException;
import jsl.core.ModelKind;
import jsl.core.ModelReader;
import jsl.core.ModelWriter;
import jsl.core.Persistable;
import java.util.Arrays;

/**
 * Gaussian Mixture Model (GMM) estimated with the Expectation-Maximization (EM) algorithm,
 * seeded via K-Means++ initialization, matching {@code sklearn.mixture.GaussianMixture}.
 */
public final class GaussianMixture implements Estimator, Persistable {
    private final int components;
    private final int maxIterations;
    private final double tolerance;
    private final double regCovar;
    private final long seed;

    private int featureCount;
    private double[] weights;
    private double[] means;     // [components * featureCount]
    private double[] variances; // [components * featureCount] diagonal covariance for stability & speed
    private boolean converged;
    private int iterations;
    private double lowerBound;

    private GaussianMixture(Builder builder) {
        this.components = builder.components;
        this.maxIterations = builder.maxIterations;
        this.tolerance = builder.tolerance;
        this.regCovar = builder.regCovar;
        this.seed = builder.seed;
    }

    public static Builder builder() {
        return new Builder();
    }

    public GaussianMixture fit(Matrix features) {
        return fit(features, null);
    }

    @Override
    public GaussianMixture fit(Matrix features, double[] ignoredTargets) {
        MlChecks.features(features);
        int rows = features.rows();
        featureCount = features.columns();
        if (components > rows) {
            throw new IllegalArgumentException("components cannot exceed sample count");
        }

        // Initialize means using KMeans++
        KMeans kmeans = KMeans.builder().clusters(components).maxIterations(30).seed(seed).build().fit(features);
        means = kmeans.centroids().toArray();
        weights = new double[components];
        Arrays.fill(weights, 1.0 / components);
        variances = new double[components * featureCount];

        double[] data = features.data();
        double[] clusterAssignments = kmeans.predict(features);
        int[] counts = new int[components];
        for (int r = 0; r < rows; r++) {
            int k = (int) clusterAssignments[r];
            counts[k]++;
            for (int f = 0; f < featureCount; f++) {
                double d = data[r * featureCount + f] - means[k * featureCount + f];
                variances[k * featureCount + f] += d * d;
            }
        }
        for (int k = 0; k < components; k++) {
            weights[k] = counts[k] > 0 ? (double) counts[k] / rows : 1.0 / components;
            double denom = Math.max(1, counts[k]);
            for (int f = 0; f < featureCount; f++) {
                variances[k * featureCount + f] = Math.max(regCovar, variances[k * featureCount + f] / denom + regCovar);
            }
        }

        double[] logResp = new double[rows * components];
        double[] resp = new double[rows * components];
        double prevBound = Double.NEGATIVE_INFINITY;
        converged = false;

        for (int iter = 1; iter <= maxIterations; iter++) {
            iterations = iter;
            // E-step
            double totalLogProb = 0.0;
            for (int r = 0; r < rows; r++) {
                int rBase = r * components;
                double maxLog = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < components; k++) {
                    double lp = Math.log(Math.max(1e-15, weights[k]))
                            + logGaussianDiag(data, r * featureCount, k);
                    logResp[rBase + k] = lp;
                    if (lp > maxLog) maxLog = lp;
                }
                double sumExp = 0.0;
                for (int k = 0; k < components; k++) {
                    double e = Math.exp(logResp[rBase + k] - maxLog);
                    resp[rBase + k] = e;
                    sumExp += e;
                }
                double logNorm = maxLog + Math.log(sumExp);
                totalLogProb += logNorm;
                for (int k = 0; k < components; k++) {
                    resp[rBase + k] /= sumExp;
                }
            }
            lowerBound = totalLogProb / rows;
            if (Math.abs(lowerBound - prevBound) <= tolerance) {
                converged = true;
                break;
            }
            prevBound = lowerBound;

            // M-step
            double[] nk = new double[components];
            Arrays.fill(means, 0.0);
            for (int r = 0; r < rows; r++) {
                for (int k = 0; k < components; k++) {
                    double w = resp[r * components + k];
                    nk[k] += w;
                    for (int f = 0; f < featureCount; f++) {
                        means[k * featureCount + f] += w * data[r * featureCount + f];
                    }
                }
            }
            for (int k = 0; k < components; k++) {
                double safeNk = Math.max(nk[k], 1e-10);
                weights[k] = safeNk / rows;
                for (int f = 0; f < featureCount; f++) {
                    means[k * featureCount + f] /= safeNk;
                }
            }
            Arrays.fill(variances, 0.0);
            for (int r = 0; r < rows; r++) {
                for (int k = 0; k < components; k++) {
                    double w = resp[r * components + k];
                    for (int f = 0; f < featureCount; f++) {
                        double d = data[r * featureCount + f] - means[k * featureCount + f];
                        variances[k * featureCount + f] += w * d * d;
                    }
                }
            }
            for (int k = 0; k < components; k++) {
                double safeNk = Math.max(nk[k], 1e-10);
                for (int f = 0; f < featureCount; f++) {
                    variances[k * featureCount + f] = variances[k * featureCount + f] / safeNk + regCovar;
                }
            }
        }
        return this;
    }

    private double logGaussianDiag(double[] x, int xOff, int component) {
        int base = component * featureCount;
        double logDet = 0.0;
        double maha = 0.0;
        for (int f = 0; f < featureCount; f++) {
            double var = variances[base + f];
            double d = x[xOff + f] - means[base + f];
            logDet += Math.log(var);
            maha += (d * d) / var;
        }
        return -0.5 * (featureCount * Math.log(2.0 * Math.PI) + logDet + maha);
    }

    /** Per-sample log probability density {@code log p(x)}. */
    public double[] scoreSamples(Matrix features) {
        MlChecks.predictionData(features, featureCount, weights != null);
        int rows = features.rows();
        double[] data = features.data();
        double[] scores = new double[rows];
        for (int r = 0; r < rows; r++) {
            double maxLog = Double.NEGATIVE_INFINITY;
            double[] lps = new double[components];
            for (int k = 0; k < components; k++) {
                lps[k] = Math.log(Math.max(1e-15, weights[k]))
                        + logGaussianDiag(data, r * featureCount, k);
                if (lps[k] > maxLog) maxLog = lps[k];
            }
            double sumExp = 0.0;
            for (int k = 0; k < components; k++) {
                sumExp += Math.exp(lps[k] - maxLog);
            }
            scores[r] = maxLog + Math.log(sumExp);
        }
        return scores;
    }

    public Matrix predictProbability(Matrix features) {
        MlChecks.predictionData(features, featureCount, weights != null);
        int rows = features.rows();
        double[] data = features.data();
        double[] probs = new double[rows * components];
        for (int r = 0; r < rows; r++) {
            int rBase = r * components;
            double maxLog = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < components; k++) {
                double lp = Math.log(Math.max(1e-15, weights[k]))
                        + logGaussianDiag(data, r * featureCount, k);
                probs[rBase + k] = lp;
                if (lp > maxLog) maxLog = lp;
            }
            double sumExp = 0.0;
            for (int k = 0; k < components; k++) {
                probs[rBase + k] = Math.exp(probs[rBase + k] - maxLog);
                sumExp += probs[rBase + k];
            }
            for (int k = 0; k < components; k++) {
                probs[rBase + k] /= sumExp;
            }
        }
        return new Matrix(rows, components, probs);
    }

    @Override
    public double[] predict(Matrix features) {
        Matrix probs = predictProbability(features);
        int rows = probs.rows();
        double[] labels = new double[rows];
        for (int r = 0; r < rows; r++) {
            int best = 0;
            for (int k = 1; k < components; k++) {
                if (probs.get(r, k) > probs.get(r, best)) {
                    best = k;
                }
            }
            labels[r] = best;
        }
        return labels;
    }

    public double aic(Matrix features) {
        double[] logProbs = scoreSamples(features);
        double logLik = 0.0;
        for (double v : logProbs) logLik += v;
        int nParams = components * 2 * featureCount + (components - 1);
        return 2.0 * nParams - 2.0 * logLik;
    }

    public double bic(Matrix features) {
        double[] logProbs = scoreSamples(features);
        double logLik = 0.0;
        for (double v : logProbs) logLik += v;
        int nParams = components * 2 * featureCount + (components - 1);
        return nParams * Math.log(features.rows()) - 2.0 * logLik;
    }

    public double[] weights() {
        return weights == null ? null : weights.clone();
    }

    public Matrix means() {
        if (means == null) throw new IllegalStateException("GaussianMixture is not fitted");
        return new Matrix(components, featureCount, means);
    }

    public Matrix variances() {
        if (variances == null) throw new IllegalStateException("GaussianMixture is not fitted");
        return new Matrix(components, featureCount, variances);
    }

    public boolean converged() {
        return converged;
    }

    public int iterations() {
        return iterations;
    }

    public double lowerBound() {
        return lowerBound;
    }

    @Override
    public ModelKind kind() {
        return ModelKind.GAUSSIAN_MIXTURE;
    }

    @Override
    public void write(ModelWriter writer) {
        if (weights == null) {
            throw new IllegalStateException("GaussianMixture must be fitted before serialization");
        }
        writer.putInt(components);
        writer.putInt(featureCount);
        writer.putByte(converged ? 1 : 0);
        writer.putInt(iterations);
        writer.putDouble(lowerBound);
        writer.putDoubles(weights);
        writer.putDoubles(means);
        writer.putDoubles(variances);
    }

    public static GaussianMixture read(ModelReader reader) {
        int comps = reader.getInt();
        int features = reader.getInt();
        boolean conv = reader.getByte() != 0;
        int iters = reader.getInt();
        double lb = reader.getDouble();
        if (comps <= 0 || features <= 0) {
            throw new ModelFormatException("Invalid GaussianMixture header");
        }
        GaussianMixture gmm = new Builder().components(comps).build();
        gmm.featureCount = features;
        gmm.converged = conv;
        gmm.iterations = iters;
        gmm.lowerBound = lb;
        gmm.weights = reader.getDoubles(comps);
        gmm.means = reader.getDoubles(Math.multiplyExact(comps, features));
        gmm.variances = reader.getDoubles(Math.multiplyExact(comps, features));
        return gmm;
    }

    public static final class Builder {
        private int components = 2;
        private int maxIterations = 100;
        private double tolerance = 1e-4;
        private double regCovar = 1e-6;
        private long seed = 42L;

        public Builder components(int components) {
            if (components <= 0) throw new IllegalArgumentException("components must be positive");
            this.components = components;
            return this;
        }

        public Builder maxIterations(int maxIterations) {
            if (maxIterations <= 0) throw new IllegalArgumentException("maxIterations must be positive");
            this.maxIterations = maxIterations;
            return this;
        }

        public Builder tolerance(double tolerance) {
            MlChecks.positiveFinite(tolerance, "tolerance");
            this.tolerance = tolerance;
            return this;
        }

        public Builder regCovar(double regCovar) {
            MlChecks.positiveFinite(regCovar, "regCovar");
            this.regCovar = regCovar;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public GaussianMixture build() {
            return new GaussianMixture(this);
        }
    }
}
