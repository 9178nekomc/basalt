package basalt.ml;

import basalt.core.Matrix;
import basalt.core.ModelFormatException;
import basalt.core.ModelKind;
import basalt.core.ModelReader;
import basalt.core.ModelWriter;
import basalt.core.Persistable;
import java.util.Arrays;

/**
 * Gaussian Mixture Model (GMM) estimated with the Expectation-Maximization (EM) algorithm,
 * seeded via K-Means++ initialization, matching {@code sklearn.mixture.GaussianMixture}.
 *
 * <p>Three covariance parametrisations are supported:
 * {@link CovarianceType#FULL} (one unconstrained p x p matrix per component),
 * {@link CovarianceType#TIED} (a single shared matrix) and
 * {@link CovarianceType#DIAG} (one variance per feature per component).
 */
public final class GaussianMixture implements Estimator, Persistable {

    /** Covariance parametrisation, mirroring {@code sklearn.mixture.GaussianMixture.covariance_type}. */
    public enum CovarianceType {
        /** One unconstrained p x p covariance matrix per component. */
        FULL,
        /** A single covariance matrix shared by every component. */
        TIED,
        /** One variance per feature per component (diagonal covariance). */
        DIAG
    }

    private final int components;
    private final int maxIterations;
    private final double tolerance;
    private final double regCovar;
    private final long seed;
    private final CovarianceType covarianceType;

    private int featureCount;
    private double[] weights;
    private double[] means;        // [components * featureCount]
    private double[] variances;    // [components * featureCount] diagonal of each covariance
    private double[] covariances;  // [components * featureCount * featureCount]
    private double[] cholesky;     // [components * featureCount * featureCount] lower factor
    private double[] logDet;       // [components]
    private boolean converged;
    private int iterations;
    private double lowerBound;

    private GaussianMixture(Builder builder) {
        this.components = builder.components;
        this.maxIterations = builder.maxIterations;
        this.tolerance = builder.tolerance;
        this.regCovar = builder.regCovar;
        this.seed = builder.seed;
        this.covarianceType = builder.covarianceType;
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

        KMeans kmeans = KMeans.builder().clusters(components).maxIterations(30).seed(seed).build().fit(features);
        means = kmeans.centroids().toArray();
        weights = new double[components];
        Arrays.fill(weights, 1.0 / components);
        variances = new double[components * featureCount];
        covariances = new double[components * featureCount * featureCount];
        cholesky = new double[components * featureCount * featureCount];
        logDet = new double[components];

        double[] data = features.data();
        double[] clusterAssignments = kmeans.predict(features);
        int[] counts = new int[components];
        for (int r = 0; r < rows; r++) {
            counts[(int) clusterAssignments[r]]++;
        }
        for (int k = 0; k < components; k++) {
            weights[k] = counts[k] > 0 ? (double) counts[k] / rows : 1.0 / components;
        }
        estimateCovariances(data, respHard(clusterAssignments, rows), counts, rows);

        double[] logResp = new double[rows * components];
        double[] resp = new double[rows * components];
        double prevBound = Double.NEGATIVE_INFINITY;
        converged = false;

        for (int iter = 1; iter <= maxIterations; iter++) {
            iterations = iter;
            refreshFactors();
            // E-step
            double totalLogProb = 0.0;
            for (int r = 0; r < rows; r++) {
                int rBase = r * components;
                double maxLog = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < components; k++) {
                    double lp = Math.log(Math.max(1e-15, weights[k]))
                            + logGaussian(data, r * featureCount, k);
                    logResp[rBase + k] = lp;
                    if (lp > maxLog) {
                        maxLog = lp;
                    }
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

            // M-step: weights and means
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
            estimateCovariances(data, resp, null, rows);
        }
        refreshFactors();
        return this;
    }

    private double[] respHard(double[] assignments, int rows) {
        double[] resp = new double[rows * components];
        for (int r = 0; r < rows; r++) {
            resp[r * components + (int) assignments[r]] = 1.0;
        }
        return resp;
    }

    /**
     * Re-estimates {@code covariances}/{@code variances} from responsibilities.
     *
     * @param resp    responsibilities, or {@code null} for a hard assignment pass
     * @param counts  optional cluster sizes used by the initialization pass
     */
    private void estimateCovariances(double[] data, double[] resp, int[] counts, int rows) {
        int p = featureCount;
        Arrays.fill(covariances, 0.0);
        double[] nk = new double[components];
        if (counts != null) {
            for (int k = 0; k < components; k++) {
                nk[k] = counts[k];
            }
        } else {
            for (int r = 0; r < rows; r++) {
                for (int k = 0; k < components; k++) {
                    nk[k] += resp[r * components + k];
                }
            }
        }

        double[] scratch = new double[p * p];
        for (int r = 0; r < rows; r++) {
            int rOff = r * p;
            for (int k = 0; k < components; k++) {
                double w = resp[r * components + k];
                if (w == 0.0) {
                    continue;
                }
                int kOff = k * p * p;
                for (int a = 0; a < p; a++) {
                    double da = data[rOff + a] - means[k * p + a];
                    double wa = w * da;
                    int rowOff = kOff + a * p;
                    for (int b = 0; b <= a; b++) {
                        covariances[rowOff + b] += wa * (data[rOff + b] - means[k * p + b]);
                    }
                }
            }
        }

        if (covarianceType == CovarianceType.TIED) {
            Arrays.fill(scratch, 0.0);
            for (int k = 0; k < components; k++) {
                int kOff = k * p * p;
                for (int a = 0; a < p; a++) {
                    for (int b = 0; b <= a; b++) {
                        scratch[a * p + b] += covariances[kOff + a * p + b];
                    }
                }
            }
            double denom = Math.max(1e-10, rows);
            for (int a = 0; a < p; a++) {
                for (int b = 0; b <= a; b++) {
                    scratch[a * p + b] /= denom;
                }
            }
            for (int k = 0; k < components; k++) {
                System.arraycopy(scratch, 0, covariances, k * p * p, p * p);
            }
        } else {
            for (int k = 0; k < components; k++) {
                int kOff = k * p * p;
                double denom = Math.max(nk[k], 1e-10);
                for (int a = 0; a < p; a++) {
                    for (int b = 0; b <= a; b++) {
                        covariances[kOff + a * p + b] /= denom;
                    }
                }
            }
        }

        // mirror the upper triangle and add the regulariser to the diagonal
        for (int k = 0; k < components; k++) {
            int kOff = k * p * p;
            for (int a = 0; a < p; a++) {
                for (int b = 0; b < a; b++) {
                    covariances[kOff + b * p + a] = covariances[kOff + a * p + b];
                }
                covariances[kOff + a * p + a] += regCovar;
                variances[k * p + a] = covariances[kOff + a * p + a];
            }
            if (covarianceType == CovarianceType.DIAG) {
                for (int a = 0; a < p; a++) {
                    for (int b = 0; b < p; b++) {
                        if (a != b) {
                            covariances[kOff + a * p + b] = 0.0;
                        }
                    }
                }
            }
        }
    }

    /** Recomputes the Cholesky factor and log-determinant of every component. */
    private void refreshFactors() {
        int p = featureCount;
        double[] work = new double[p * p];
        for (int k = 0; k < components; k++) {
            if (covarianceType == CovarianceType.DIAG) {
                double sum = 0.0;
                for (int f = 0; f < p; f++) {
                    sum += Math.log(Math.max(1e-15, variances[k * p + f]));
                }
                logDet[k] = sum;
                continue;
            }
            int kOff = k * p * p;
            System.arraycopy(covariances, kOff, work, 0, p * p);
            double jitter = 0.0;
            boolean ok = false;
            for (int attempt = 0; attempt < 6 && !ok; attempt++) {
                if (attempt > 0) {
                    jitter = Math.max(regCovar * Math.pow(10, attempt), 1e-12 * Math.pow(10, attempt));
                    for (int a = 0; a < p; a++) {
                        work[a * p + a] = covariances[kOff + a * p + a] + jitter;
                    }
                }
                ok = choleskyInPlace(work, p);
            }
            if (!ok) {
                throw new IllegalStateException(
                        "covariance of component " + k + " is not positive definite");
            }
            double sum = 0.0;
            for (int a = 0; a < p; a++) {
                sum += Math.log(Math.max(1e-15, work[a * p + a]));
            }
            logDet[k] = 2.0 * sum;
            System.arraycopy(work, 0, cholesky, kOff, p * p);
        }
    }

    /** In-place Cholesky (lower).  Returns false if the matrix is not positive definite. */
    private static boolean choleskyInPlace(double[] a, int p) {
        for (int j = 0; j < p; j++) {
            double sum = a[j * p + j];
            for (int k = 0; k < j; k++) {
                sum -= a[j * p + k] * a[j * p + k];
            }
            if (!(sum > 0.0)) {
                return false;
            }
            double ljj = Math.sqrt(sum);
            a[j * p + j] = ljj;
            for (int i = j + 1; i < p; i++) {
                double s = a[i * p + j];
                for (int k = 0; k < j; k++) {
                    s -= a[i * p + k] * a[j * p + k];
                }
                a[i * p + j] = s / ljj;
            }
        }
        for (int i = 0; i < p; i++) {
            for (int j = i + 1; j < p; j++) {
                a[i * p + j] = 0.0;
            }
        }
        return true;
    }

    private double logGaussian(double[] x, int xOff, int component) {
        int p = featureCount;
        int base = component * p;
        if (covarianceType == CovarianceType.DIAG) {
            double maha = 0.0;
            for (int f = 0; f < p; f++) {
                double d = x[xOff + f] - means[base + f];
                maha += (d * d) / Math.max(1e-15, variances[base + f]);
            }
            return -0.5 * (p * Math.log(2.0 * Math.PI) + logDet[component] + maha);
        }
        int kOff = component * p * p;
        double[] d = new double[p];
        for (int f = 0; f < p; f++) {
            d[f] = x[xOff + f] - means[base + f];
        }
        // forward substitution: L z = d  =>  d' Sigma^-1 d = z'z
        double maha = 0.0;
        for (int i = 0; i < p; i++) {
            double s = d[i];
            for (int j = 0; j < i; j++) {
                s -= cholesky[kOff + i * p + j] * d[j];
            }
            d[i] = s / cholesky[kOff + i * p + i];
            maha += d[i] * d[i];
        }
        return -0.5 * (p * Math.log(2.0 * Math.PI) + logDet[component] + maha);
    }

    /** Per-sample log probability density {@code log p(x)}. */
    public double[] scoreSamples(Matrix features) {
        MlChecks.predictionData(features, featureCount, weights != null);
        int rows = features.rows();
        double[] data = features.data();
        double[] scores = new double[rows];
        double[] lps = new double[components];
        for (int r = 0; r < rows; r++) {
            double maxLog = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < components; k++) {
                lps[k] = Math.log(Math.max(1e-15, weights[k])) + logGaussian(data, r * featureCount, k);
                if (lps[k] > maxLog) {
                    maxLog = lps[k];
                }
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
                double lp = Math.log(Math.max(1e-15, weights[k])) + logGaussian(data, r * featureCount, k);
                probs[rBase + k] = lp;
                if (lp > maxLog) {
                    maxLog = lp;
                }
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

    /** Free parameters in the mixture, as used by {@link #aic} and {@link #bic}. */
    public int parameterCount() {
        int p = featureCount;
        int mixture = components - 1;
        return switch (covarianceType) {
            case DIAG -> components * 2 * p + mixture;
            case FULL -> components * (p + p * (p + 1) / 2) + mixture;
            case TIED -> components * p + p * (p + 1) / 2 + mixture;
        };
    }

    public double aic(Matrix features) {
        return 2.0 * parameterCount() - 2.0 * totalLogLikelihood(features);
    }

    public double bic(Matrix features) {
        return parameterCount() * Math.log(features.rows()) - 2.0 * totalLogLikelihood(features);
    }

    private double totalLogLikelihood(Matrix features) {
        double[] logProbs = scoreSamples(features);
        double logLik = 0.0;
        for (double v : logProbs) {
            logLik += v;
        }
        return logLik;
    }

    public double[] weights() {
        return weights == null ? null : weights.clone();
    }

    public CovarianceType covarianceType() {
        return covarianceType;
    }

    public Matrix means() {
        if (means == null) {
            throw new IllegalStateException("GaussianMixture is not fitted");
        }
        return new Matrix(components, featureCount, means);
    }

    /** Diagonal of every component covariance, shape {@code [components, featureCount]}. */
    public Matrix variances() {
        if (variances == null) {
            throw new IllegalStateException("GaussianMixture is not fitted");
        }
        return new Matrix(components, featureCount, variances);
    }

    /** Full covariance matrix of one component, shape {@code [featureCount, featureCount]}. */
    public Matrix covariance(int component) {
        if (covariances == null) {
            throw new IllegalStateException("GaussianMixture is not fitted");
        }
        if (component < 0 || component >= components) {
            throw new IllegalArgumentException("component index out of range: " + component);
        }
        int p = featureCount;
        return new Matrix(p, p, Arrays.copyOfRange(covariances, component * p * p, (component + 1) * p * p));
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
        writer.putByte(covarianceType.ordinal());
        writer.putDoubles(weights);
        writer.putDoubles(means);
        writer.putDoubles(variances);
        writer.putDoubles(covariances);
    }

    public static GaussianMixture read(ModelReader reader) {
        int comps = reader.getInt();
        int features = reader.getInt();
        boolean conv = reader.getByte() != 0;
        int iters = reader.getInt();
        double lb = reader.getDouble();
        int typeOrdinal = reader.getByte();
        if (comps <= 0 || features <= 0) {
            throw new ModelFormatException("Invalid GaussianMixture header");
        }
        if (typeOrdinal < 0 || typeOrdinal >= CovarianceType.values().length) {
            throw new ModelFormatException("Unknown covariance type code: " + typeOrdinal);
        }
        CovarianceType type = CovarianceType.values()[typeOrdinal];
        GaussianMixture gmm = new Builder().components(comps).covarianceType(type).build();
        gmm.featureCount = features;
        gmm.converged = conv;
        gmm.iterations = iters;
        gmm.lowerBound = lb;
        gmm.weights = reader.getDoubles(comps);
        gmm.means = reader.getDoubles(Math.multiplyExact(comps, features));
        gmm.variances = reader.getDoubles(Math.multiplyExact(comps, features));
        gmm.covariances = reader.getDoubles(Math.multiplyExact(Math.multiplyExact(comps, features), features));
        gmm.cholesky = new double[gmm.covariances.length];
        gmm.logDet = new double[comps];
        gmm.refreshFactors();
        return gmm;
    }

    public static final class Builder {
        private int components = 2;
        private int maxIterations = 100;
        private double tolerance = 1e-4;
        private double regCovar = 1e-6;
        private long seed = 42L;
        private CovarianceType covarianceType = CovarianceType.FULL;

        public Builder components(int components) {
            if (components <= 0) {
                throw new IllegalArgumentException("components must be positive");
            }
            this.components = components;
            return this;
        }

        public Builder covarianceType(CovarianceType covarianceType) {
            this.covarianceType = java.util.Objects.requireNonNull(covarianceType, "covarianceType");
            return this;
        }

        public Builder maxIterations(int maxIterations) {
            if (maxIterations <= 0) {
                throw new IllegalArgumentException("maxIterations must be positive");
            }
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
