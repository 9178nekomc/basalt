package basalt.ml;

import basalt.core.Matrix;
import basalt.core.ModelFormatException;
import basalt.core.ModelKind;
import basalt.core.ModelReader;
import basalt.core.ModelWriter;
import basalt.core.Persistable;
import basalt.core.VectorMath;
import java.util.Objects;
import java.util.Random;

/**
 * Kernel Density Estimation (KDE) supporting {@code GAUSSIAN} and {@code TOPHAT} kernels,
 * matching {@code sklearn.neighbors.KernelDensity}.
 */
public final class KernelDensity implements Estimator, Persistable {
    public enum KernelType {
        GAUSSIAN,
        TOPHAT
    }

    private final double bandwidth;
    private final KernelType kernel;
    private Matrix trainX;
    private int featureCount;

    private KernelDensity(Builder builder) {
        this.bandwidth = builder.bandwidth;
        this.kernel = builder.kernel;
    }

    public static Builder builder() {
        return new Builder();
    }

    public KernelDensity fit(Matrix features) {
        return fit(features, null);
    }

    @Override
    public KernelDensity fit(Matrix features, double[] ignoredTargets) {
        MlChecks.features(features);
        featureCount = features.columns();
        trainX = features.copy();
        return this;
    }

    /** Evaluates the log-density model on the provided points ({@code log p(x)}). */
    public double[] scoreSamples(Matrix features) {
        MlChecks.predictionData(features, featureCount, trainX != null);
        int rows = features.rows();
        int n = trainX.rows();
        double[] xData = features.data();
        double[] tData = trainX.data();
        double[] logDens = new double[rows];
        double invBw2 = 1.0 / (bandwidth * bandwidth);

        if (kernel == KernelType.GAUSSIAN) {
            double normConst = -Math.log(n) - featureCount * Math.log(bandwidth)
                    - 0.5 * featureCount * Math.log(2.0 * Math.PI);
            double[] logTerms = new double[n];
            for (int r = 0; r < rows; r++) {
                int rOff = r * featureCount;
                double maxTerm = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < n; i++) {
                    double sq = VectorMath.squaredEuclidean(xData, rOff, tData, i * featureCount, featureCount);
                    double term = -0.5 * sq * invBw2;
                    logTerms[i] = term;
                    if (term > maxTerm) maxTerm = term;
                }
                double sumExp = 0.0;
                for (int i = 0; i < n; i++) {
                    sumExp += Math.exp(logTerms[i] - maxTerm);
                }
                logDens[r] = normConst + maxTerm + Math.log(sumExp);
            }
        } else {
            double logVol = 0.5 * featureCount * Math.log(Math.PI)
                    - logGamma(0.5 * featureCount + 1.0)
                    + featureCount * Math.log(bandwidth);
            for (int r = 0; r < rows; r++) {
                int rOff = r * featureCount;
                int hits = 0;
                for (int i = 0; i < n; i++) {
                    double sq = VectorMath.squaredEuclidean(xData, rOff, tData, i * featureCount, featureCount);
                    if (sq <= bandwidth * bandwidth) {
                        hits++;
                    }
                }
                logDens[r] = hits == 0 ? Double.NEGATIVE_INFINITY : Math.log((double) hits / n) - logVol;
            }
        }
        return logDens;
    }

    /** Generates random samples from the fitted kernel density model. */
    public Matrix sample(int nSamples, long seed) {
        if (trainX == null) throw new IllegalStateException("KernelDensity must be fitted first");
        if (nSamples <= 0) throw new IllegalArgumentException("nSamples must be positive");
        Random rng = new Random(seed);
        int n = trainX.rows();
        double[] tData = trainX.data();
        double[] out = new double[nSamples * featureCount];
        for (int s = 0; s < nSamples; s++) {
            int idx = rng.nextInt(n);
            for (int f = 0; f < featureCount; f++) {
                double noise = kernel == KernelType.GAUSSIAN
                        ? rng.nextGaussian() * bandwidth
                        : (rng.nextDouble() * 2.0 - 1.0) * bandwidth;
                out[s * featureCount + f] = tData[idx * featureCount + f] + noise;
            }
        }
        return new Matrix(nSamples, featureCount, out);
    }

    private static double logGamma(double x) {
        double[] coef = {
                76.18009172947146, -86.50532032941677,
                24.01409824083091, -1.231739572450155,
                0.001208650973866179, -0.000005395239384953
        };
        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.log(tmp);
        double ser = 1.000000000190015;
        for (double c : coef) {
            y += 1.0;
            ser += c / y;
        }
        return -tmp + Math.log(2.5066282746310005 * ser / x);
    }

    @Override
    public double[] predict(Matrix features) {
        return scoreSamples(features);
    }

    public double bandwidth() {
        return bandwidth;
    }

    @Override
    public ModelKind kind() {
        return ModelKind.KERNEL_DENSITY;
    }

    @Override
    public void write(ModelWriter writer) {
        if (trainX == null) {
            throw new IllegalStateException("KernelDensity must be fitted before serialization");
        }
        writer.putDouble(bandwidth);
        writer.putInt(kernel.ordinal());
        writer.putInt(featureCount);
        writer.putInt(trainX.rows());
        writer.putDoubles(trainX.data());
    }

    public static KernelDensity read(ModelReader reader) {
        double bw = reader.getDouble();
        int kOrd = reader.getInt();
        int features = reader.getInt();
        int rows = reader.getInt();
        if (kOrd < 0 || kOrd >= KernelType.values().length || features <= 0 || rows <= 0) {
            throw new ModelFormatException("Invalid KernelDensity header");
        }
        double[] data = reader.getDoubles(Math.multiplyExact(rows, features));
        KernelDensity kde = new Builder().bandwidth(bw).kernel(KernelType.values()[kOrd]).build();
        kde.featureCount = features;
        kde.trainX = new Matrix(rows, features, data);
        return kde;
    }

    public static final class Builder {
        private double bandwidth = 1.0;
        private KernelType kernel = KernelType.GAUSSIAN;

        public Builder bandwidth(double bandwidth) {
            MlChecks.positiveFinite(bandwidth, "bandwidth");
            this.bandwidth = bandwidth;
            return this;
        }

        public Builder kernel(KernelType kernel) {
            this.kernel = Objects.requireNonNull(kernel, "kernel");
            return this;
        }

        public KernelDensity build() {
            return new KernelDensity(this);
        }
    }
}
