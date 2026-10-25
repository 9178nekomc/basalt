package basalt.core;

import basalt.BasaltRuntimeException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Optional Rust kernels loaded through JNI. The library is discovered from
 * {@code -Djsl.native.path=...} (default {@code build/native/basalt_native.dll|.so|.dylib}) or from
 * {@code java.library.path}. If loading fails, every caller falls back to the pure-Java path.
 */
public final class NativeKernels {
    private static final boolean AVAILABLE;
    private static final String LOAD_ERROR;
    private static final String LIBRARY_PATH;

    static {
        String os = System.getProperty("os.name", "").toLowerCase();
        String arch = System.getProperty("os.arch", "").toLowerCase();
        String fileName = os.contains("win") ? "basalt_native.dll"
                : os.contains("mac") ? "libbasalt_native.dylib" : "libbasalt_native.so";
        String platform = platformTag(os, arch);
        Path path = Path.of(System.getProperty("basalt.native.path", "build/native/" + fileName));
        boolean loaded = false;
        String error = "";
        if (Files.isRegularFile(path)) {
            try {
                System.load(path.toAbsolutePath().toString());
                loaded = true;
            } catch (Throwable failure) {
                error = failure.getClass().getSimpleName() + ": " + failure.getMessage();
            }
        }
        if (!loaded) {
            Path extracted = extractFromClasspath(platform, fileName);
            if (extracted != null) {
                try {
                    System.load(extracted.toAbsolutePath().toString());
                    path = extracted;
                    loaded = true;
                } catch (Throwable failure) {
                    error = failure.getClass().getSimpleName() + ": " + failure.getMessage();
                }
            }
        }
        if (!loaded) {
            try {
                System.loadLibrary("basalt_native");
                path = Path.of(System.getProperty("java.library.path", ".")).resolve(fileName);
                loaded = true;
            } catch (Throwable failure) {
                if (error.isEmpty()) {
                    error = failure.getClass().getSimpleName() + ": " + failure.getMessage();
                }
            }
        }
        AVAILABLE = loaded;
        LOAD_ERROR = error;
        LIBRARY_PATH = path.toAbsolutePath().toString();
    }

    private static String platformTag(String os, String arch) {
        String cpu = arch.contains("aarch64") || arch.contains("arm64") ? "aarch64" : "x86_64";
        if (os.contains("win")) {
            return "windows-" + cpu;
        }
        if (os.contains("mac")) {
            return "macos-" + cpu;
        }
        return "linux-" + cpu;
    }

    private static Path extractFromClasspath(String platform, String fileName) {
        String resource = "/native/" + platform + "/" + fileName;
        try (InputStream stream = NativeKernels.class.getResourceAsStream(resource)) {
            if (stream == null) {
                return null;
            }
            Path directory = Files.createTempDirectory("basalt-native-");
            Path target = directory.resolve(fileName);
            Files.copy(stream, target, StandardCopyOption.REPLACE_EXISTING);
            target.toFile().deleteOnExit();
            directory.toFile().deleteOnExit();
            return target;
        } catch (Exception failure) {
            return null;
        }
    }

    private NativeKernels() {
    }

    public static boolean available() {
        return AVAILABLE;
    }

    public static String loadError() {
        return LOAD_ERROR;
    }

    public static String libraryPath() {
        return LIBRARY_PATH;
    }

    public static native int version();

    private static native int gemm(double[] a, double[] b, double[] c,
                                   int m, int k, int n, int threads);

    private static native int kmeansAssign(double[] data, double[] centers, int rows,
                                           int clusters, int features, int threads,
                                           int[] assignments, double[] distances);

    private static native int forestPredict(double[] data, int rows, int columns,
                                            int[] feature, double[] threshold, int[] left,
                                            int[] right, double[] value, int[] offsets,
                                            double[] output, double scale);

    private static native int forestVote(double[] data, int rows, int columns,
                                         int[] feature, double[] threshold, int[] left,
                                         int[] right, double[] value, int[] offsets,
                                         double[] labels, double[] votes, double scale);

    private static native int histogramAccumulate(double[] gradients, double[] hessians,
                                                  byte[] bins, int[] samples,
                                                  int[] activeFeatures, int featureCount,
                                                  int maxBins, int threads,
                                                  double[] outGradient, double[] outHessian,
                                                  int[] outCount);

    private static native int ridgeFit(double[] data, int rows, int columns, double[] targets,
                                       double alpha, boolean fitIntercept, int threads,
                                       double[] weights, double[] intercept);

    private static native int gaussianNbFit(double[] data, int rows, int columns,
                                            int[] classIndices, int classCount,
                                            double[] means, double[] variances,
                                            double[] logPriors, int[] counts,
                                            double varianceSmoothing);


    private static native int kmeansFit(double[] data, int rows, int clusters, int features,
                                        int restarts, int maxIterations, double tolerance,
                                        long seed, int threads, double[] centers,
                                        double[] inertia, int[] iterations);

    private static native long histogramSessionCreate(byte[] bins, double[] gradients,
                                                      double[] hessians, int featureCount,
                                                      int maxBins);

    private static native int histogramSessionUpdate(long session, double[] gradients,
                                                     double[] hessians);

    private static native int histogramSessionAccumulate(long session, int[] samples,
                                                         int[] activeFeatures, int threads,
                                                         double[] outGradient,
                                                         double[] outHessian, int[] outCount);

    private static native int histogramSessionDestroy(long session);

    private static native int binFeatures(double[] data, int rows, int columns, int maxBins,
                                          byte[] outBins, double[] outThresholds,
                                          int[] outCounts);


    /** Returns 0 on success; throws if the native kernel reports an error. */
    public static void gemmChecked(double[] a, double[] b, double[] c,
                                   int m, int k, int n, int threads) {
        int status = gemm(a, b, c, m, k, n, threads);
        if (status != 0) {
            throw new BasaltRuntimeException("Native GEMM failed with status " + status);
        }
    }

    /** Returns true on success, false when the native kernel rejects the shapes. */
    public static boolean kmeansAssignChecked(double[] data, double[] centers, int rows,
                                              int clusters, int features, int threads,
                                              int[] assignments, double[] distances) {
        return kmeansAssign(data, centers, rows, clusters, features, threads,
                assignments, distances) == 0;
    }

    public static boolean forestPredictChecked(double[] data, int rows, int columns,
                                               int[] feature, double[] threshold, int[] left,
                                               int[] right, double[] value, int[] offsets,
                                               double[] output, double scale) {
        return forestPredict(data, rows, columns, feature, threshold, left, right, value,
                offsets, output, scale) == 0;
    }

    public static boolean forestVoteChecked(double[] data, int rows, int columns,
                                            int[] feature, double[] threshold, int[] left,
                                            int[] right, double[] value, int[] offsets,
                                            double[] labels, double[] votes, double scale) {
        return forestVote(data, rows, columns, feature, threshold, left, right, value,
                offsets, labels, votes, scale) == 0;
    }

    /**
     * Accumulates gradient/hessian/count histograms over {@code samples} for the active features.
     * Output arrays are indexed by {@code feature * maxBins + bin} and must be cleared by the
     * caller for the active ranges.
     */
    public static boolean histogramAccumulateChecked(double[] gradients, double[] hessians,
                                                     byte[] bins, int[] samples,
                                                     int[] activeFeatures, int featureCount,
                                                     int maxBins, int threads,
                                                     double[] outGradient, double[] outHessian,
                                                     int[] outCount) {
        return histogramAccumulate(gradients, hessians, bins, samples, activeFeatures,
                featureCount, maxBins, threads, outGradient, outHessian, outCount) == 0;
    }

    /** Solves ridge regression in native code; returns false when the kernel rejects the shapes. */
    public static boolean ridgeFitChecked(double[] data, int rows, int columns, double[] targets,
                                          double alpha, boolean fitIntercept, int threads,
                                          double[] weights, double[] intercept) {
        return ridgeFit(data, rows, columns, targets, alpha, fitIntercept, threads,
                weights, intercept) == 0;
    }

    /** Computes GaussianNB sufficient statistics in native code. */
    public static boolean gaussianNbFitChecked(double[] data, int rows, int columns,
                                               int[] classIndices, int classCount,
                                               double[] means, double[] variances,
                                               double[] logPriors, int[] counts,
                                               double varianceSmoothing) {
        return gaussianNbFit(data, rows, columns, classIndices, classCount, means, variances,
                logPriors, counts, varianceSmoothing) == 0;
    }

    /** Runs one full Lloyd iteration (assign, accumulate, update) in native code. */

    /** Runs the complete K-Means++ / Lloyd loop (all restarts) in native code. */
    public static boolean kmeansFitChecked(double[] data, int rows, int clusters, int features,
                                           int restarts, int maxIterations, double tolerance,
                                           long seed, int threads, double[] centers,
                                           double[] inertia, int[] iterations) {
        return kmeansFit(data, rows, clusters, features, restarts, maxIterations, tolerance,
                seed, threads, centers, inertia, iterations) == 0;
    }

    /** Creates a native histogram session; returns {@code 0} when the kernel rejects the input. */
    public static long createHistogramSession(byte[] bins, double[] gradients, double[] hessians,
                                              int featureCount, int maxBins) {
        return histogramSessionCreate(bins, gradients, hessians, featureCount, maxBins);
    }

    public static boolean updateHistogramSession(long session, double[] gradients,
                                                 double[] hessians) {
        return histogramSessionUpdate(session, gradients, hessians) == 0;
    }

    public static boolean histogramSessionAccumulateChecked(long session, int[] samples,
                                                            int[] activeFeatures, int threads,
                                                            double[] outGradient,
                                                            double[] outHessian, int[] outCount) {
        return histogramSessionAccumulate(session, samples, activeFeatures, threads,
                outGradient, outHessian, outCount) == 0;
    }

    public static void destroyHistogramSession(long session) {
        if (session != 0L) {
            histogramSessionDestroy(session);
        }
    }

    /** Quantile-bins every feature column in native code. */
    public static boolean binFeaturesChecked(double[] data, int rows, int columns, int maxBins,
                                             byte[] outBins, double[] outThresholds,
                                             int[] outCounts) {
        return binFeatures(data, rows, columns, maxBins, outBins, outThresholds, outCounts) == 0;
    }


    /**
     * AutoCloseable + Cleaner-backed handle for a native histogram session, preventing native
     * memory leaks even if an exception aborts tree construction.
     */
    public static final class HistogramSessionHandle implements AutoCloseable {
        private static final java.lang.ref.Cleaner CLEANER = java.lang.ref.Cleaner.create();
        private final java.util.concurrent.atomic.AtomicLong handle;
        private final java.lang.ref.Cleaner.Cleanable cleanable;

        public HistogramSessionHandle(long sessionPointer) {
            this.handle = new java.util.concurrent.atomic.AtomicLong(sessionPointer);
            final java.util.concurrent.atomic.AtomicLong ref = this.handle;
            this.cleanable = CLEANER.register(this, () -> {
                long ptr = ref.getAndSet(0L);
                if (ptr != 0L) {
                    destroyHistogramSession(ptr);
                }
            });
        }

        public long handle() {
            return handle.get();
        }

        @Override
        public void close() {
            cleanable.clean();
        }
    }
}
