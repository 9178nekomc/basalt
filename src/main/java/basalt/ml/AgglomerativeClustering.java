package basalt.ml;

import basalt.core.Matrix;
import basalt.core.VectorMath;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Hierarchical Agglomerative Clustering supporting {@code WARD}, {@code COMPLETE}, {@code AVERAGE},
 * and {@code SINGLE} linkage criteria.
 */
public final class AgglomerativeClustering implements Estimator {
    public enum Linkage {
        WARD,
        COMPLETE,
        AVERAGE,
        SINGLE
    }

    private final int clusters;
    private final Linkage linkage;
    private Matrix fitData;
    private int[] labels;
    private double[] centroids;
    private int featureCount;

    private AgglomerativeClustering(Builder builder) {
        this.clusters = builder.clusters;
        this.linkage = builder.linkage;
    }

    public static Builder builder() {
        return new Builder();
    }

    public AgglomerativeClustering fit(Matrix features) {
        return fit(features, null);
    }

    @Override
    public AgglomerativeClustering fit(Matrix features, double[] ignoredTargets) {
        MlChecks.features(features);
        int n = features.rows();
        featureCount = features.columns();
        if (clusters > n) {
            throw new IllegalArgumentException("clusters cannot exceed sample count");
        }
        fitData = features.copy();
        double[] data = features.data();

        // Maintain active clusters
        List<List<Integer>> groups = new ArrayList<>(n);
        double[][] groupMeans = new double[n][featureCount];
        boolean[] active = new boolean[n];
        for (int i = 0; i < n; i++) {
            List<Integer> list = new ArrayList<>();
            list.add(i);
            groups.add(list);
            System.arraycopy(data, i * featureCount, groupMeans[i], 0, featureCount);
            active[i] = true;
        }

        double[] dist = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double d = Math.sqrt(VectorMath.squaredEuclidean(
                        data, i * featureCount, data, j * featureCount, featureCount));
                dist[i * n + j] = dist[j * n + i] = d;
            }
        }

        int activeCount = n;
        while (activeCount > clusters) {
            int bestA = -1;
            int bestB = -1;
            double bestCost = Double.POSITIVE_INFINITY;
            for (int i = 0; i < n; i++) {
                if (!active[i]) continue;
                for (int j = i + 1; j < n; j++) {
                    if (!active[j]) continue;
                    double cost = switch (linkage) {
                        case SINGLE, COMPLETE, AVERAGE -> dist[i * n + j];
                        case WARD -> {
                            int ni = groups.get(i).size();
                            int nj = groups.get(j).size();
                            double sq = VectorMath.squaredEuclidean(groupMeans[i], 0, groupMeans[j], 0, featureCount);
                            yield ((double) (ni * nj) / (ni + nj)) * sq;
                        }
                    };
                    if (cost < bestCost) {
                        bestCost = cost;
                        bestA = i;
                        bestB = j;
                    }
                }
            }
            if (bestA < 0) break;

            List<Integer> listA = groups.get(bestA);
            List<Integer> listB = groups.get(bestB);
            int sizeA = listA.size();
            int sizeB = listB.size();
            int newSize = sizeA + sizeB;
            for (int f = 0; f < featureCount; f++) {
                groupMeans[bestA][f] = (groupMeans[bestA][f] * sizeA + groupMeans[bestB][f] * sizeB) / newSize;
            }
            for (int k = 0; k < n; k++) {
                if (!active[k] || k == bestA || k == bestB) continue;
                double dAK = dist[bestA * n + k];
                double dBK = dist[bestB * n + k];
                double merged = switch (linkage) {
                    case SINGLE -> Math.min(dAK, dBK);
                    case COMPLETE -> Math.max(dAK, dBK);
                    case AVERAGE, WARD -> (sizeA * dAK + sizeB * dBK) / newSize;
                };
                dist[bestA * n + k] = dist[k * n + bestA] = merged;
            }
            listA.addAll(listB);
            active[bestB] = false;
            activeCount--;
        }

        labels = new int[n];
        centroids = new double[clusters * featureCount];
        int clusterId = 0;
        for (int i = 0; i < n; i++) {
            if (!active[i]) continue;
            for (int member : groups.get(i)) {
                labels[member] = clusterId;
            }
            System.arraycopy(groupMeans[i], 0, centroids, clusterId * featureCount, featureCount);
            clusterId++;
        }
        return this;
    }

    @Override
    public double[] predict(Matrix features) {
        MlChecks.predictionData(features, featureCount, labels != null);
        if (features.rows() == fitData.rows()
                && Arrays.equals(features.data(), fitData.data())) {
            double[] out = new double[labels.length];
            for (int i = 0; i < labels.length; i++) out[i] = labels[i];
            return out;
        }
        int rows = features.rows();
        double[] out = new double[rows];
        double[] data = features.data();
        for (int r = 0; r < rows; r++) {
            int best = 0;
            double bestDist = Double.POSITIVE_INFINITY;
            for (int c = 0; c < clusters; c++) {
                double d = VectorMath.squaredEuclidean(
                        data, r * featureCount, centroids, c * featureCount, featureCount);
                if (d < bestDist) {
                    bestDist = d;
                    best = c;
                }
            }
            out[r] = best;
        }
        return out;
    }

    public int[] labels() {
        if (labels == null) throw new IllegalStateException("AgglomerativeClustering is not fitted");
        return labels.clone();
    }

    public Linkage linkage() {
        return linkage;
    }

    public static final class Builder {
        private int clusters = 2;
        private Linkage linkage = Linkage.WARD;

        public Builder clusters(int clusters) {
            if (clusters <= 0) throw new IllegalArgumentException("clusters must be positive");
            this.clusters = clusters;
            return this;
        }

        public Builder linkage(Linkage linkage) {
            this.linkage = Objects.requireNonNull(linkage, "linkage");
            return this;
        }

        public AgglomerativeClustering build() {
            return new AgglomerativeClustering(this);
        }
    }
}
