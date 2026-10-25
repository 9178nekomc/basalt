package basalt.ml;

import basalt.core.VectorMath;

/** Strongly typed KNN distance functions backed by Vector API primitives. */
public enum DistanceMetric {
    EUCLIDEAN {
        @Override
        public double distance(double[] a, int aOffset, double[] b, int bOffset, int length) {
            return Math.sqrt(VectorMath.squaredEuclidean(a, aOffset, b, bOffset, length));
        }
    },
    MANHATTAN {
        @Override
        public double distance(double[] a, int aOffset, double[] b, int bOffset, int length) {
            return VectorMath.manhattan(a, aOffset, b, bOffset, length);
        }
    };

    public abstract double distance(double[] a, int aOffset, double[] b, int bOffset, int length);
}
