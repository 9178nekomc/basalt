package basalt.core;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.util.Objects;

/** Bounds-checked big-endian reader paired with {@link ModelWriter}. */
public final class ModelReader {
    private final ByteBuffer buffer;

    public ModelReader(byte[] payload) {
        Objects.requireNonNull(payload, "payload");
        buffer = ByteBuffer.wrap(payload);
    }

    /** Zero-copy reader over the remaining region of {@code payload}. */
    public ModelReader(ByteBuffer payload) {
        Objects.requireNonNull(payload, "payload");
        buffer = payload.slice();
    }

    public int getByte() {
        require(Byte.BYTES);
        return Byte.toUnsignedInt(buffer.get());
    }

    public int getInt() {
        require(Integer.BYTES);
        return buffer.getInt();
    }

    public long getLong() {
        require(Long.BYTES);
        return buffer.getLong();
    }

    public double getDouble() {
        require(Double.BYTES);
        return buffer.getDouble();
    }

    public double[] getDoubles(int length) {
        if (length < 0) {
            throw new ModelFormatException("Negative array length " + length);
        }
        require(Math.multiplyExact(length, Double.BYTES));
        double[] values = new double[length];
        DoubleBuffer view = buffer.asDoubleBuffer();
        view.get(values, 0, length);
        buffer.position(buffer.position() + Math.multiplyExact(length, Double.BYTES));
        return values;
    }

    public int[] getInts(int length) {
        if (length < 0) {
            throw new ModelFormatException("Negative array length " + length);
        }
        require(Math.multiplyExact(length, Integer.BYTES));
        int[] values = new int[length];
        for (int i = 0; i < length; i++) {
            values[i] = buffer.getInt();
        }
        return values;
    }

    public int remaining() {
        return buffer.remaining();
    }

    public void requireFullyConsumed() {
        if (buffer.hasRemaining()) {
            throw new ModelFormatException(
                    "Trailing payload bytes: " + buffer.remaining());
        }
    }

    private void require(int bytes) {
        if (buffer.remaining() < bytes) {
            throw new ModelFormatException("Truncated .bsl payload");
        }
    }
}
