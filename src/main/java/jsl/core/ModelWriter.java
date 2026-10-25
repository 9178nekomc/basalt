package jsl.core;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.util.Arrays;

/**
 * Growable big-endian binary writer. Primitive arrays are copied through a {@link DoubleBuffer}
 * view in bulk, which keeps serialization bandwidth close to {@code System.arraycopy}.
 */
public final class ModelWriter {
    private static final int DEFAULT_CAPACITY = 256;

    private ByteBuffer buffer;

    public ModelWriter() {
        this(DEFAULT_CAPACITY);
    }

    public ModelWriter(int initialCapacity) {
        if (initialCapacity <= 0) {
            throw new IllegalArgumentException("initialCapacity must be positive");
        }
        buffer = ByteBuffer.allocate(initialCapacity);
    }

    public ModelWriter putByte(int value) {
        ensure(Byte.BYTES);
        buffer.put((byte) value);
        return this;
    }

    public ModelWriter putInt(int value) {
        ensure(Integer.BYTES);
        buffer.putInt(value);
        return this;
    }

    public ModelWriter putLong(long value) {
        ensure(Long.BYTES);
        buffer.putLong(value);
        return this;
    }

    public ModelWriter putDouble(double value) {
        ensure(Double.BYTES);
        buffer.putDouble(value);
        return this;
    }

    public ModelWriter putDoubles(double[] values) {
        return putDoubles(values, 0, values.length);
    }

    public ModelWriter putDoubles(double[] values, int offset, int length) {
        if (offset < 0 || length < 0 || offset + length > values.length) {
            throw new IndexOutOfBoundsException("Invalid double slice");
        }
        ensure(Math.multiplyExact(length, Double.BYTES));
        DoubleBuffer view = buffer.asDoubleBuffer();
        view.put(values, offset, length);
        buffer.position(buffer.position() + Math.multiplyExact(length, Double.BYTES));
        return this;
    }

    public ModelWriter putInts(int[] values) {
        ensure(Math.multiplyExact(values.length, Integer.BYTES));
        for (int value : values) {
            buffer.putInt(value);
        }
        return this;
    }

    public ModelWriter putBytes(byte[] values) {
        ensure(values.length);
        buffer.put(values);
        return this;
    }

    public int size() {
        return buffer.position();
    }

    public byte[] toByteArray() {
        return Arrays.copyOf(buffer.array(), buffer.position());
    }

    private void ensure(int additional) {
        if (additional < 0) {
            throw new IllegalArgumentException("Negative write length");
        }
        if (buffer.remaining() >= additional) {
            return;
        }
        long required = (long) buffer.position() + additional;
        if (required > Integer.MAX_VALUE) {
            throw new ModelFormatException("Serialized model exceeds the 2 GiB array limit");
        }
        long doubled = (long) buffer.capacity() * 2;
        int capacity = (int) Math.min(Integer.MAX_VALUE, Math.max(doubled, required));
        ByteBuffer grown = ByteBuffer.allocate(capacity);
        buffer.flip();
        grown.put(buffer);
        buffer = grown;
    }
}
