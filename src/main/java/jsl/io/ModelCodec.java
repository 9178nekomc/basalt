package jsl.io;

import jsl.core.ModelFormatException;
import jsl.core.ModelKind;
import jsl.core.ModelReader;
import jsl.core.ModelWriter;
import jsl.core.Persistable;
import jsl.ml.GaussianNaiveBayes;
import jsl.ml.HistogramGradientBoosting;
import jsl.ml.KMeans;
import jsl.ml.PCA;
import jsl.ml.RandomForest;
import jsl.nn.CNN;
import jsl.nn.GRU;
import jsl.nn.LSTM;
import jsl.nn.ModelSerializer;
import jsl.nn.SelfAttention;
import jsl.nn.SequenceNetwork;
import jsl.nn.VanillaRNN;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;
import java.util.zip.CRC32C;

/**
 * Versioned .jsl container: 24-byte header + CRC32C-checked payload.
 *
 * <pre>
 * int  magic     0x4A534C4D ("JSLM")
 * int  version   2
 * int  kind      stable ModelKind code
 * int  length    payload bytes
 * long crc32c    checksum of the payload
 * byte[length]   model-specific payload
 * </pre>
 *
 * <p>Saving writes one {@link ByteBuffer} through a single channel call. Loading memory-maps the
 * file, validates the checksum without copying, and hands a zero-copy slice to the model reader.</p>
 */
public final class ModelCodec {
    public static final int MAGIC = 0x4A534C4D;
    public static final int VERSION = 2;
    private static final int HEADER_BYTES = 24;

    private ModelCodec() {
    }

    public static void save(Path target, Persistable model) {
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(model, "model");
        ModelWriter payload = new ModelWriter(4096);
        model.write(payload);
        byte[] bytes = payload.toByteArray();
        if (bytes.length > Integer.MAX_VALUE - HEADER_BYTES) {
            throw new ModelFormatException("Serialized model exceeds the .jsl container limit");
        }
        CRC32C checksum = new CRC32C();
        checksum.update(bytes, 0, bytes.length);
        ByteBuffer file = ByteBuffer.allocate(HEADER_BYTES + bytes.length);
        file.putInt(MAGIC);
        file.putInt(VERSION);
        file.putInt(model.kind().code());
        file.putInt(bytes.length);
        file.putLong(checksum.getValue());
        file.put(bytes);
        file.flip();
        try (FileChannel channel = FileChannel.open(target,
                StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            while (file.hasRemaining()) {
                channel.write(file);
            }
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
    }

    public static Persistable load(Path source) {
        Objects.requireNonNull(source, "source");
        try (FileChannel channel = FileChannel.open(source, StandardOpenOption.READ)) {
            long size = channel.size();
            if (size < HEADER_BYTES) {
                throw new ModelFormatException("File is shorter than the .jsl header");
            }
            MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, size);
            int magic = mapped.getInt();
            int version = mapped.getInt();
            if (magic != MAGIC) {
                throw new ModelFormatException("Invalid .jsl magic");
            }
            if (version != VERSION) {
                throw new ModelFormatException("Unsupported .jsl version " + version);
            }
            ModelKind kind = ModelKind.fromCode(mapped.getInt());
            int length = mapped.getInt();
            long expectedChecksum = mapped.getLong();
            if (length < 0 || length != size - HEADER_BYTES) {
                throw new ModelFormatException("Declared payload length does not match file size");
            }
            CRC32C checksum = new CRC32C();
            checksum.update(mapped.duplicate().position(HEADER_BYTES));
            if (checksum.getValue() != expectedChecksum) {
                throw new ModelFormatException("CRC32C mismatch: file is corrupt");
            }
            ModelReader reader = new ModelReader(mapped.position(HEADER_BYTES));
            Persistable model = read(kind, reader);
            reader.requireFullyConsumed();
            return model;
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
    }

    public static <T extends Persistable> T load(Path source, Class<T> type) {
        Persistable model = load(source);
        if (!type.isInstance(model)) {
            throw new ModelFormatException(
                    "Expected " + type.getSimpleName() + " but file contains " + model.kind());
        }
        return type.cast(model);
    }

    private static Persistable read(ModelKind kind, ModelReader reader) {
        return switch (kind) {
            case CNN -> ModelSerializer.readCNN(reader);
            case HISTOGRAM_BOOSTING -> HistogramGradientBoosting.read(reader);
            case RANDOM_FOREST -> RandomForest.readState(reader);
            case KMEANS -> KMeans.read(reader);
            case PCA -> PCA.read(reader);
            case GAUSSIAN_NAIVE_BAYES -> GaussianNaiveBayes.read(reader);
            case LSTM -> LSTM.read(reader);
            case GRU -> GRU.read(reader);
            case VANILLA_RNN -> VanillaRNN.read(reader);
            case SELF_ATTENTION -> SelfAttention.read(reader);
            case SEQUENCE_NETWORK -> SequenceNetwork.read(reader);
        };
    }

    /** Utility used by tests and tooling; reads only the header without mapping the payload. */
    public static ModelKind peekKind(Path source) {
        try (FileChannel channel = FileChannel.open(source, StandardOpenOption.READ)) {
            ByteBuffer header = ByteBuffer.allocate(12);
            while (header.hasRemaining()) {
                if (channel.read(header) < 0) {
                    throw new ModelFormatException("Truncated .jsl header");
                }
            }
            header.flip();
            if (header.getInt() != MAGIC) {
                throw new ModelFormatException("Invalid .jsl magic");
            }
            int version = header.getInt();
            if (version != VERSION) {
                throw new ModelFormatException("Unsupported .jsl version " + version);
            }
            return ModelKind.fromCode(header.getInt());
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
    }

    /** Returns the serialized size without decoding the payload. */
    public static long fileSize(Path source) {
        try {
            return Files.size(source);
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
    }
}
