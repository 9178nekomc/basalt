package jsl.io;

import jsl.core.ModelFormatException;
import jsl.core.ModelKind;
import jsl.core.ModelReader;
import jsl.core.ModelWriter;
import jsl.core.Persistable;
import jsl.ml.DecisionTree;
import jsl.ml.GaussianMixture;
import jsl.ml.GaussianNaiveBayes;
import jsl.ml.HistogramGradientBoosting;
import jsl.ml.IsolationForest;
import jsl.ml.KBinsDiscretizer;
import jsl.ml.KMeans;
import jsl.ml.KernelDensity;
import jsl.ml.KernelPCA;
import jsl.ml.KernelRidge;
import jsl.ml.LabelEncoder;
import jsl.ml.Lasso;
import jsl.ml.LinearRegression;
import jsl.ml.LinearSVM;
import jsl.ml.LogisticRegression;
import jsl.ml.MinMaxScaler;
import jsl.ml.OneHotEncoder;
import jsl.ml.PCA;
import jsl.ml.Pipeline;
import jsl.ml.PolynomialFeatures;
import jsl.ml.RandomForest;
import jsl.ml.Ridge;
import jsl.ml.SVC;
import jsl.ml.SelectKBest;
import jsl.ml.SimpleImputer;
import jsl.ml.StandardScaler;
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
            Persistable model = readPayload(kind, reader);
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

    public static Persistable readPayload(ModelKind kind, ModelReader reader) {
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
            case STANDARD_SCALER -> StandardScaler.read(reader);
            case MIN_MAX_SCALER -> MinMaxScaler.read(reader);
            case ONE_HOT_ENCODER -> OneHotEncoder.read(reader);
            case LABEL_ENCODER -> LabelEncoder.read(reader);
            case SIMPLE_IMPUTER -> SimpleImputer.read(reader);
            case PIPELINE -> Pipeline.read(reader);
            case SVC -> SVC.read(reader);
            case KERNEL_RIDGE -> KernelRidge.read(reader);
            case KERNEL_PCA -> KernelPCA.read(reader);
            case GAUSSIAN_MIXTURE -> GaussianMixture.read(reader);
            case ISOLATION_FOREST -> IsolationForest.read(reader);
            case KERNEL_DENSITY -> KernelDensity.read(reader);
            case SELECT_K_BEST -> SelectKBest.read(reader);
            case POLYNOMIAL_FEATURES -> PolynomialFeatures.read(reader);
            case KBINS_DISCRETIZER -> KBinsDiscretizer.read(reader);
            case LINEAR_REGRESSION -> LinearRegression.read(reader);
            case LOGISTIC_REGRESSION -> LogisticRegression.read(reader);
            case RIDGE -> Ridge.read(reader);
            case LASSO -> Lasso.read(reader);
            case LINEAR_SVM -> LinearSVM.read(reader);
            case DECISION_TREE -> DecisionTree.read(reader);
        };
    }

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

    public static long fileSize(Path source) {
        try {
            return Files.size(source);
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
    }
}
