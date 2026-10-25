package basalt.io;

import basalt.core.ModelFormatException;
import basalt.core.ModelKind;
import basalt.core.ModelReader;
import basalt.core.ModelWriter;
import basalt.core.Persistable;
import basalt.ml.DecisionTree;
import basalt.ml.GaussianMixture;
import basalt.ml.GaussianNaiveBayes;
import basalt.ml.HistogramGradientBoosting;
import basalt.ml.IsolationForest;
import basalt.ml.KBinsDiscretizer;
import basalt.ml.KMeans;
import basalt.ml.KernelDensity;
import basalt.ml.KernelPCA;
import basalt.ml.KernelRidge;
import basalt.ml.LabelEncoder;
import basalt.ml.Lasso;
import basalt.ml.LinearRegression;
import basalt.ml.LinearSVM;
import basalt.ml.LogisticRegression;
import basalt.ml.MinMaxScaler;
import basalt.ml.OneHotEncoder;
import basalt.ml.PCA;
import basalt.ml.Pipeline;
import basalt.ml.PolynomialFeatures;
import basalt.ml.RandomForest;
import basalt.ml.Ridge;
import basalt.ml.SVC;
import basalt.ml.SelectKBest;
import basalt.ml.SimpleImputer;
import basalt.ml.StandardScaler;
import basalt.nn.CNN;
import basalt.nn.GRU;
import basalt.nn.LSTM;
import basalt.nn.ModelSerializer;
import basalt.nn.SelfAttention;
import basalt.nn.SequenceNetwork;
import basalt.nn.VanillaRNN;
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
 * Versioned .bsl container: 24-byte header + CRC32C-checked payload.
 *
 * <pre>
 * int  magic     0x42534C54 ("BSLT")
 * int  version   2
 * int  kind      stable ModelKind code
 * int  length    payload bytes
 * long crc32c    checksum of the payload
 * byte[length]   model-specific payload
 * </pre>
 */
public final class ModelCodec {
    public static final int MAGIC = 0x42534C54;
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
            throw new ModelFormatException("Serialized model exceeds the .bsl container limit");
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
                throw new ModelFormatException("File is shorter than the .bsl header");
            }
            MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, size);
            int magic = mapped.getInt();
            int version = mapped.getInt();
            if (magic != MAGIC) {
                throw new ModelFormatException("Invalid .bsl magic");
            }
            if (version != VERSION) {
                throw new ModelFormatException("Unsupported .bsl version " + version);
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
                    throw new ModelFormatException("Truncated .bsl header");
                }
            }
            header.flip();
            if (header.getInt() != MAGIC) {
                throw new ModelFormatException("Invalid .bsl magic");
            }
            int version = header.getInt();
            if (version != VERSION) {
                throw new ModelFormatException("Unsupported .bsl version " + version);
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
