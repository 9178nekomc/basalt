package basalt.data;

import basalt.core.Matrix;
import java.io.EOFException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;
import java.util.Random;

/**
 * Allocation-stable streaming over BasaltD records. Every record is {@code featureCount} doubles
 * followed by one label double.
 *
 * <p>Two readers share one {@link Batch} contract:</p>
 * <ul>
 *   <li>{@link BatchStream} performs sequential channel reads into a direct buffer and decodes
 *       with a {@link DoubleBuffer} view. The batch, its feature array, its label array and the
 *       byte buffer are allocated once per stream.</li>
 *   <li>{@link MappedBatchStream} memory-maps the file and gathers rows with absolute bulk
 *       {@code DoubleBuffer.get(index, dst, off, len)} calls, so an epoch performs zero syscalls
 *       after the initial map. Optional shuffling permutes an {@code int[]} row index, not the
 *       data.</li>
 * </ul>
 */
public final class DataLoader {
    private static final int MAGIC = 0x4a534c44;
    private static final int VERSION = 1;
    private static final int HEADER_BYTES = 16;
    private static final int IO_BUFFER_BYTES = 1 << 20;

    private DataLoader() {
    }

    public static Builder builder(Path source) {
        return new Builder(source);
    }

    public static Normalization estimateNormalization(Path source, int featureCount) {
        double[] means = new double[featureCount];
        double[] m2 = new double[featureCount];
        long count = 0;
        try (BatchStream stream = builder(source).featureCount(featureCount).batchSize(1024).open()) {
            while (stream.next()) {
                Batch batch = stream.batch();
                for (int row = 0; row < batch.rows(); row++) {
                    count++;
                    int offset = row * featureCount;
                    for (int feature = 0; feature < featureCount; feature++) {
                        double value = batch.features()[offset + feature];
                        double delta = value - means[feature];
                        means[feature] += delta / count;
                        m2[feature] += delta * (value - means[feature]);
                    }
                }
            }
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
        if (count < 2) {
            throw new IllegalArgumentException("Normalization requires at least two samples");
        }
        double[] standardDeviations = new double[featureCount];
        for (int feature = 0; feature < featureCount; feature++) {
            standardDeviations[feature] = Math.sqrt(m2[feature] / (count - 1));
        }
        return Normalization.standardize(means, standardDeviations);
    }

    public static void write(Path target, Matrix features, double[] labels) {
        Objects.requireNonNull(features, "features");
        Objects.requireNonNull(labels, "labels");
        if (features.rows() != labels.length) {
            throw new IllegalArgumentException("Label count mismatch");
        }
        int featureCount = features.columns();
        int recordBytes = Math.multiplyExact(featureCount + 1, Double.BYTES);
        ByteBuffer buffer = ByteBuffer.allocate(Math.max(IO_BUFFER_BYTES, recordBytes * 2));
        double[] record = new double[featureCount + 1];
        try (FileChannel channel = FileChannel.open(target,
                StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            buffer.putInt(MAGIC);
            buffer.putInt(VERSION);
            buffer.putInt(featureCount);
            buffer.putInt(0);
            for (int row = 0; row < features.rows(); row++) {
                if (buffer.remaining() < recordBytes) {
                    buffer.flip();
                    while (buffer.hasRemaining()) {
                        channel.write(buffer);
                    }
                    buffer.clear();
                }
                features.copyRow(row, record, 0);
                record[featureCount] = labels[row];
                buffer.asDoubleBuffer().put(record);
                buffer.position(buffer.position() + recordBytes);
            }
            buffer.flip();
            while (buffer.hasRemaining()) {
                channel.write(buffer);
            }
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
    }

    /** Shared configuration for both readers. */
    public static final class Builder {
        private final Path source;
        private int featureCount;
        private int batchSize = 1024;
        private Normalization normalization;
        private boolean shuffle;
        private long seed = 42;

        private Builder(Path source) {
            this.source = Objects.requireNonNull(source, "source");
        }

        public Builder featureCount(int featureCount) {
            this.featureCount = featureCount;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder normalization(Normalization normalization) {
            this.normalization = normalization;
            return this;
        }

        /** Only affects {@link #openMapped()}; permutes row order per epoch. */
        public Builder shuffle(boolean shuffle) {
            this.shuffle = shuffle;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public BatchStream open() {
            validate();
            try {
                FileChannel channel = FileChannel.open(source, StandardOpenOption.READ);
                boolean success = false;
                try {
                    readAndValidateHeader(channel, featureCount);
                    int recordBytes = Math.multiplyExact(featureCount + 1, Double.BYTES);
                    int bufferBytes = Math.multiplyExact(recordBytes, batchSize);
                    BatchStream stream = new BatchStream(channel, featureCount, batchSize,
                            recordBytes, bufferBytes, normalization);
                    success = true;
                    return stream;
                } finally {
                    if (!success) {
                        channel.close();
                    }
                }
            } catch (IOException failure) {
                throw new UncheckedIOException(failure);
            }
        }

        public MappedBatchStream openMapped() {
            validate();
            try {
                FileChannel channel = FileChannel.open(source, StandardOpenOption.READ);
                boolean success = false;
                try {
                    readAndValidateHeader(channel, featureCount);
                    long size = channel.size();
                    int recordBytes = Math.multiplyExact(featureCount + 1, Double.BYTES);
                    long payload = size - HEADER_BYTES;
                    if (payload % recordBytes != 0) {
                        throw new IOException("Truncated BasaltD record");
                    }
                    long rows = payload / recordBytes;
                    if (rows > Integer.MAX_VALUE) {
                        throw new IOException("BasaltD row count exceeds the int index range");
                    }
                    MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, size);
                    DoubleBuffer data = mapped.position(HEADER_BYTES).slice().asDoubleBuffer();
                    MappedBatchStream stream = new MappedBatchStream(featureCount, batchSize,
                            (int) rows, data, normalization, shuffle, seed);
                    success = true;
                    return stream;
                } finally {
                    if (!success) {
                        channel.close();
                    }
                }
            } catch (IOException failure) {
                throw new UncheckedIOException(failure);
            }
        }

        private void validate() {
            if (featureCount <= 0 || batchSize <= 0) {
                throw new IllegalArgumentException("featureCount and batchSize must be positive");
            }
            if (normalization != null && normalization.featureCount() != featureCount) {
                throw new IllegalArgumentException("Normalization feature count mismatch");
            }
        }

        private static void readAndValidateHeader(FileChannel channel, int expectedFeatures)
                throws IOException {
            ByteBuffer header = ByteBuffer.allocate(HEADER_BYTES);
            while (header.hasRemaining()) {
                if (channel.read(header) < 0) {
                    throw new EOFException("Truncated BasaltD header");
                }
            }
            header.flip();
            int magic = header.getInt();
            int version = header.getInt();
            int features = header.getInt();
            int reserved = header.getInt();
            if (magic != MAGIC || version != VERSION || features != expectedFeatures || reserved != 0) {
                throw new IOException("Invalid or incompatible BasaltD header");
            }
        }
    }

    public static final class Batch {
        private final int featureCount;
        private final double[] features;
        private final double[] labels;
        private int rows;

        private Batch(int featureCount, int batchSize) {
            this.featureCount = featureCount;
            features = new double[featureCount * batchSize];
            labels = new double[batchSize];
        }

        public int rows() {
            return rows;
        }

        public int featureCount() {
            return featureCount;
        }

        /** Live reusable storage; only the first rows()*featureCount() entries are valid. */
        public double[] features() {
            return features;
        }

        /** Live reusable storage; only the first rows() entries are valid. */
        public double[] labels() {
            return labels;
        }

        public double feature(int row, int feature) {
            if (row < 0 || row >= rows || feature < 0 || feature >= featureCount) {
                throw new IndexOutOfBoundsException("Batch index outside current rows/features");
            }
            return features[row * featureCount + feature];
        }

        public double label(int row) {
            if (row < 0 || row >= rows) {
                throw new IndexOutOfBoundsException("Batch row outside current batch");
            }
            return labels[row];
        }
    }

    /** Sequential reader; {@link #rewind()} restarts without reopening the file. */
    public static final class BatchStream implements AutoCloseable {
        private final FileChannel channel;
        private final int batchSize;
        private final int recordBytes;
        private final ByteBuffer buffer;
        private final Normalization normalization;
        private final Batch batch;
        private boolean endOfFile;
        private boolean closed;

        private BatchStream(FileChannel channel, int featureCount, int batchSize,
                            int recordBytes, int bufferBytes, Normalization normalization) {
            this.channel = channel;
            this.batchSize = batchSize;
            this.recordBytes = recordBytes;
            this.normalization = normalization;
            buffer = ByteBuffer.allocateDirect(bufferBytes);
            buffer.limit(0);
            batch = new Batch(featureCount, batchSize);
        }

        /** Advances without allocating a Batch, feature array, label array or byte buffer. */
        public boolean next() throws IOException {
            if (closed) {
                throw new IOException("BatchStream is closed");
            }
            buffer.compact();
            while (buffer.hasRemaining() && !endOfFile) {
                int read = channel.read(buffer);
                if (read < 0) {
                    endOfFile = true;
                }
            }
            buffer.flip();
            int bytes = buffer.remaining();
            if (endOfFile && bytes % recordBytes != 0) {
                throw new EOFException("Truncated BasaltD record");
            }
            int availableRows = bytes / recordBytes;
            if (availableRows == 0) {
                batch.rows = 0;
                return false;
            }
            int rows = Math.min(batchSize, availableRows);
            DoubleBuffer view = buffer.asDoubleBuffer();
            double[] features = batch.features;
            double[] labels = batch.labels;
            for (int row = 0; row < rows; row++) {
                int offset = row * batch.featureCount;
                view.get(features, offset, batch.featureCount);
                labels[row] = view.get();
                if (normalization != null) {
                    normalization.apply(features, offset);
                }
            }
            buffer.position(buffer.position() + rows * recordBytes);
            batch.rows = rows;
            return true;
        }

        public Batch batch() {
            return batch;
        }

        /** Restarts at the first record; buffers are retained. */
        public void rewind() throws IOException {
            if (closed) {
                throw new IOException("BatchStream is closed");
            }
            channel.position(HEADER_BYTES);
            buffer.clear();
            buffer.limit(0);
            endOfFile = false;
            batch.rows = 0;
        }

        @Override
        public void close() throws IOException {
            if (!closed) {
                closed = true;
                channel.close();
            }
        }
    }

    /**
     * Memory-mapped reader with epoch shuffling. Every {@link #next()} gathers rows through
     * absolute bulk reads; no byte buffer, batch, feature array or label array is allocated.
     */
    public static final class MappedBatchStream implements AutoCloseable {
        private final int featureCount;
        private final int batchSize;
        private final int recordWidth;
        private final int totalRows;
        private final DoubleBuffer data;
        private final Normalization normalization;
        private final Batch batch;
        private final int[] order;
        private final boolean shuffle;
        private final long seed;
        private int cursor;
        private int epoch;
        private boolean closed;

        private MappedBatchStream(int featureCount, int batchSize, int totalRows,
                                  DoubleBuffer data, Normalization normalization,
                                  boolean shuffle, long seed) {
            this.featureCount = featureCount;
            this.batchSize = batchSize;
            this.recordWidth = featureCount + 1;
            this.totalRows = totalRows;
            this.data = data;
            this.normalization = normalization;
            this.batch = new Batch(featureCount, batchSize);
            this.order = new int[totalRows];
            this.shuffle = shuffle;
            this.seed = seed;
            for (int row = 0; row < totalRows; row++) {
                order[row] = row;
            }
            if (shuffle) {
                permute(new Random(seed));
            }
        }

        public boolean next() {
            if (closed) {
                throw new IllegalStateException("MappedBatchStream is closed");
            }
            if (cursor >= totalRows) {
                batch.rows = 0;
                return false;
            }
            int rows = Math.min(batchSize, totalRows - cursor);
            double[] features = batch.features;
            double[] labels = batch.labels;
            for (int row = 0; row < rows; row++) {
                int sourceRow = order[cursor + row];
                int sourceIndex = sourceRow * recordWidth;
                int offset = row * featureCount;
                data.get(sourceIndex, features, offset, featureCount);
                labels[row] = data.get(sourceIndex + featureCount);
                if (normalization != null) {
                    normalization.apply(features, offset);
                }
            }
            cursor += rows;
            batch.rows = rows;
            return true;
        }

        public Batch batch() {
            return batch;
        }

        /** Starts a new epoch; reshuffles when shuffle mode is enabled. */
        public void rewind() {
            if (closed) {
                throw new IllegalStateException("MappedBatchStream is closed");
            }
            cursor = 0;
            epoch++;
            if (shuffle) {
                permute(new Random(seed + epoch * 1_000_003L));
            }
        }

        public int totalRows() {
            return totalRows;
        }

        public int epoch() {
            return epoch;
        }

        private void permute(Random random) {
            for (int i = totalRows - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int temporary = order[i];
                order[i] = order[j];
                order[j] = temporary;
            }
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    /** Total number of records in a BasaltD file, validating the header. */
    public static long rowCount(Path source, int featureCount) {
        try {
            long size = Files.size(source);
            int recordBytes = Math.multiplyExact(featureCount + 1, Double.BYTES);
            long payload = size - HEADER_BYTES;
            if (payload < 0 || payload % recordBytes != 0) {
                throw new IOException("Truncated BasaltD file");
            }
            return payload / recordBytes;
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
    }
}
