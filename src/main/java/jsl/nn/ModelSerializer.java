package jsl.nn;

import jsl.core.ModelFormatException;
import jsl.core.ModelReader;
import jsl.core.ModelWriter;
import jsl.io.ModelCodec;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * CNN persistence facade. The on-disk container (magic, version, CRC32C) lives in
 * {@link ModelCodec}; this class owns the CNN layer payload.
 */
public final class ModelSerializer {
    private static final int CONV2D = 1;
    private static final int MAX_POOLING2D = 2;
    private static final int FLATTEN = 3;
    private static final int DENSE = 4;

    private ModelSerializer() {
    }

    public static void save(Path target, CNN model) {
        Objects.requireNonNull(model, "model");
        ModelCodec.save(target, model);
    }

    public static CNN loadCNN(Path source) {
        return ModelCodec.load(source, CNN.class);
    }

    public static void writeCNN(ModelWriter output, CNN model) {
        output.putInt(model.loss().ordinal());
        output.putInt(model.optimizer().optimizer().ordinal());
        output.putDouble(model.optimizer().learningRate());
        output.putDouble(model.optimizer().beta1());
        output.putDouble(model.optimizer().beta2());
        output.putDouble(model.optimizer().epsilon());
        output.putInt(model.layers().size());
        for (Layer<?, ?> layer : model.layers()) {
            writeLayer(output, layer);
        }
    }

    public static CNN readCNN(ModelReader input) {
        Loss[] losses = Loss.values();
        int lossOrdinal = input.getInt();
        if (lossOrdinal < 0 || lossOrdinal >= losses.length) {
            throw new ModelFormatException("Invalid loss ordinal");
        }
        Loss loss = losses[lossOrdinal];
        Optimizer[] optimizers = Optimizer.values();
        int optimizerOrdinal = input.getInt();
        if (optimizerOrdinal < 0 || optimizerOrdinal >= optimizers.length) {
            throw new ModelFormatException("Invalid optimizer ordinal");
        }
        OptimizerConfig optimizer = new OptimizerConfig(optimizers[optimizerOrdinal],
                input.getDouble(), input.getDouble(), input.getDouble(), input.getDouble());
        int layerCount = input.getInt();
        if (layerCount <= 0 || layerCount > 100_000) {
            throw new ModelFormatException("Invalid layer count");
        }
        List<Layer<?, ?>> layers = new ArrayList<>(layerCount);
        for (int i = 0; i < layerCount; i++) {
            layers.add(readLayer(input));
        }
        validateTopology(layers);
        return new CNN(layers, loss, optimizer);
    }

    private static void writeLayer(ModelWriter output, Layer<?, ?> layer) {
        if (layer instanceof Conv2D convolution) {
            output.putInt(CONV2D);
            output.putInt(convolution.filters());
            output.putInt(convolution.kernelHeight());
            output.putInt(convolution.kernelWidth());
            output.putInt(convolution.strideHeight());
            output.putInt(convolution.strideWidth());
            output.putInt(convolution.padding().ordinal());
            output.putInt(convolution.activation().ordinal());
            output.putLong(convolution.seed());
            output.putInt(convolution.inputChannels());
            if (convolution.inputChannels() > 0) {
                output.putDoubles(convolution.kernelValues());
                output.putDoubles(convolution.biasValues());
            }
        } else if (layer instanceof MaxPooling2D pooling) {
            output.putInt(MAX_POOLING2D);
            output.putInt(pooling.poolHeight());
            output.putInt(pooling.poolWidth());
            output.putInt(pooling.strideHeight());
            output.putInt(pooling.strideWidth());
        } else if (layer instanceof Flatten) {
            output.putInt(FLATTEN);
        } else if (layer instanceof Dense dense) {
            output.putInt(DENSE);
            dense.write(output);
        } else {
            throw new IllegalStateException(
                    "Unsupported CNN layer type: " + layer.getClass().getName());
        }
    }

    private static Layer<?, ?> readLayer(ModelReader input) {
        int type = input.getInt();
        switch (type) {
            case CONV2D -> {
                int filters = input.getInt();
                int kernelHeight = input.getInt();
                int kernelWidth = input.getInt();
                int strideHeight = input.getInt();
                int strideWidth = input.getInt();
                int paddingOrdinal = input.getInt();
                int activationOrdinal = input.getInt();
                Padding[] paddings = Padding.values();
                Activation[] activations = Activation.values();
                if (paddingOrdinal < 0 || paddingOrdinal >= paddings.length
                        || activationOrdinal < 0 || activationOrdinal >= activations.length) {
                    throw new ModelFormatException("Invalid convolution enum ordinal");
                }
                long seed = input.getLong();
                int channels = input.getInt();
                Conv2D convolution = Conv2D.builder().filters(filters)
                        .kernelSize(kernelHeight, kernelWidth)
                        .strides(strideHeight, strideWidth)
                        .padding(paddings[paddingOrdinal])
                        .activation(activations[activationOrdinal]).seed(seed).build();
                if (channels > 0) {
                    double[] kernels = input.getDoubles(Math.multiplyExact(
                            Math.multiplyExact(filters, channels),
                            Math.multiplyExact(kernelHeight, kernelWidth)));
                    double[] bias = input.getDoubles(filters);
                    convolution.restore(channels, kernels, bias);
                } else if (channels != -1) {
                    throw new ModelFormatException("Invalid convolution input channel marker");
                }
                return convolution;
            }
            case MAX_POOLING2D -> {
                return MaxPooling2D.builder().poolSize(input.getInt(), input.getInt())
                        .strides(input.getInt(), input.getInt()).build();
            }
            case FLATTEN -> {
                return Flatten.builder().build();
            }
            case DENSE -> {
                return Dense.read(input);
            }
            default -> throw new ModelFormatException("Unknown JSL layer type " + type);
        }
    }

    private static void validateTopology(List<Layer<?, ?>> layers) {
        boolean flattened = false;
        for (Layer<?, ?> layer : layers) {
            if (layer instanceof Flatten) {
                if (flattened) {
                    throw new ModelFormatException("CNN contains multiple Flatten layers");
                }
                flattened = true;
            } else if (flattened) {
                if (!(layer instanceof Dense)) {
                    throw new ModelFormatException("Only Dense layers may follow Flatten");
                }
            } else if (!(layer instanceof Conv2D || layer instanceof MaxPooling2D)) {
                throw new ModelFormatException("Only image layers may precede Flatten");
            }
        }
        if (!flattened) {
            throw new ModelFormatException("CNN requires a Flatten transition");
        }
    }
}
