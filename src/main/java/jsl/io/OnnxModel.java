package jsl.io;

import jsl.core.Matrix;
import jsl.nativebridge.NativeContext;
import jsl.torch.Tensor;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Standard ONNX (Open Neural Network Exchange) {@code ModelProto} representation and native-backed
 * inference engine.
 *
 * <p>Implements a zero-dependency Protobuf wire-format decoder/encoder for the official
 * {@code onnx.proto3} schema (IR v8, Opset 17) so Python-trained ONNX models (from PyTorch,
 * scikit-learn {@code skl2onnx}, or ONNX Runtime) can be loaded and executed directly in the JVM,
 * and JSL models can be exported to `.onnx` files consumable by Python `onnxruntime`.</p>
 */
public final class OnnxModel {
    public record TensorData(String name, int[] dims, float[] floatData) {
        public TensorData {
            Objects.requireNonNull(name, "name");
            dims = dims.clone();
            floatData = floatData.clone();
        }
    }

    public record Attribute(String name, int type, long intVal, float floatVal, String stringVal,
                            long[] intsVal, float[] floatsVal) {
        public static final int FLOAT = 1;
        public static final int INT = 2;
        public static final int STRING = 3;
        public static final int FLOATS = 6;
        public static final int INTS = 7;

        public static Attribute ofInt(String name, long value) {
            return new Attribute(name, INT, value, 0.0f, "", new long[0], new float[0]);
        }

        public static Attribute ofFloat(String name, float value) {
            return new Attribute(name, FLOAT, 0L, value, "", new long[0], new float[0]);
        }

        public static Attribute ofInts(String name, long... values) {
            return new Attribute(name, INTS, 0L, 0.0f, "", values.clone(), new float[0]);
        }

        public static Attribute ofFloats(String name, float... values) {
            return new Attribute(name, FLOATS, 0L, 0.0f, "", new long[0], values.clone());
        }
    }

    public record Node(String name, String opType, List<String> inputs, List<String> outputs,
                       Map<String, Attribute> attributes) {
        public long attrInt(String key, long defaultVal) {
            Attribute a = attributes.get(key);
            return a != null && a.type() == Attribute.INT ? a.intVal() : defaultVal;
        }

        public float attrFloat(String key, float defaultVal) {
            Attribute a = attributes.get(key);
            return a != null && a.type() == Attribute.FLOAT ? a.floatVal() : defaultVal;
        }

        public int[] attrInts(String key, int[] defaultVal) {
            Attribute a = attributes.get(key);
            if (a == null || a.type() != Attribute.INTS) return defaultVal;
            int[] res = new int[a.intsVal().length];
            for (int i = 0; i < res.length; i++) res[i] = (int) a.intsVal()[i];
            return res;
        }
    }

    public record ValueInfo(String name, int elemType, int[] shape) {
    }

    private final long irVersion;
    private final long opsetVersion;
    private final String producerName;
    private final String graphName;
    private final List<ValueInfo> inputs;
    private final List<ValueInfo> outputs;
    private final Map<String, TensorData> initializers;
    private final List<Node> nodes;

    public OnnxModel(long irVersion, long opsetVersion, String producerName, String graphName,
                     List<ValueInfo> inputs, List<ValueInfo> outputs,
                     Map<String, TensorData> initializers, List<Node> nodes) {
        this.irVersion = irVersion;
        this.opsetVersion = opsetVersion;
        this.producerName = producerName;
        this.graphName = graphName;
        this.inputs = List.copyOf(inputs);
        this.outputs = List.copyOf(outputs);
        this.initializers = Collections.unmodifiableMap(new LinkedHashMap<>(initializers));
        this.nodes = List.copyOf(nodes);
    }

    public static OnnxModel load(Path path) {
        try {
            return parseModelProto(Files.readAllBytes(path));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public void save(Path path) {
        try {
            Files.write(path, toBytes());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public long irVersion() {
        return irVersion;
    }

    public long opsetVersion() {
        return opsetVersion;
    }

    public String producerName() {
        return producerName;
    }

    public String graphName() {
        return graphName;
    }

    public List<ValueInfo> inputs() {
        return inputs;
    }

    public List<ValueInfo> outputs() {
        return outputs;
    }

    public Map<String, TensorData> initializers() {
        return initializers;
    }

    public List<Node> nodes() {
        return nodes;
    }

    /**
     * Convenience inference entry point for 2D tabular {@link Matrix} inputs.
     */
    public Matrix predictMatrix(Matrix features) {
        int rows = features.rows();
        int cols = features.columns();
        double[] src = features.data();
        float[] fData = new float[src.length];
        for (int i = 0; i < src.length; i++) fData[i] = (float) src[i];
        try (NativeContext ctx = NativeContext.create(1)) {
            Tensor inTensor = Tensor.of(ctx, fData, rows, cols);
            Tensor outTensor = run(ctx, inTensor);
            int[] outShape = outTensor.shape();
            float[] outData = outTensor.data();
            double[] dst = new double[outData.length];
            for (int i = 0; i < outData.length; i++) dst[i] = outData[i];
            int outCols = outShape.length >= 2 ? outShape[1] : 1;
            return new Matrix(rows, outCols, dst);
        }
    }

    /**
     * Single-input / single-output inference on an active {@link NativeContext}.
     */
    public Tensor run(NativeContext context, Tensor input) {
        if (inputs.isEmpty()) {
            throw new IllegalStateException("ONNX graph has no declared inputs");
        }
        Map<String, Tensor> feed = Map.of(inputs.get(0).name(), input);
        Map<String, Tensor> result = run(context, feed);
        return result.get(outputs.get(0).name());
    }

    /**
     * Executes the ONNX computation graph for the given input map and returns all graph outputs.
     */
    public Map<String, Tensor> run(NativeContext context, Map<String, Tensor> feed) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(feed, "feed");
        Map<String, Tensor> env = new HashMap<>();
        for (Map.Entry<String, TensorData> entry : initializers.entrySet()) {
            TensorData td = entry.getValue();
            int[] shape = td.dims().length == 0 ? new int[]{1} : td.dims();
            env.put(entry.getKey(), Tensor.of(context, td.floatData(), shape));
        }
        env.putAll(feed);

        for (Node node : nodes) {
            executeNode(context, node, env);
        }

        Map<String, Tensor> outMap = new LinkedHashMap<>();
        for (ValueInfo outInfo : outputs) {
            Tensor t = env.get(outInfo.name());
            if (t == null) {
                throw new IllegalStateException("ONNX execution did not produce output " + outInfo.name());
            }
            outMap.put(outInfo.name(), t);
        }
        return outMap;
    }

    private void executeNode(NativeContext ctx, Node node, Map<String, Tensor> env) {
        String op = node.opType();
        List<String> in = node.inputs();
        String outName = node.outputs().get(0);

        switch (op) {
            case "MatMul" -> {
                Tensor a = env.get(in.get(0));
                Tensor b = env.get(in.get(1));
                env.put(outName, a.rank() == 3 && b.rank() == 3 ? a.batchedMatMul(b) : a.matMul(b));
            }
            case "Gemm" -> {
                Tensor a = env.get(in.get(0));
                Tensor b = env.get(in.get(1));
                long transA = node.attrInt("transA", 0L);
                long transB = node.attrInt("transB", 0L);
                float alpha = node.attrFloat("alpha", 1.0f);
                float beta = node.attrFloat("beta", 1.0f);
                if (transA != 0) a = a.permute(1, 0);
                if (transB != 0) b = b.permute(1, 0);
                Tensor prod = a.matMul(b);
                if (alpha != 1.0f) prod = prod.mul(Tensor.scalar(ctx, alpha));
                if (in.size() >= 3 && !in.get(2).isEmpty()) {
                    Tensor c = env.get(in.get(2));
                    if (beta != 1.0f) c = c.mul(Tensor.scalar(ctx, beta));
                    prod = prod.add(c);
                }
                env.put(outName, prod);
            }
            case "Add" -> env.put(outName, env.get(in.get(0)).add(env.get(in.get(1))));
            case "Sub" -> env.put(outName, env.get(in.get(0)).sub(env.get(in.get(1))));
            case "Mul" -> env.put(outName, env.get(in.get(0)).mul(env.get(in.get(1))));
            case "Div" -> {
                Tensor a = env.get(in.get(0));
                Tensor b = env.get(in.get(1));
                float[] bData = b.data();
                float[] inv = new float[bData.length];
                for (int i = 0; i < bData.length; i++) inv[i] = 1.0f / bData[i];
                env.put(outName, a.mul(Tensor.of(ctx, inv, b.shape())));
            }
            case "Relu" -> env.put(outName, env.get(in.get(0)).relu());
            case "Sigmoid" -> env.put(outName, env.get(in.get(0)).sigmoid());
            case "Tanh" -> env.put(outName, env.get(in.get(0)).tanh());
            case "Gelu" -> env.put(outName, env.get(in.get(0)).gelu());
            case "Silu" -> env.put(outName, env.get(in.get(0)).silu());
            case "LeakyRelu" -> {
                float alpha = node.attrFloat("alpha", 0.01f);
                env.put(outName, env.get(in.get(0)).leakyRelu(alpha));
            }
            case "Elu" -> {
                float alpha = node.attrFloat("alpha", 1.0f);
                env.put(outName, env.get(in.get(0)).elu(alpha));
            }
            case "Softmax" -> {
                int axis = (int) node.attrInt("axis", -1L);
                env.put(outName, env.get(in.get(0)).softmax(axis));
            }
            case "Identity", "Dropout" -> env.put(outName, env.get(in.get(0)));
            case "Flatten" -> {
                Tensor x = env.get(in.get(0));
                int[] s = x.shape();
                int axis = (int) node.attrInt("axis", 1L);
                if (axis < 0) axis += s.length;
                int d0 = 1;
                for (int i = 0; i < axis; i++) d0 *= s[i];
                int d1 = x.size() / d0;
                env.put(outName, x.reshape(d0, d1));
            }
            case "Reshape" -> {
                Tensor x = env.get(in.get(0));
                Tensor shapeTensor = env.get(in.get(1));
                float[] rawDims = shapeTensor.data();
                int[] targetShape = new int[rawDims.length];
                int inferIdx = -1;
                int knownProd = 1;
                for (int i = 0; i < rawDims.length; i++) {
                    int d = (int) rawDims[i];
                    if (d == 0) d = x.shape()[i];
                    if (d == -1) {
                        inferIdx = i;
                    } else {
                        targetShape[i] = d;
                        knownProd *= d;
                    }
                }
                if (inferIdx >= 0) {
                    targetShape[inferIdx] = x.size() / knownProd;
                }
                env.put(outName, x.reshape(targetShape));
            }
            case "Transpose" -> {
                Tensor x = env.get(in.get(0));
                int[] defaultPerm = new int[x.rank()];
                for (int i = 0; i < defaultPerm.length; i++) defaultPerm[i] = defaultPerm.length - 1 - i;
                int[] perm = node.attrInts("perm", defaultPerm);
                env.put(outName, x.permute(perm));
            }
            case "Conv" -> {
                Tensor x = env.get(in.get(0));
                Tensor w = env.get(in.get(1));
                int[] kShape = w.shape();
                int[] strides = node.attrInts("strides", new int[]{1, 1});
                int[] pads = node.attrInts("pads", new int[]{0, 0, 0, 0});
                int[] dilations = node.attrInts("dilations", new int[]{1, 1});
                int group = (int) node.attrInt("group", 1L);
                Tensor out = x.conv2d(w, kShape[2], kShape[3],
                        strides[0], strides[1], pads[0], pads[1],
                        dilations[0], dilations[1], group);
                if (in.size() >= 3 && !in.get(2).isEmpty()) {
                    Tensor b = env.get(in.get(2));
                    if (b.rank() == 1) {
                        b = b.reshape(1, b.size(), 1, 1);
                    }
                    out = out.add(b);
                }
                env.put(outName, out);
            }
            case "MaxPool" -> {
                Tensor x = env.get(in.get(0));
                int[] kernel = node.attrInts("kernel_shape", new int[]{2, 2});
                int[] strides = node.attrInts("strides", kernel);
                env.put(outName, x.maxPool2d(kernel[0], kernel[1], strides[0], strides[1]));
            }
            case "BatchNormalization" -> {
                Tensor x = env.get(in.get(0));
                Tensor scale = env.get(in.get(1));
                Tensor b = env.get(in.get(2));
                Tensor mean = env.get(in.get(3));
                Tensor var = env.get(in.get(4));
                float eps = node.attrFloat("epsilon", 1e-5f);
                float mom = node.attrFloat("momentum", 0.9f);
                env.put(outName, x.batchNorm2d(scale, b, mean, var, eps, 1.0f - mom, false));
            }
            case "LayerNormalization" -> {
                Tensor x = env.get(in.get(0));
                Tensor scale = env.get(in.get(1));
                Tensor b = in.size() >= 3 && !in.get(2).isEmpty()
                        ? env.get(in.get(2))
                        : Tensor.zeros(ctx, scale.shape());
                float eps = node.attrFloat("epsilon", 1e-5f);
                env.put(outName, x.layerNorm(scale, b, eps));
            }
            case "GreaterOrEqual" -> {
                Tensor a = env.get(in.get(0));
                Tensor b = env.get(in.get(1));
                float[] ad = a.data();
                float[] bd = b.data();
                float[] out = new float[ad.length];
                for (int i = 0; i < ad.length; i++) {
                    float bv = bd.length == 1 ? bd[0] : bd[i % bd.length];
                    out[i] = ad[i] >= bv ? 1.0f : 0.0f;
                }
                env.put(outName, Tensor.of(ctx, out, a.shape()));
            }
            case "Where" -> {
                Tensor cond = env.get(in.get(0));
                Tensor x = env.get(in.get(1));
                Tensor y = env.get(in.get(2));
                float[] cd = cond.data();
                float[] xd = x.data();
                float[] yd = y.data();
                float[] out = new float[cd.length];
                for (int i = 0; i < cd.length; i++) {
                    float xv = xd.length == 1 ? xd[0] : xd[i % xd.length];
                    float yv = yd.length == 1 ? yd[0] : yd[i % yd.length];
                    out[i] = cd[i] != 0.0f ? xv : yv;
                }
                env.put(outName, Tensor.of(ctx, out, cond.shape()));
            }
            default -> throw new jsl.JSLUnsupportedOperationException(
                    "Unsupported ONNX operator in JSL runtime: " + op);
        }
    }

    // ====================== Protobuf Wire Serialization (onnx.proto3) ======================

    public byte[] toBytes() {
        ProtoWriter model = new ProtoWriter();
        model.writeVarintField(1, irVersion); // ir_version = 1
        // opset_import = 8 (OperatorSetIdProto: domain=1, version=2)
        ProtoWriter opset = new ProtoWriter();
        opset.writeStringField(1, "");
        opset.writeVarintField(2, opsetVersion);
        model.writeBytesField(8, opset.toByteArray());
        model.writeStringField(2, producerName); // producer_name = 2
        model.writeStringField(3, "2.4.0");      // producer_version = 3

        ProtoWriter graph = new ProtoWriter();
        for (Node n : nodes) {
            graph.writeBytesField(1, encodeNode(n));
        }
        graph.writeStringField(2, graphName);
        for (TensorData init : initializers.values()) {
            graph.writeBytesField(5, encodeTensor(init));
        }
        for (ValueInfo vi : inputs) {
            graph.writeBytesField(11, encodeValueInfo(vi));
        }
        for (ValueInfo vo : outputs) {
            graph.writeBytesField(12, encodeValueInfo(vo));
        }
        model.writeBytesField(7, graph.toByteArray()); // graph = 7
        return model.toByteArray();
    }

    private static byte[] encodeNode(Node node) {
        ProtoWriter w = new ProtoWriter();
        for (String in : node.inputs()) w.writeStringField(1, in);
        for (String out : node.outputs()) w.writeStringField(2, out);
        if (node.name() != null && !node.name().isEmpty()) {
            w.writeStringField(3, node.name());
        }
        w.writeStringField(4, node.opType());
        for (Attribute attr : node.attributes().values()) {
            w.writeBytesField(5, encodeAttribute(attr));
        }
        return w.toByteArray();
    }

    private static byte[] encodeAttribute(Attribute attr) {
        ProtoWriter w = new ProtoWriter();
        w.writeStringField(1, attr.name());
        if (attr.type() == Attribute.FLOAT) {
            w.writeFloatField(2, attr.floatVal());
        } else if (attr.type() == Attribute.INT) {
            w.writeVarintField(3, attr.intVal());
        } else if (attr.type() == Attribute.STRING) {
            w.writeStringField(4, attr.stringVal());
        } else if (attr.type() == Attribute.FLOATS) {
            for (float f : attr.floatsVal()) w.writeFloatField(7, f);
        } else if (attr.type() == Attribute.INTS) {
            for (long iv : attr.intsVal()) w.writeVarintField(8, iv);
        }
        w.writeVarintField(20, attr.type());
        return w.toByteArray();
    }

    private static byte[] encodeTensor(TensorData td) {
        ProtoWriter w = new ProtoWriter();
        for (int d : td.dims()) {
            w.writeVarintField(1, d);
        }
        w.writeVarintField(2, 1); // DataType.FLOAT = 1
        w.writeStringField(8, td.name());
        ByteBuffer raw = ByteBuffer.allocate(td.floatData().length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : td.floatData()) raw.putFloat(v);
        w.writeBytesField(9, raw.array()); // raw_data = 9
        return w.toByteArray();
    }

    private static byte[] encodeValueInfo(ValueInfo vi) {
        ProtoWriter w = new ProtoWriter();
        w.writeStringField(1, vi.name());

        ProtoWriter tensorType = new ProtoWriter();
        tensorType.writeVarintField(1, vi.elemType()); // elem_type = 1 (FLOAT)
        if (vi.shape() != null) {
            ProtoWriter shapeProto = new ProtoWriter();
            for (int d : vi.shape()) {
                ProtoWriter dimProto = new ProtoWriter();
                if (d > 0) {
                    dimProto.writeVarintField(1, d); // dim_value = 1
                } else {
                    dimProto.writeStringField(2, "N"); // dim_param = 2
                }
                shapeProto.writeBytesField(1, dimProto.toByteArray());
            }
            tensorType.writeBytesField(2, shapeProto.toByteArray());
        }
        ProtoWriter typeProto = new ProtoWriter();
        typeProto.writeBytesField(1, tensorType.toByteArray()); // tensor_type = 1
        w.writeBytesField(2, typeProto.toByteArray());
        return w.toByteArray();
    }

    private static OnnxModel parseModelProto(byte[] bytes) {
        ProtoReader r = new ProtoReader(bytes);
        long irVer = 8;
        long opsetVer = 17;
        String producer = "JSL";
        String gName = "jsl_graph";
        List<ValueInfo> inputs = new ArrayList<>();
        List<ValueInfo> outputs = new ArrayList<>();
        Map<String, TensorData> inits = new LinkedHashMap<>();
        List<Node> nodes = new ArrayList<>();

        while (r.hasMore()) {
            int tag = (int) r.readVarint();
            int field = tag >>> 3;
            int wire = tag & 7;
            switch (field) {
                case 1 -> irVer = r.readVarint();
                case 2 -> producer = r.readString();
                case 7 -> {
                    byte[] graphBytes = r.readBytes();
                    ProtoReader gr = new ProtoReader(graphBytes);
                    while (gr.hasMore()) {
                        int gtag = (int) gr.readVarint();
                        int gfield = gtag >>> 3;
                        int gwire = gtag & 7;
                        switch (gfield) {
                            case 1 -> nodes.add(parseNode(gr.readBytes()));
                            case 2 -> gName = gr.readString();
                            case 5 -> {
                                TensorData td = parseTensor(gr.readBytes());
                                inits.put(td.name(), td);
                            }
                            case 11 -> inputs.add(parseValueInfo(gr.readBytes()));
                            case 12 -> outputs.add(parseValueInfo(gr.readBytes()));
                            default -> gr.skip(gwire);
                        }
                    }
                }
                case 8 -> {
                    ProtoReader or = new ProtoReader(r.readBytes());
                    while (or.hasMore()) {
                        int otag = (int) or.readVarint();
                        int ofield = otag >>> 3;
                        int owire = otag & 7;
                        if (ofield == 2) opsetVer = or.readVarint();
                        else or.skip(owire);
                    }
                }
                default -> r.skip(wire);
            }
        }
        return new OnnxModel(irVer, opsetVer, producer, gName, inputs, outputs, inits, nodes);
    }

    private static Node parseNode(byte[] bytes) {
        ProtoReader r = new ProtoReader(bytes);
        String name = "";
        String opType = "";
        List<String> inputs = new ArrayList<>();
        List<String> outputs = new ArrayList<>();
        Map<String, Attribute> attrs = new LinkedHashMap<>();
        while (r.hasMore()) {
            int tag = (int) r.readVarint();
            int field = tag >>> 3;
            int wire = tag & 7;
            switch (field) {
                case 1 -> inputs.add(r.readString());
                case 2 -> outputs.add(r.readString());
                case 3 -> name = r.readString();
                case 4 -> opType = r.readString();
                case 5 -> {
                    Attribute a = parseAttribute(r.readBytes());
                    attrs.put(a.name(), a);
                }
                default -> r.skip(wire);
            }
        }
        return new Node(name, opType, inputs, outputs, attrs);
    }

    private static Attribute parseAttribute(byte[] bytes) {
        ProtoReader r = new ProtoReader(bytes);
        String name = "";
        int type = 0;
        long intVal = 0;
        float floatVal = 0.0f;
        String strVal = "";
        List<Long> ints = new ArrayList<>();
        List<Float> floats = new ArrayList<>();
        while (r.hasMore()) {
            int tag = (int) r.readVarint();
            int field = tag >>> 3;
            int wire = tag & 7;
            switch (field) {
                case 1 -> name = r.readString();
                case 2 -> floatVal = r.readFloat();
                case 3 -> intVal = r.readVarint();
                case 4 -> strVal = r.readString();
                case 7 -> {
                    if (wire == 2) {
                        byte[] packed = r.readBytes();
                        ByteBuffer bb = ByteBuffer.wrap(packed).order(ByteOrder.LITTLE_ENDIAN);
                        while (bb.remaining() >= 4) floats.add(bb.getFloat());
                    } else {
                        floats.add(r.readFloat());
                    }
                }
                case 8 -> {
                    if (wire == 2) {
                        ProtoReader pr = new ProtoReader(r.readBytes());
                        while (pr.hasMore()) ints.add(pr.readVarint());
                    } else {
                        ints.add(r.readVarint());
                    }
                }
                case 20 -> type = (int) r.readVarint();
                default -> r.skip(wire);
            }
        }
        long[] iArr = ints.stream().mapToLong(Long::longValue).toArray();
        float[] fArr = new float[floats.size()];
        for (int i = 0; i < floats.size(); i++) fArr[i] = floats.get(i);
        return new Attribute(name, type, intVal, floatVal, strVal, iArr, fArr);
    }

    private static TensorData parseTensor(byte[] bytes) {
        ProtoReader r = new ProtoReader(bytes);
        String name = "";
        int dataType = 1;
        List<Integer> dims = new ArrayList<>();
        List<Float> floatVals = new ArrayList<>();
        List<Long> int64Vals = new ArrayList<>();
        byte[] rawData = null;

        while (r.hasMore()) {
            int tag = (int) r.readVarint();
            int field = tag >>> 3;
            int wire = tag & 7;
            switch (field) {
                case 1 -> {
                    if (wire == 2) {
                        ProtoReader pr = new ProtoReader(r.readBytes());
                        while (pr.hasMore()) dims.add((int) pr.readVarint());
                    } else {
                        dims.add((int) r.readVarint());
                    }
                }
                case 2 -> dataType = (int) r.readVarint();
                case 4 -> {
                    if (wire == 2) {
                        byte[] packed = r.readBytes();
                        ByteBuffer bb = ByteBuffer.wrap(packed).order(ByteOrder.LITTLE_ENDIAN);
                        while (bb.remaining() >= 4) floatVals.add(bb.getFloat());
                    } else {
                        floatVals.add(r.readFloat());
                    }
                }
                case 7 -> {
                    if (wire == 2) {
                        ProtoReader pr = new ProtoReader(r.readBytes());
                        while (pr.hasMore()) int64Vals.add(pr.readVarint());
                    } else {
                        int64Vals.add(r.readVarint());
                    }
                }
                case 8 -> name = r.readString();
                case 9 -> rawData = r.readBytes();
                default -> r.skip(wire);
            }
        }
        int[] shape = dims.stream().mapToInt(Integer::intValue).toArray();
        float[] data;
        if (rawData != null && rawData.length > 0) {
            ByteBuffer bb = ByteBuffer.wrap(rawData).order(ByteOrder.LITTLE_ENDIAN);
            if (dataType == 7) { // INT64
                data = new float[rawData.length / 8];
                for (int i = 0; i < data.length; i++) data[i] = (float) bb.getLong();
            } else if (dataType == 11) { // DOUBLE
                data = new float[rawData.length / 8];
                for (int i = 0; i < data.length; i++) data[i] = (float) bb.getDouble();
            } else { // FLOAT
                data = new float[rawData.length / 4];
                for (int i = 0; i < data.length; i++) data[i] = bb.getFloat();
            }
        } else if (!floatVals.isEmpty()) {
            data = new float[floatVals.size()];
            for (int i = 0; i < data.length; i++) data[i] = floatVals.get(i);
        } else if (!int64Vals.isEmpty()) {
            data = new float[int64Vals.size()];
            for (int i = 0; i < data.length; i++) data[i] = int64Vals.get(i);
        } else {
            data = new float[0];
        }
        return new TensorData(name, shape, data);
    }

    private static ValueInfo parseValueInfo(byte[] bytes) {
        ProtoReader r = new ProtoReader(bytes);
        String name = "";
        int elemType = 1;
        List<Integer> dims = new ArrayList<>();
        while (r.hasMore()) {
            int tag = (int) r.readVarint();
            int field = tag >>> 3;
            int wire = tag & 7;
            if (field == 1) {
                name = r.readString();
            } else if (field == 2) {
                ProtoReader tr = new ProtoReader(r.readBytes());
                while (tr.hasMore()) {
                    int ttag = (int) tr.readVarint();
                    int tfield = ttag >>> 3;
                    int twire = ttag & 7;
                    if (tfield == 1) {
                        ProtoReader ttr = new ProtoReader(tr.readBytes());
                        while (ttr.hasMore()) {
                            int tttag = (int) ttr.readVarint();
                            int ttfield = tttag >>> 3;
                            int ttwire = tttag & 7;
                            if (ttfield == 1) {
                                elemType = (int) ttr.readVarint();
                            } else if (ttfield == 2) {
                                ProtoReader sr = new ProtoReader(ttr.readBytes());
                                while (sr.hasMore()) {
                                    int stag = (int) sr.readVarint();
                                    int sfield = stag >>> 3;
                                    int swire = stag & 7;
                                    if (sfield == 1) {
                                        ProtoReader dr = new ProtoReader(sr.readBytes());
                                        int dimVal = -1;
                                        while (dr.hasMore()) {
                                            int dtag = (int) dr.readVarint();
                                            int dfield = dtag >>> 3;
                                            int dwire = dtag & 7;
                                            if (dfield == 1) dimVal = (int) dr.readVarint();
                                            else dr.skip(dwire);
                                        }
                                        dims.add(dimVal);
                                    } else {
                                        sr.skip(swire);
                                    }
                                }
                            } else {
                                ttr.skip(ttwire);
                            }
                        }
                    } else {
                        tr.skip(twire);
                    }
                }
            } else {
                r.skip(wire);
            }
        }
        return new ValueInfo(name, elemType, dims.stream().mapToInt(Integer::intValue).toArray());
    }

    private static final class ProtoWriter {
        private byte[] buf = new byte[256];
        private int pos = 0;

        private void ensure(int add) {
            if (pos + add > buf.length) {
                buf = Arrays.copyOf(buf, Math.max(buf.length * 2, pos + add));
            }
        }

        void writeByte(int b) {
            ensure(1);
            buf[pos++] = (byte) b;
        }

        void writeBytes(byte[] bytes) {
            ensure(bytes.length);
            System.arraycopy(bytes, 0, buf, pos, bytes.length);
            pos += bytes.length;
        }

        void writeVarint(long value) {
            long v = value;
            while ((v & ~0x7FL) != 0L) {
                writeByte((int) ((v & 0x7FL) | 0x80L));
                v >>>= 7;
            }
            writeByte((int) (v & 0x7FL));
        }

        void writeTag(int field, int wireType) {
            writeVarint(((long) field << 3) | wireType);
        }

        void writeVarintField(int field, long value) {
            writeTag(field, 0);
            writeVarint(value);
        }

        void writeFloatField(int field, float value) {
            writeTag(field, 5);
            ensure(4);
            int bits = Float.floatToRawIntBits(value);
            buf[pos++] = (byte) (bits & 0xFF);
            buf[pos++] = (byte) ((bits >>> 8) & 0xFF);
            buf[pos++] = (byte) ((bits >>> 16) & 0xFF);
            buf[pos++] = (byte) ((bits >>> 24) & 0xFF);
        }

        void writeBytesField(int field, byte[] value) {
            writeTag(field, 2);
            writeVarint(value.length);
            writeBytes(value);
        }

        void writeStringField(int field, String value) {
            writeBytesField(field, value.getBytes(StandardCharsets.UTF_8));
        }

        byte[] toByteArray() {
            return Arrays.copyOf(buf, pos);
        }
    }

    private static final class ProtoReader {
        private final byte[] buf;
        private int pos = 0;

        ProtoReader(byte[] buf) {
            this.buf = buf;
        }

        boolean hasMore() {
            return pos < buf.length;
        }

        long readVarint() {
            long result = 0;
            int shift = 0;
            while (pos < buf.length) {
                byte b = buf[pos++];
                result |= (long) (b & 0x7F) << shift;
                if ((b & 0x80) == 0) {
                    return result;
                }
                shift += 7;
            }
            throw new IllegalStateException("Truncated varint in ONNX protobuf");
        }

        float readFloat() {
            int bits = (buf[pos] & 0xFF)
                    | ((buf[pos + 1] & 0xFF) << 8)
                    | ((buf[pos + 2] & 0xFF) << 16)
                    | ((buf[pos + 3] & 0xFF) << 24);
            pos += 4;
            return Float.intBitsToFloat(bits);
        }

        byte[] readBytes() {
            int len = (int) readVarint();
            byte[] out = Arrays.copyOfRange(buf, pos, pos + len);
            pos += len;
            return out;
        }

        String readString() {
            return new String(readBytes(), StandardCharsets.UTF_8);
        }

        void skip(int wire) {
            switch (wire) {
                case 0 -> readVarint();
                case 1 -> pos += 8;
                case 2 -> {
                    int len = (int) readVarint();
                    pos += len;
                }
                case 5 -> pos += 4;
                default -> throw new IllegalStateException("Unknown protobuf wire type " + wire);
            }
        }
    }
}
