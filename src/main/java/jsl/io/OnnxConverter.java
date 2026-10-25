package jsl.io;

import jsl.ml.Estimator;
import jsl.ml.Lasso;
import jsl.ml.LinearRegression;
import jsl.ml.LinearSVM;
import jsl.ml.LogisticRegression;
import jsl.ml.MinMaxScaler;
import jsl.ml.PCA;
import jsl.ml.Pipeline;
import jsl.ml.Ridge;
import jsl.ml.StandardScaler;
import jsl.ml.Transformer;
import jsl.torch.Activation;
import jsl.torch.BatchNorm2d;
import jsl.torch.ComputationGraph;
import jsl.torch.Conv2d;
import jsl.torch.DepthwiseSeparableConv2d;
import jsl.torch.Dropout;
import jsl.torch.Flatten;
import jsl.torch.LayerNorm;
import jsl.torch.Linear;
import jsl.torch.MaxPool2d;
import jsl.torch.Module;
import jsl.torch.Sequential;
import jsl.torch.Tensor;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * High-level ONNX model import and export facade.
 *
 * <p>Supports:</p>
 * <ul>
 *   <li>Loading standard `.onnx` files into {@link OnnxModel} for native JVM execution.</li>
 *   <li>Exporting JSL classical models ({@link LinearRegression}, {@link Ridge}, {@link Lasso},
 *       {@link LogisticRegression}, {@link LinearSVM}, {@link PCA}, {@link StandardScaler},
 *       {@link MinMaxScaler}, {@link Pipeline}) and deep learning modules ({@link Linear},
 *       {@link Conv2d}, {@link DepthwiseSeparableConv2d}, {@link Sequential},
 *       {@link ComputationGraph}) to standard `.onnx` files (IR v8, Opset 17).</li>
 * </ul>
 */
public final class OnnxConverter {
    private static final long IR_VERSION = 8L;
    private static final long OPSET_VERSION = 17L;

    private OnnxConverter() {
    }

    public static OnnxModel load(Path path) {
        return OnnxModel.load(path);
    }

    public static void export(Estimator estimator, int inputFeatures, Path target) {
        toOnnx(estimator, inputFeatures).save(target);
    }

    public static void export(Module<?, ?> module, int[] inputShape, Path target) {
        toOnnx(module, inputShape).save(target);
    }

    public static OnnxModel toOnnx(Estimator estimator, int inputFeatures) {
        Objects.requireNonNull(estimator, "estimator");
        if (inputFeatures <= 0) throw new IllegalArgumentException("inputFeatures must be positive");
        Map<String, OnnxModel.TensorData> inits = new LinkedHashMap<>();
        List<OnnxModel.Node> nodes = new ArrayList<>();
        String current = "input";
        int currentDim = inputFeatures;
        int idx = 0;

        if (estimator instanceof Pipeline pipeline) {
            for (Pipeline.Step step : pipeline.steps()) {
                StepResult sr = appendTransformer(step.transformer(), current, currentDim, "step" + (idx++), inits, nodes);
                current = sr.outputName;
                currentDim = sr.outputDim;
            }
            if (pipeline.estimator() != null) {
                StepResult er = appendEstimator(pipeline.estimator(), current, currentDim, "est", inits, nodes);
                current = er.outputName;
                currentDim = er.outputDim;
            }
        } else if (estimator instanceof Transformer transformer) {
            StepResult sr = appendTransformer(transformer, current, currentDim, "tr0", inits, nodes);
            current = sr.outputName;
            currentDim = sr.outputDim;
        } else {
            StepResult er = appendEstimator(estimator, current, currentDim, "est0", inits, nodes);
            current = er.outputName;
            currentDim = er.outputDim;
        }

        List<OnnxModel.ValueInfo> inputs = List.of(
                new OnnxModel.ValueInfo("input", 1, new int[]{-1, inputFeatures}));
        List<OnnxModel.ValueInfo> outputs = List.of(
                new OnnxModel.ValueInfo(current, 1, new int[]{-1, currentDim}));
        return new OnnxModel(IR_VERSION, OPSET_VERSION, "JSL", "jsl_ml_graph",
                inputs, outputs, inits, nodes);
    }

    public static OnnxModel toOnnx(Module<?, ?> module, int[] inputShape) {
        Objects.requireNonNull(module, "module");
        Objects.requireNonNull(inputShape, "inputShape");
        Map<String, OnnxModel.TensorData> inits = new LinkedHashMap<>();
        List<OnnxModel.Node> nodes = new ArrayList<>();

        if (module instanceof ComputationGraph cg) {
            for (String vName : cg.executionOrder()) {
                ComputationGraph.Vertex v = cg.vertices().get(vName);
                switch (v.op()) {
                    case MODULE -> appendModule(v.module(), v.inputs().get(0), vName, vName, inits, nodes);
                    case ADD -> nodes.add(new OnnxModel.Node(vName, "Add",
                            List.of(v.inputs().get(0), v.inputs().get(1)), List.of(vName), Map.of()));
                    case SUB -> nodes.add(new OnnxModel.Node(vName, "Sub",
                            List.of(v.inputs().get(0), v.inputs().get(1)), List.of(vName), Map.of()));
                    case MUL -> nodes.add(new OnnxModel.Node(vName, "Mul",
                            List.of(v.inputs().get(0), v.inputs().get(1)), List.of(vName), Map.of()));
                }
            }
            List<OnnxModel.ValueInfo> ins = new ArrayList<>();
            for (String inName : cg.inputNames()) {
                ins.add(new OnnxModel.ValueInfo(inName, 1, inputShape));
            }
            List<OnnxModel.ValueInfo> outs = new ArrayList<>();
            for (String outName : cg.outputNames()) {
                outs.add(new OnnxModel.ValueInfo(outName, 1, null));
            }
            return new OnnxModel(IR_VERSION, OPSET_VERSION, "JSL", "jsl_dag_graph",
                    ins, outs, inits, nodes);
        }

        String current = appendModule(module, "input", "output", "m0", inits, nodes);
        List<OnnxModel.ValueInfo> ins = List.of(new OnnxModel.ValueInfo("input", 1, inputShape));
        List<OnnxModel.ValueInfo> outs = List.of(new OnnxModel.ValueInfo(current, 1, null));
        return new OnnxModel(IR_VERSION, OPSET_VERSION, "JSL", "jsl_torch_graph",
                ins, outs, inits, nodes);
    }

    private static String appendModule(Module<?, ?> module, String inName, String outName,
                                       String prefix, Map<String, OnnxModel.TensorData> inits,
                                       List<OnnxModel.Node> nodes) {
        if (module instanceof Sequential<?, ?> seq) {
            String cur = inName;
            List<Module<?, ?>> mods = seq.modules();
            for (int i = 0; i < mods.size(); i++) {
                String nextOut = (i == mods.size() - 1) ? outName : (prefix + "_s" + i);
                cur = appendModule(mods.get(i), cur, nextOut, prefix + "_l" + i, inits, nodes);
            }
            return cur;
        }
        if (module instanceof Linear linear) {
            Tensor w = linear.weight();
            String wName = prefix + "_W";
            inits.put(wName, new OnnxModel.TensorData(wName, w.shape(), w.data()));
            Tensor b = linear.bias();
            if (b != null) {
                String mmOut = prefix + "_mm";
                nodes.add(new OnnxModel.Node(prefix + "_matmul", "MatMul",
                        List.of(inName, wName), List.of(mmOut), Map.of()));
                String bName = prefix + "_b";
                inits.put(bName, new OnnxModel.TensorData(bName, b.shape(), b.data()));
                nodes.add(new OnnxModel.Node(prefix + "_add", "Add",
                        List.of(mmOut, bName), List.of(outName), Map.of()));
            } else {
                nodes.add(new OnnxModel.Node(prefix + "_matmul", "MatMul",
                        List.of(inName, wName), List.of(outName), Map.of()));
            }
            return outName;
        }
        if (module instanceof Conv2d conv) {
            Tensor w = conv.weight();
            String wName = prefix + "_W";
            inits.put(wName, new OnnxModel.TensorData(wName, w.shape(), w.data()));
            List<String> convInputs = new ArrayList<>();
            convInputs.add(inName);
            convInputs.add(wName);
            if (conv.bias() != null) {
                String bName = prefix + "_B";
                inits.put(bName, new OnnxModel.TensorData(bName,
                        new int[]{conv.outputChannels()}, conv.bias().data()));
                convInputs.add(bName);
            }
            Map<String, OnnxModel.Attribute> attrs = new LinkedHashMap<>();
            attrs.put("kernel_shape", OnnxModel.Attribute.ofInts("kernel_shape",
                    conv.kernelHeight(), conv.kernelWidth()));
            attrs.put("strides", OnnxModel.Attribute.ofInts("strides",
                    conv.strideHeight(), conv.strideWidth()));
            attrs.put("pads", OnnxModel.Attribute.ofInts("pads",
                    conv.paddingHeight(), conv.paddingWidth(),
                    conv.paddingHeight(), conv.paddingWidth()));
            attrs.put("dilations", OnnxModel.Attribute.ofInts("dilations",
                    conv.dilationHeight(), conv.dilationWidth()));
            attrs.put("group", OnnxModel.Attribute.ofInt("group", conv.groups()));
            nodes.add(new OnnxModel.Node(prefix + "_conv", "Conv", convInputs, List.of(outName), attrs));
            return outName;
        }
        if (module instanceof DepthwiseSeparableConv2d ds) {
            String mid = prefix + "_dw";
            appendModule(ds.depthwise(), inName, mid, prefix + "_dw", inits, nodes);
            return appendModule(ds.pointwise(), mid, outName, prefix + "_pw", inits, nodes);
        }
        if (module instanceof Activation<?> act) {
            Map<String, OnnxModel.Attribute> attrs = new LinkedHashMap<>();
            String op = switch (act.kind()) {
                case RELU -> "Relu";
                case SIGMOID -> "Sigmoid";
                case TANH -> "Tanh";
                case GELU -> "Gelu";
                case SILU -> "Silu";
                case LEAKY_RELU -> {
                    attrs.put("alpha", OnnxModel.Attribute.ofFloat("alpha", act.alpha()));
                    yield "LeakyRelu";
                }
                case ELU -> {
                    attrs.put("alpha", OnnxModel.Attribute.ofFloat("alpha", act.alpha()));
                    yield "Elu";
                }
            };
            nodes.add(new OnnxModel.Node(prefix + "_" + op, op, List.of(inName), List.of(outName), attrs));
            return outName;
        }
        if (module instanceof MaxPool2d mp) {
            Map<String, OnnxModel.Attribute> attrs = new LinkedHashMap<>();
            attrs.put("kernel_shape", OnnxModel.Attribute.ofInts("kernel_shape",
                    mp.kernelHeight(), mp.kernelWidth()));
            attrs.put("strides", OnnxModel.Attribute.ofInts("strides",
                    mp.strideHeight(), mp.strideWidth()));
            nodes.add(new OnnxModel.Node(prefix + "_maxpool", "MaxPool",
                    List.of(inName), List.of(outName), attrs));
            return outName;
        }
        if (module instanceof Flatten) {
            Map<String, OnnxModel.Attribute> attrs = Map.of(
                    "axis", OnnxModel.Attribute.ofInt("axis", 1L));
            nodes.add(new OnnxModel.Node(prefix + "_flatten", "Flatten",
                    List.of(inName), List.of(outName), attrs));
            return outName;
        }
        if (module instanceof BatchNorm2d bn) {
            String sName = prefix + "_scale";
            String bName = prefix + "_B";
            String mName = prefix + "_mean";
            String vName = prefix + "_var";
            inits.put(sName, new OnnxModel.TensorData(sName, bn.gamma().shape(), bn.gamma().data()));
            inits.put(bName, new OnnxModel.TensorData(bName, bn.beta().shape(), bn.beta().data()));
            inits.put(mName, new OnnxModel.TensorData(mName, bn.runningMean().shape(), bn.runningMean().data()));
            inits.put(vName, new OnnxModel.TensorData(vName, bn.runningVar().shape(), bn.runningVar().data()));
            Map<String, OnnxModel.Attribute> attrs = Map.of(
                    "epsilon", OnnxModel.Attribute.ofFloat("epsilon", bn.epsilon()));
            nodes.add(new OnnxModel.Node(prefix + "_bn", "BatchNormalization",
                    List.of(inName, sName, bName, mName, vName), List.of(outName), attrs));
            return outName;
        }
        if (module instanceof LayerNorm<?> ln) {
            String sName = prefix + "_scale";
            String bName = prefix + "_B";
            inits.put(sName, new OnnxModel.TensorData(sName, ln.gamma().shape(), ln.gamma().data()));
            inits.put(bName, new OnnxModel.TensorData(bName, ln.beta().shape(), ln.beta().data()));
            Map<String, OnnxModel.Attribute> attrs = Map.of(
                    "epsilon", OnnxModel.Attribute.ofFloat("epsilon", ln.epsilon()));
            nodes.add(new OnnxModel.Node(prefix + "_ln", "LayerNormalization",
                    List.of(inName, sName, bName), List.of(outName), attrs));
            return outName;
        }
        if (module instanceof Dropout<?>) {
            nodes.add(new OnnxModel.Node(prefix + "_id", "Identity",
                    List.of(inName), List.of(outName), Map.of()));
            return outName;
        }
        throw new jsl.JSLUnsupportedOperationException(
                "ONNX export does not yet support module type: " + module.getClass().getSimpleName());
    }

    private static StepResult appendTransformer(Transformer tr, String inName, int inDim,
                                                String prefix, Map<String, OnnxModel.TensorData> inits,
                                                List<OnnxModel.Node> nodes) {
        if (tr instanceof StandardScaler scaler) {
            double[] mean = scaler.mean();
            double[] scale = scaler.scale();
            float[] fMean = toFloat(mean);
            float[] fInvScale = new float[scale.length];
            for (int i = 0; i < scale.length; i++) {
                fInvScale[i] = scaler.withStd() ? (float) (1.0 / scale[i]) : 1.0f;
            }
            String cur = inName;
            if (scaler.withMean()) {
                String mName = prefix + "_mean";
                String subOut = prefix + "_centered";
                inits.put(mName, new OnnxModel.TensorData(mName, new int[]{1, inDim}, fMean));
                nodes.add(new OnnxModel.Node(prefix + "_sub", "Sub",
                        List.of(cur, mName), List.of(subOut), Map.of()));
                cur = subOut;
            }
            String sName = prefix + "_inv_scale";
            String outName = prefix + "_scaled";
            inits.put(sName, new OnnxModel.TensorData(sName, new int[]{1, inDim}, fInvScale));
            nodes.add(new OnnxModel.Node(prefix + "_mul", "Mul",
                    List.of(cur, sName), List.of(outName), Map.of()));
            return new StepResult(outName, inDim);
        }
        if (tr instanceof MinMaxScaler mms) {
            float[] fScale = toFloat(mms.scale());
            double[] dMin = mms.dataMin();
            double[] dScale = mms.scale();
            // transform(dMin) gives minOffset
            jsl.core.Matrix z = mms.transform(jsl.core.Matrix.zeros(1, inDim));
            float[] fOffset = toFloat(z.data());
            String sName = prefix + "_scale";
            String oName = prefix + "_offset";
            String mulOut = prefix + "_mul";
            String outName = prefix + "_out";
            inits.put(sName, new OnnxModel.TensorData(sName, new int[]{1, inDim}, fScale));
            inits.put(oName, new OnnxModel.TensorData(oName, new int[]{1, inDim}, fOffset));
            nodes.add(new OnnxModel.Node(prefix + "_mul", "Mul", List.of(inName, sName), List.of(mulOut), Map.of()));
            nodes.add(new OnnxModel.Node(prefix + "_add", "Add", List.of(mulOut, oName), List.of(outName), Map.of()));
            return new StepResult(outName, inDim);
        }
        if (tr instanceof PCA pca) {
            float[] fMean = toFloat(pca.means());
            jsl.core.Matrix comps = pca.components(); // [k, inDim]
            int k = comps.rows();
            // Transpose components to [inDim, k]
            jsl.core.Matrix compsT = comps.transpose();
            float[] fCompsT = toFloat(compsT.data());
            String mName = prefix + "_pca_mean";
            String wName = prefix + "_pca_W";
            String centered = prefix + "_pca_centered";
            String outName = prefix + "_pca_out";
            inits.put(mName, new OnnxModel.TensorData(mName, new int[]{1, inDim}, fMean));
            inits.put(wName, new OnnxModel.TensorData(wName, new int[]{inDim, k}, fCompsT));
            nodes.add(new OnnxModel.Node(prefix + "_sub", "Sub", List.of(inName, mName), List.of(centered), Map.of()));
            nodes.add(new OnnxModel.Node(prefix + "_mm", "MatMul", List.of(centered, wName), List.of(outName), Map.of()));
            return new StepResult(outName, k);
        }
        throw new jsl.JSLUnsupportedOperationException(
                "ONNX export does not support transformer: " + tr.getClass().getSimpleName());
    }

    private static StepResult appendEstimator(Estimator est, String inName, int inDim,
                                              String prefix, Map<String, OnnxModel.TensorData> inits,
                                              List<OnnxModel.Node> nodes) {
        if (est instanceof Ridge ridge) {
            return appendLinearModel(ridge.weights(), ridge.intercept(), inName, inDim, prefix, inits, nodes);
        }
        if (est instanceof Lasso lasso) {
            return appendLinearModel(lasso.weights(), lasso.intercept(), inName, inDim, prefix, inits, nodes);
        }
        if (est instanceof LinearRegression lr) {
            return appendLinearModel(lr.weights(), lr.intercept(), inName, inDim, prefix, inits, nodes);
        }
        if (est instanceof LogisticRegression logr) {
            StepResult logit = appendLinearModel(logr.weights(), logr.intercept(),
                    inName, inDim, prefix + "_lin", inits, nodes);
            String probOut = prefix + "_prob";
            nodes.add(new OnnxModel.Node(prefix + "_sig", "Sigmoid",
                    List.of(logit.outputName), List.of(probOut), Map.of()));
            return new StepResult(probOut, 1);
        }
        if (est instanceof LinearSVM svm) {
            return appendLinearModel(svm.weights(), svm.bias(), inName, inDim, prefix, inits, nodes);
        }
        if (est instanceof PCA pca) {
            return appendTransformer(pca, inName, inDim, prefix, inits, nodes);
        }
        throw new jsl.JSLUnsupportedOperationException(
                "ONNX export does not support estimator: " + est.getClass().getSimpleName());
    }

    private static StepResult appendLinearModel(double[] weights, double intercept, String inName,
                                                int inDim, String prefix,
                                                Map<String, OnnxModel.TensorData> inits,
                                                List<OnnxModel.Node> nodes) {
        float[] fW = toFloat(weights);
        float[] fB = new float[]{(float) intercept};
        String wName = prefix + "_W";
        String bName = prefix + "_b";
        String mmOut = prefix + "_mm";
        String outName = prefix + "_out";
        inits.put(wName, new OnnxModel.TensorData(wName, new int[]{inDim, 1}, fW));
        inits.put(bName, new OnnxModel.TensorData(bName, new int[]{1, 1}, fB));
        nodes.add(new OnnxModel.Node(prefix + "_matmul", "MatMul",
                List.of(inName, wName), List.of(mmOut), Map.of()));
        nodes.add(new OnnxModel.Node(prefix + "_add", "Add",
                List.of(mmOut, bName), List.of(outName), Map.of()));
        return new StepResult(outName, 1);
    }

    private static float[] toFloat(double[] arr) {
        float[] out = new float[arr.length];
        for (int i = 0; i < arr.length; i++) out[i] = (float) arr[i];
        return out;
    }

    private record StepResult(String outputName, int outputDim) {
    }
}
