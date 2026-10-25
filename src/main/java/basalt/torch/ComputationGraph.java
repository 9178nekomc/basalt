package basalt.torch;

import basalt.BasaltRuntimeException;
import basalt.core.Layout;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Queue;
import java.util.Set;

/**
 * Directed Acyclic Graph (DAG) neural network architecture (inspired by DL4J's
 * {@code ComputationGraph}), breaking through the linear limitation of {@link Sequential}.
 *
 * <p>Supports arbitrary branching, residual skip connections ({@link VertexOp#ADD}), multiplicative
 * gating ({@link VertexOp#MUL}), and multi-input / multi-output models while preserving full
 * reverse-mode autograd, {@link #train()}/{@link #eval()} mode propagation, and
 * {@link #stateDict()}/{@link #loadStateDict(Map)} persistence.</p>
 */
public final class ComputationGraph extends Module<Layout, Layout> {
    public enum VertexOp {
        MODULE,
        ADD,
        SUB,
        MUL
    }

    public record Vertex(String name, VertexOp op, Module<?, ?> module, List<String> inputs) {
    }

    private final List<String> inputNames;
    private final List<String> outputNames;
    private final Map<String, Vertex> vertices;
    private final List<String> executionOrder;

    private ComputationGraph(List<String> inputNames, List<String> outputNames,
                             Map<String, Vertex> vertices, List<String> executionOrder) {
        this.inputNames = List.copyOf(inputNames);
        this.outputNames = List.copyOf(outputNames);
        this.vertices = Collections.unmodifiableMap(new LinkedHashMap<>(vertices));
        this.executionOrder = List.copyOf(executionOrder);
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Tensor forward(Tensor input) {
        if (inputNames.size() != 1) {
            throw new IllegalStateException(
                    "Single-tensor forward requires 1 graph input, but graph has " + inputNames);
        }
        Map<String, Tensor> outs = forward(Map.of(inputNames.get(0), input));
        return outs.get(outputNames.get(0));
    }

    public Tensor[] forwardMany(Tensor... inputs) {
        if (inputs.length != inputNames.size()) {
            throw new IllegalArgumentException(
                    "Expected " + inputNames.size() + " inputs " + inputNames
                            + ", received " + inputs.length);
        }
        Map<String, Tensor> feed = new LinkedHashMap<>();
        for (int i = 0; i < inputNames.size(); i++) {
            feed.put(inputNames.get(i), inputs[i]);
        }
        Map<String, Tensor> outs = forward(feed);
        Tensor[] result = new Tensor[outputNames.size()];
        for (int i = 0; i < outputNames.size(); i++) {
            result[i] = outs.get(outputNames.get(i));
        }
        return result;
    }

    public Map<String, Tensor> forward(Map<String, Tensor> inputs) {
        BasaltRuntimeException.requireNonNull(inputs, "inputs");
        Map<String, Tensor> activations = new HashMap<>();
        for (String inName : inputNames) {
            Tensor t = inputs.get(inName);
            if (t == null) {
                throw new IllegalArgumentException("Missing required graph input: " + inName);
            }
            activations.put(inName, t);
        }

        for (String vertexName : executionOrder) {
            Vertex v = vertices.get(vertexName);
            switch (v.op()) {
                case MODULE -> {
                    Tensor in = activations.get(v.inputs().get(0));
                    activations.put(vertexName, v.module().forward(in));
                }
                case ADD -> {
                    Tensor acc = activations.get(v.inputs().get(0));
                    for (int i = 1; i < v.inputs().size(); i++) {
                        acc = acc.add(activations.get(v.inputs().get(i)));
                    }
                    activations.put(vertexName, acc);
                }
                case SUB -> {
                    Tensor left = activations.get(v.inputs().get(0));
                    Tensor right = activations.get(v.inputs().get(1));
                    activations.put(vertexName, left.sub(right));
                }
                case MUL -> {
                    Tensor acc = activations.get(v.inputs().get(0));
                    for (int i = 1; i < v.inputs().size(); i++) {
                        acc = acc.mul(activations.get(v.inputs().get(i)));
                    }
                    activations.put(vertexName, acc);
                }
            }
        }

        Map<String, Tensor> result = new LinkedHashMap<>();
        for (String outName : outputNames) {
            result.put(outName, activations.get(outName));
        }
        return result;
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> params = new ArrayList<>();
        for (String name : executionOrder) {
            Vertex v = vertices.get(name);
            if (v.module() != null) {
                params.addAll(v.module().parameters());
            }
        }
        return params;
    }

    @Override
    public List<Map.Entry<String, Tensor>> namedParameters() {
        List<Map.Entry<String, Tensor>> entries = new ArrayList<>();
        for (String name : executionOrder) {
            Vertex v = vertices.get(name);
            if (v.module() != null) {
                for (Map.Entry<String, Tensor> entry : v.module().namedParameters()) {
                    entries.add(Map.entry(name + "." + entry.getKey(), entry.getValue()));
                }
            }
        }
        return entries;
    }

    @Override
    public void train() {
        super.train();
        for (Vertex v : vertices.values()) {
            if (v.module() != null) v.module().train();
        }
    }

    @Override
    public void eval() {
        super.eval();
        for (Vertex v : vertices.values()) {
            if (v.module() != null) v.module().eval();
        }
    }

    @Override
    public void close() {
        for (Vertex v : vertices.values()) {
            if (v.module() != null) v.module().close();
        }
    }

    public List<String> inputNames() {
        return inputNames;
    }

    public List<String> outputNames() {
        return outputNames;
    }

    public List<String> executionOrder() {
        return executionOrder;
    }

    public Map<String, Vertex> vertices() {
        return vertices;
    }

    public static final class Builder {
        private final List<String> inputNames = new ArrayList<>();
        private final List<String> outputNames = new ArrayList<>();
        private final Map<String, Vertex> vertices = new LinkedHashMap<>();

        public Builder addInputs(String... names) {
            for (String name : names) {
                Objects.requireNonNull(name, "input name");
                if (inputNames.contains(name) || vertices.containsKey(name)) {
                    throw new IllegalArgumentException("Duplicate graph node name: " + name);
                }
                inputNames.add(name);
            }
            return this;
        }

        public Builder addLayer(String name, Module<?, ?> module, String input) {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(module, "module");
            Objects.requireNonNull(input, "input");
            checkUnique(name);
            vertices.put(name, new Vertex(name, VertexOp.MODULE, module, List.of(input)));
            return this;
        }

        public Builder addAddVertex(String name, String... inputs) {
            if (inputs == null || inputs.length < 2) {
                throw new IllegalArgumentException("addAddVertex requires at least 2 inputs");
            }
            checkUnique(name);
            vertices.put(name, new Vertex(name, VertexOp.ADD, null, List.of(inputs)));
            return this;
        }

        public Builder addSubVertex(String name, String leftInput, String rightInput) {
            checkUnique(name);
            vertices.put(name, new Vertex(name, VertexOp.SUB, null, List.of(leftInput, rightInput)));
            return this;
        }

        public Builder addMulVertex(String name, String... inputs) {
            if (inputs == null || inputs.length < 2) {
                throw new IllegalArgumentException("addMulVertex requires at least 2 inputs");
            }
            checkUnique(name);
            vertices.put(name, new Vertex(name, VertexOp.MUL, null, List.of(inputs)));
            return this;
        }

        public Builder setOutputs(String... names) {
            outputNames.clear();
            for (String name : names) {
                Objects.requireNonNull(name, "output name");
                outputNames.add(name);
            }
            return this;
        }

        private void checkUnique(String name) {
            if (inputNames.contains(name) || vertices.containsKey(name)) {
                throw new IllegalArgumentException("Duplicate vertex name: " + name);
            }
        }

        public ComputationGraph build() {
            if (inputNames.isEmpty()) {
                throw new IllegalArgumentException("ComputationGraph requires at least one input");
            }
            if (outputNames.isEmpty()) {
                throw new IllegalArgumentException("ComputationGraph requires at least one output");
            }
            Set<String> known = new HashSet<>(inputNames);
            known.addAll(vertices.keySet());
            for (String out : outputNames) {
                if (!known.contains(out)) {
                    throw new IllegalArgumentException("Unknown output node: " + out);
                }
            }

            Map<String, Integer> indegree = new LinkedHashMap<>();
            Map<String, List<String>> consumers = new HashMap<>();
            for (Vertex v : vertices.values()) {
                indegree.put(v.name(), v.inputs().size());
                for (String in : v.inputs()) {
                    if (!known.contains(in)) {
                        throw new IllegalArgumentException(
                                "Vertex " + v.name() + " references unknown input " + in);
                    }
                    consumers.computeIfAbsent(in, k -> new ArrayList<>()).add(v.name());
                }
            }

            Queue<String> queue = new ArrayDeque<>();
            for (String in : inputNames) {
                for (String consumer : consumers.getOrDefault(in, List.of())) {
                    int deg = indegree.merge(consumer, -1, Integer::sum);
                    if (deg == 0) {
                        queue.add(consumer);
                    }
                }
            }

            List<String> order = new ArrayList<>(vertices.size());
            while (!queue.isEmpty()) {
                String current = queue.poll();
                order.add(current);
                for (String consumer : consumers.getOrDefault(current, List.of())) {
                    int deg = indegree.merge(consumer, -1, Integer::sum);
                    if (deg == 0) {
                        queue.add(consumer);
                    }
                }
            }

            if (order.size() != vertices.size()) {
                throw new IllegalArgumentException("Cycle detected in ComputationGraph definition");
            }
            return new ComputationGraph(inputNames, outputNames, vertices, order);
        }
    }
}
