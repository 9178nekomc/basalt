package jsl.torch;

import jsl.core.Layout;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Type-state module pipeline. {@link #add} returns a new {@code Sequential} whose output layout
 * becomes the next module's input layout, so invalid connections fail at compile time.
 */
public final class Sequential<I extends Layout, O extends Layout> extends Module<I, O> {
    private final List<Module<?, ?>> modules;

    private Sequential(List<Module<?, ?>> modules) {
        this.modules = List.copyOf(modules);
    }

    public static <L extends Layout> Sequential<L, L> start() {
        return new Sequential<>(List.of());
    }

    public <N extends Layout> Sequential<I, N> add(Module<O, N> module) {
        List<Module<?, ?>> next = new ArrayList<>(modules);
        next.add(module);
        return new Sequential<>(next);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor current = input;
        for (Module<?, ?> module : modules) {
            current = module.forward(current);
        }
        return current;
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> parameters = new ArrayList<>();
        for (Module<?, ?> module : modules) {
            parameters.addAll(module.parameters());
        }
        return parameters;
    }

    @Override
    public List<Map.Entry<String, Tensor>> namedParameters() {
        List<Map.Entry<String, Tensor>> entries = new ArrayList<>();
        for (int index = 0; index < modules.size(); index++) {
            String prefix = "layer" + index + ".";
            for (Map.Entry<String, Tensor> entry : modules.get(index).namedParameters()) {
                entries.add(Map.entry(prefix + entry.getKey(), entry.getValue()));
            }
        }
        return entries;
    }

    @Override
    public void train() {
        super.train();
        for (Module<?, ?> module : modules) {
            module.train();
        }
    }

    @Override
    public void eval() {
        super.eval();
        for (Module<?, ?> module : modules) {
            module.eval();
        }
    }

    @Override
    public void close() {
        for (Module<?, ?> module : modules) {
            module.close();
        }
    }

    public int depth() {
        return modules.size();
    }
}
