package basalt.torch;

import basalt.BasaltRuntimeException;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;

/**
 * 用户自定义算子：自己写前向，也自己写反向。
 *
 * <p>库内置的算子覆盖不到的地方——直通估计（量化、二值网络）、自定义激活的解析梯度、
 * 强化学习里的 surrogate gradient、梯度裁剪在反向内部的变体——都可以用这个类表达，
 * 不需要改 Rust、不需要重新编译原生库。</p>
 *
 * <h2>协议</h2>
 *
 * <p>{@link #apply} 会先把输入 {@link Tensor#detach() 分离} 成新的叶子张量，再交给
 * {@link #forward}；前向结果也会被包成一个新叶子返回。因此引擎的反向只负责把
 * 损失梯度送到这个叶子，绝不会穿过本算子继续往回算——整段反向都由你的
 * {@link #backward} 决定。</p>
 *
 * <p>这也意味着：{@code forward} 里用到的算子不会参与自动微分，你必须自己在
 * {@code backward} 里给出输入梯度。这正是自定义算子的语义。</p>
 *
 * <p>一次完整的训练步是：</p>
 *
 * <pre>{@code
 * Tensor out = myFn.apply(x);      // 前向，同时登记一条待处理记录
 * Tensor loss = out.mse(target);
 * Backward.run(loss);              // 引擎反向 + 你的自定义反向，按顺序跑完
 * }</pre>
 *
 * <h2>边界</h2>
 *
 * <p>自定义反向在引擎反向之后运行，因此它处理的是「引擎算到分离张量为止」的梯度。
 * 多层自定义算子叠加时，{@link Backward#run} 会按登记顺序逐层推进，直到没有待处理记录。
 * 这与 PyTorch 的 {@code autograd.Function} 语义一致，区别在于引擎侧没有回调——
 * 自定义反向由 Java 侧驱动，而不是在原生反向循环里被调用。</p>
 */
public abstract class Function {

    /** 一条待处理记录：谁调用了、用了哪些输入、产生了哪些输出。 */
    static final class Pending {
        final Function function;
        final Tensor[] inputs;
        final Tensor[] detached;
        final Tensor[] outputs;

        Pending(Function function, Tensor[] inputs, Tensor[] detached, Tensor[] outputs) {
            this.function = function;
            this.inputs = inputs;
            this.detached = detached;
            this.outputs = outputs;
        }
    }

    /** 全局待处理队列，按登记顺序。跨线程不安全，一个 Context 单线程训练。 */
    private static final Deque<Pending> PENDING = new ArrayDeque<>();

    static List<Pending> takePending() {
        List<Pending> drained = new ArrayList<>(PENDING);
        PENDING.clear();
        return drained;
    }

    static boolean hasPending() {
        return !PENDING.isEmpty();
    }

    static void reset() {
        PENDING.clear();
    }

    /**
     * 前向：输入是分离后的叶子张量，可以用任何公开算子组合。
     */
    protected abstract Tensor forward(Tensor... inputs);

    /**
     * 反向：给定输出梯度，返回每个输入的梯度（顺序与 {@link #forward} 的输入一致）。
     *
     * <p>返回 {@code null} 表示该输入不需要梯度。梯度必须是新张量，不要原地改输入。</p>
     */
    protected abstract Tensor[] backward(Tensor[] gradOutputs, Tensor[] inputs, Tensor[] outputs);

    /**
     * 应用本算子：分离输入 → 跑前向 → 登记待处理记录。
     *
     * @return 前向的输出张量
     */
    public final Tensor apply(Tensor... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("Function.apply needs at least one input");
        }
        for (Tensor input : inputs) {
            BasaltRuntimeException.requireNonNull(input, "input");
        }
        Tensor[] detached = new Tensor[inputs.length];
        for (int index = 0; index < inputs.length; index++) {
            detached[index] = inputs[index].detach();
        }
        Tensor output = forward(detached);
        BasaltRuntimeException.requireNonNull(output, "forward must return a tensor");

        // 引擎只往叶子张量累加梯度，中间张量的 grad() 永远是空的。
        // 所以把前向结果再包一层新叶子：反向从这里拿到 dL/d输出，
        // 内部那段计算图则完全交给用户自己的 backward 处理。
        Tensor leaf = Tensor.of(output.handle().context(), output.data(), true, output.shape());
        PENDING.addLast(new Pending(this, inputs.clone(), detached, new Tensor[]{leaf}));
        return leaf;
    }

    /** 取分离张量的梯度，转成张量返回；没有梯度时给零。 */
    static Tensor[] gradsOf(Tensor[] detached) {
        Tensor[] grads = new Tensor[detached.length];
        for (int index = 0; index < detached.length; index++) {
            Tensor source = detached[index];
            grads[index] = Tensor.of(source.handle().context(), source.grad(), false, source.shape());
        }
        return grads;
    }

    /** 供子类做形状检查，避免把错误留到原生层。 */
    protected static void requireSameShape(Tensor expected, Tensor actual, String what) {
        if (!Arrays.equals(expected.shape(), actual.shape())) {
            throw new IllegalArgumentException(what + " shape " + Arrays.toString(actual.shape())
                    + " does not match " + Arrays.toString(expected.shape()));
        }
    }
}
