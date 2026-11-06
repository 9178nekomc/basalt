package basalt.torch;

import java.util.List;

/**
 * 把「引擎反向」和「自定义反向」串起来跑完一次。
 *
 * <p>{@link Function} 的前向把输入分离成叶子张量，所以引擎的反向会停在那些叶子上。
 * 这里负责在引擎反向之后，按登记顺序调用每个自定义算子的 {@code backward}，
 * 把梯度累加回真实输入，再对输入继续调用 {@code backward} 往前传——
 * 直到没有新的待处理记录为止。</p>
 *
 * <pre>{@code
 * Tensor out = myFn.apply(x);
 * Tensor loss = out.mse(target);
 * Backward.run(loss);        // 代替 loss.backward()
 * optimizer.step();
 * }</pre>
 *
 * <p>只用内置算子、没有自定义算子时，{@code Backward.run} 等价于直接调用
 * {@code loss.backward()}，不会多跑一轮。</p>
 */
public final class Backward {

    /** 防御性上限：自定义算子链长度不会超过这个数，超出说明登记逻辑出了问题。 */
    private static final int MAX_ROUNDS = 64;

    private Backward() {
    }

    /**
     * 从 {@code loss} 开始反向，直到所有自定义算子的梯度都算完。
     *
     * @throws IllegalStateException 自定义反向没有收敛（登记队列反复非空）
     */
    public static void run(Tensor loss) {
        if (loss == null) {
            throw new IllegalArgumentException("loss must not be null");
        }
        List<Function.Pending> pending = Function.takePending();
        loss.backward();

        int rounds = 0;
        while (!pending.isEmpty()) {
            if (++rounds > MAX_ROUNDS) {
                Function.reset();
                throw new IllegalStateException(
                        "custom backward did not converge after " + MAX_ROUNDS + " rounds");
            }

            // 逆序：后登记的先算，与计算图的反向顺序一致
            for (int index = pending.size() - 1; index >= 0; index--) {
                Function.Pending entry = pending.get(index);
                Tensor[] gradOutputs = Function.gradsOf(entry.outputs);
                Tensor[] gradInputs = entry.function.backward(
                        gradOutputs, entry.inputs, entry.outputs);
                if (gradInputs == null) {
                    continue;
                }
                if (gradInputs.length != entry.inputs.length) {
                    throw new IllegalStateException("backward returned " + gradInputs.length
                            + " gradients for " + entry.inputs.length + " inputs");
                }
                for (int slot = 0; slot < gradInputs.length; slot++) {
                    if (gradInputs[slot] != null) {
                        entry.inputs[slot].accumulateGrad(gradInputs[slot]);
                    }
                }
            }

            // 梯度已经落到真实输入上，继续沿它们的计算图往前传。
            // 叶子输入（参数）不在图里，对它调用 backward 会报错，必须先判断。
            for (Function.Pending entry : pending) {
                for (Tensor input : entry.inputs) {
                    if (input.hasGraphNode()) {
                        input.backward();
                    }
                }
            }
            pending = Function.takePending();
        }
    }
}
