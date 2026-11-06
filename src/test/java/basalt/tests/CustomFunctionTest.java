package basalt.tests;

import basalt.Basalt;
import basalt.torch.Backward;
import basalt.torch.Function;
import basalt.torch.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * 自定义算子：手写前向与反向，不改 Rust、不重编原生库。
 *
 * <p>这是「自创算法」的入口。用例分别验证：手写反向与自动微分结果一致；
 * 可以写出自动微分给不出的梯度（梯度裁剪）；多个自定义算子叠加时梯度逐层传下去。</p>
 */
class CustomFunctionTest {

    /** 在张量所在上下文里造一个标量常量。 */
    private static Tensor scalar(Tensor like, float value) {
        return Tensor.of(like.handle().context(), new float[]{value}, false, 1);
    }

    /** y = x²，反向 2x·g。与自动微分对照，验证手写路径算对了。 */
    private static final class Square extends Function {
        @Override
        protected Tensor forward(Tensor... inputs) {
            return inputs[0].mul(inputs[0]);
        }

        @Override
        protected Tensor[] backward(Tensor[] gradOutputs, Tensor[] inputs, Tensor[] outputs) {
            Tensor twoX = inputs[0].mul(scalar(inputs[0], 2f));
            return new Tensor[]{twoX.mul(gradOutputs[0])};
        }
    }

    /** 前向恒等，反向把梯度裁到 [-1, 1]：自动微分给不出的梯度。 */
    private static final class ClippedGradient extends Function {
        @Override
        protected Tensor forward(Tensor... inputs) {
            return inputs[0];
        }

        @Override
        protected Tensor[] backward(Tensor[] gradOutputs, Tensor[] inputs, Tensor[] outputs) {
            float[] raw = gradOutputs[0].data();
            float[] clipped = new float[raw.length];
            for (int index = 0; index < raw.length; index++) {
                clipped[index] = Math.max(-1f, Math.min(1f, raw[index]));
            }
            return new Tensor[]{Tensor.of(inputs[0].handle().context(), clipped, false,
                    inputs[0].shape())};
        }
    }

    @Test
    void manualBackwardMatchesAutograd() {
        try (Basalt engine = Basalt.create()) {
            Tensor automatic = engine.parameter(new float[]{1f, 2f, 3f}, 3);
            automatic.mul(automatic).sumAll().backward();
            float[] fromAutograd = automatic.grad();

            Tensor manual = engine.parameter(new float[]{1f, 2f, 3f}, 3);
            Tensor out = new Square().apply(manual);
            Backward.run(out.sumAll());

            assertArrayEquals(fromAutograd, manual.grad(), 1e-5f);
        }
    }

    @Test
    void customBackwardCanDifferFromAutograd() {
        try (Basalt engine = Basalt.create()) {
            Tensor x = engine.parameter(new float[]{0f, 0f}, 2);
            Tensor out = new ClippedGradient().apply(x);
            // 输出梯度被放大到 100，自定义反向会把它裁到 1
            Tensor scaled = out.mul(engine.tensor(new float[]{100f, 100f}, 2));
            Backward.run(scaled.sumAll());

            float[] grad = x.grad();
            assertEquals(1.0, grad[0], 1e-5, "梯度必须被裁到 1");
            assertEquals(1.0, grad[1], 1e-5);
        }
    }

    @Test
    void stackedCustomFunctionsPropagate() {
        try (Basalt engine = Basalt.create()) {
            Tensor x = engine.parameter(new float[]{1f, 2f, 3f}, 3);
            Tensor first = new Square().apply(x);      // x²
            Tensor second = new Square().apply(first); // x⁴
            Backward.run(second.sumAll());

            // d(sum(x⁴))/dx = 4x³
            float[] grad = x.grad();
            assertEquals(4.0, grad[0], 1e-2);
            assertEquals(32.0, grad[1], 1e-2);
            assertEquals(108.0, grad[2], 1e-2);
        }
    }

    @Test
    void accumulateGradRejectsShapeMismatch() {
        try (Basalt engine = Basalt.create()) {
            Tensor x = engine.tensor(new float[]{1f, 2f}, 2);
            Tensor wrong = engine.tensor(new float[]{1f, 2f, 3f}, 3);
            assertThrows(IllegalArgumentException.class, () -> x.accumulateGrad(wrong));
        }
    }

    @Test
    void accumulateGradSumsRepeatedCalls() {
        try (Basalt engine = Basalt.create()) {
            Tensor x = engine.parameter(new float[]{0f, 0f}, 2);
            x.accumulateGrad(engine.tensor(new float[]{1f, 2f}, 2));
            x.accumulateGrad(engine.tensor(new float[]{10f, 20f}, 2));
            float[] grad = x.grad();
            assertEquals(11.0, grad[0], 1e-6);
            assertEquals(22.0, grad[1], 1e-6);
        }
    }

    @Test
    void plainGraphStillWorksWithoutCustomFunctions() {
        try (Basalt engine = Basalt.create()) {
            Tensor x = engine.parameter(new float[]{3f, 4f}, 2);
            Backward.run(x.mul(x).sumAll());
            float[] grad = x.grad();
            assertEquals(6.0, grad[0], 1e-6);
            assertEquals(8.0, grad[1], 1e-6);
        }
    }
}
