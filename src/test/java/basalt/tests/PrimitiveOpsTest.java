package basalt.tests;

import basalt.Basalt;
import basalt.torch.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 基础算子：前向数值、广播、以及自动微分是否真的接到了反向。
 *
 * <p>Rust 侧已经单测过算子数学；这里测的是另一件事——{@link Tensor} 的调用链
 * 有没有把新算子登记进计算图。所以每个可导算子都做中心差分梯度检验。</p>
 */
class PrimitiveOpsTest {

    /** 中心差分：用数值梯度检验解析梯度。 */
    private static void checkGradient(Basalt engine, double[] values, java.util.function.Function<Tensor, Tensor> loss) {
        Tensor input = engine.parameter(toFloats(values), values.length);
        Tensor output = loss.apply(input);
        Tensor scalar = output.sumAll();
        scalar.backward();
        float[] analytic = input.grad();

        double epsilon = 1e-3;
        for (int index = 0; index < values.length; index++) {
            double[] plus = values.clone();
            plus[index] += epsilon;
            double[] minus = values.clone();
            minus[index] -= epsilon;
            double forward = evaluate(engine, plus, loss);
            double backward = evaluate(engine, minus, loss);
            double numeric = (forward - backward) / (2 * epsilon);
            double relative = Math.abs(numeric - analytic[index])
                    / Math.max(1.0, Math.abs(numeric) + Math.abs(analytic[index]));
            assertTrue(relative < 2e-2, "gradient mismatch at " + index + ": analytic="
                    + analytic[index] + " numeric=" + numeric + " relative=" + relative);
        }
    }

    private static double evaluate(Basalt engine, double[] values,
                                   java.util.function.Function<Tensor, Tensor> loss) {
        Tensor input = engine.tensor(toFloats(values), values.length);
        return loss.apply(input).sumAll().data()[0];
    }

    private static float[] toFloats(double[] values) {
        float[] out = new float[values.length];
        for (int index = 0; index < values.length; index++) {
            out[index] = (float) values[index];
        }
        return out;
    }

    @Test
    void expLogSqrtForwardValues() {
        try (Basalt engine = Basalt.create()) {
            Tensor x = engine.tensor(new float[]{0f, 1f, 4f, 9f}, 4);
            float[] exp = x.exp().data();
            assertEquals(1.0, exp[0], 1e-6);
            assertEquals(Math.E, exp[1], 1e-6);

            float[] log = engine.tensor(new float[]{1f, (float) Math.E}, 2).log().data();
            assertEquals(0.0, log[0], 1e-6);
            assertEquals(1.0, log[1], 1e-6);

            float[] sqrt = engine.tensor(new float[]{4f, 9f, 0.25f}, 3).sqrt().data();
            assertEquals(2.0, sqrt[0], 1e-6);
            assertEquals(3.0, sqrt[1], 1e-6);
            assertEquals(0.5, sqrt[2], 1e-6);
        }
    }

    @Test
    void absNegAndDivPowForward() {
        try (Basalt engine = Basalt.create()) {
            Tensor x = engine.tensor(new float[]{-2f, 0f, 3f}, 3);
            float[] abs = x.abs().data();
            assertEquals(2.0, abs[0], 1e-6);
            assertEquals(0.0, abs[1], 1e-6);
            assertEquals(3.0, abs[2], 1e-6);

            float[] neg = x.neg().data();
            assertEquals(2.0, neg[0], 1e-6);
            assertEquals(-3.0, neg[2], 1e-6);

            Tensor a = engine.tensor(new float[]{4f, 9f}, 2);
            Tensor b = engine.tensor(new float[]{2f, 3f}, 2);
            float[] div = a.div(b).data();
            assertEquals(2.0, div[0], 1e-6);
            assertEquals(3.0, div[1], 1e-6);

            float[] pow = a.pow(engine.tensor(new float[]{0.5f, 0.5f}, 2)).data();
            assertEquals(2.0, pow[0], 1e-6);
            assertEquals(3.0, pow[1], 1e-6);
        }
    }

    @Test
    void divBroadcastsAgainstRowVector() {
        try (Basalt engine = Basalt.create()) {
            Tensor matrix = engine.tensor(new float[]{1f, 2f, 3f, 4f}, 2, 2);
            Tensor row = engine.tensor(new float[]{1f, 2f}, 1, 2);
            float[] out = matrix.div(row).data();
            assertEquals(1.0, out[0], 1e-6);
            assertEquals(1.0, out[1], 1e-6);
            assertEquals(3.0, out[2], 1e-6);
            assertEquals(2.0, out[3], 1e-6);
        }
    }

    @Test
    void gradientsFlowThroughNewOps() {
        try (Basalt engine = Basalt.create()) {
            checkGradient(engine, new double[]{0.5, 1.2, -0.4}, x -> x.exp());
            checkGradient(engine, new double[]{0.5, 1.2, 2.5}, x -> x.log());
            checkGradient(engine, new double[]{0.5, 1.2, 2.5}, x -> x.sqrt());
            checkGradient(engine, new double[]{-1.5, 0.7, 2.0}, x -> x.abs());
            checkGradient(engine, new double[]{-1.5, 0.7, 2.0}, x -> x.neg());
            checkGradient(engine, new double[]{0.5, 1.2, 2.5}, x -> x.mul(x).exp().log());
            checkGradient(engine, new double[]{0.5, 1.2, 2.5},
                    x -> x.div(engine.tensor(new float[]{2f, 3f, 4f}, 3)));
            checkGradient(engine, new double[]{0.5, 1.2, 2.5},
                    x -> x.pow(engine.tensor(new float[]{2f, 3f, 2f}, 3)));
        }
    }

    @Test
    void softmaxCanBeWrittenFromPrimitives() {
        try (Basalt engine = Basalt.create()) {
            // 用新算子手写 softmax：exp / sum / div，验证组合起来是对的
            Tensor x = engine.tensor(new float[]{1f, 2f, 3f}, 3);
            Tensor shifted = x.sub(engine.tensor(new float[]{3f, 3f, 3f}, 3));
            Tensor numer = shifted.exp();
            Tensor denom = numer.sumAll().reshape(1, 1);
            float[] out = numer.div(denom).data();
            double total = 0;
            for (float value : out) {
                total += value;
            }
            assertEquals(1.0, total, 1e-5, "手写 softmax 的分布必须归一");
            assertTrue(out[2] > out[1] && out[1] > out[0], "大的输入要有大的概率");
        }
    }

    @Test
    void axisReductionsProduceTheRightValuesAndShapes() {
        try (Basalt engine = Basalt.create()) {
            // [[1,2,3],[4,5,6]]
            Tensor x = engine.tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, 2, 3);

            Tensor rowSum = x.sum(1);
            assertArrayEquals(new int[]{2}, rowSum.shape());
            assertArrayEquals(new float[]{6f, 15f}, rowSum.data(), 1e-6f);

            Tensor rowSumKeep = x.sum(1, true);
            assertArrayEquals(new int[]{2, 1}, rowSumKeep.shape());

            Tensor colMean = x.mean(0);
            assertArrayEquals(new int[]{3}, colMean.shape());
            assertArrayEquals(new float[]{2.5f, 3.5f, 4.5f}, colMean.data(), 1e-6f);

            Tensor rowMax = x.max(1);
            assertArrayEquals(new float[]{3f, 6f}, rowMax.data(), 1e-6f);

            // rank-1 归约退化成 [1]，与 sumAll 保持一致
            Tensor flat = engine.tensor(new float[]{1f, 2f, 3f, 4f}, 4);
            Tensor total = flat.sum(0);
            assertArrayEquals(new int[]{1}, total.shape());
            assertEquals(10.0, total.data()[0], 1e-6);
        }
    }

    @Test
    void reductionsRejectBadDimensions() {
        try (Basalt engine = Basalt.create()) {
            Tensor x = engine.tensor(new float[]{1f, 2f, 3f, 4f}, 2, 2);
            assertThrows(IllegalArgumentException.class, () -> x.sum(5));
            assertThrows(IllegalArgumentException.class, () -> x.mean(-1));
        }
    }

    @Test
    void numericallyStableSoftmaxFromPrimitives() {
        try (Basalt engine = Basalt.create()) {
            // 输入放大到 1000 量级：不先减最大值的话 exp 会溢出成 inf
            Tensor x = engine.tensor(new float[]{1000f, 1001f, 1002f}, 1, 3);
            Tensor shifted = x.sub(x.max(1, true));
            Tensor numer = shifted.exp();
            Tensor denom = numer.sum(1, true);
            float[] probs = numer.div(denom).data();

            double total = 0;
            for (float value : probs) {
                assertTrue(Float.isFinite(value), "概率不能是 inf/NaN");
                total += value;
            }
            assertEquals(1.0, total, 1e-5);
            assertTrue(probs[2] > probs[1] && probs[1] > probs[0]);
        }
    }

    @Test
    void gradientsFlowThroughAxisReductions() {
        try (Basalt engine = Basalt.create()) {
            // 通过 sum(1) 再求和，等价于对全部元素求和
            checkGradient(engine, new double[]{0.5, 1.2, 2.5, -0.7},
                    x -> x.reshape(2, 2).sum(1));
            checkGradient(engine, new double[]{0.5, 1.2, 2.5, -0.7},
                    x -> x.reshape(2, 2).mean(0));
            // max 的梯度是分段常数：选互不相等的值，避免并列
            checkGradient(engine, new double[]{0.5, 1.2, 2.5, -0.7},
                    x -> x.reshape(2, 2).max(1));
        }
    }

    @Test
    void rejectsCrossContextOperands() {
        try (Basalt first = Basalt.create(); Basalt second = Basalt.create()) {
            Tensor left = first.tensor(new float[]{1f, 2f}, 2);
            Tensor right = second.tensor(new float[]{1f, 2f}, 2);
            assertThrows(IllegalArgumentException.class, () -> left.div(right));
        }
    }
}
