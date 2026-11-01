package basalt.tests;

import basalt.Basalt;
import basalt.Device;
import basalt.torch.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 设备层：标签一致性、传输通道、以及「未实现设备必须显式失败」。
 *
 * <p>本机只有 CPU 后端。这些测试的意义是让设备校验与传输路径在接 GPU 之前
 * 就被执行过——否则那条通道要等到第一次上 GPU 才第一次运行。</p>
 */
class DeviceTest {

    @Test
    void tensorsReportCpuByDefault() {
        try (Basalt engine = Basalt.create()) {
            Tensor x = engine.tensor(new float[]{1f, 2f, 3f}, 3);
            assertEquals(Device.CPU(), x.device());
            assertTrue(x.device().isCpu());
            assertTrue(x.device().implemented());
            assertEquals("cpu", x.device().toString());
        }
    }

    @Test
    void movingToCpuCopiesStorage() {
        try (Basalt engine = Basalt.create()) {
            Tensor original = engine.tensor(new float[]{1f, 2f, 3f, 4f}, 2, 2);
            Tensor copy = original.to(Device.CPU());

            assertNotSame(original, copy, "传输必须返回新张量");
            assertNotSame(original.handle(), copy.handle(), "底层存储必须是新的一份");
            assertArrayEquals(original.data(), copy.data(), 0f);
            assertArrayEquals(new int[]{2, 2}, copy.shape());
            assertEquals(Device.CPU(), copy.device());
        }
    }

    @Test
    void copiedTensorIsIndependent() {
        try (Basalt engine = Basalt.create()) {
            Tensor original = engine.tensor(new float[]{1f, 2f}, 2);
            Tensor copy = original.to(Device.CPU());

            // 改副本不应该影响原张量：它们是两块独立存储
            Tensor scaled = copy.mul(engine.tensor(new float[]{10f, 10f}, 2));
            assertArrayEquals(new float[]{1f, 2f}, original.data(), 0f);
            assertArrayEquals(new float[]{10f, 20f}, scaled.data(), 0f);
        }
    }

    @Test
    void unimplementedDeviceFailsLoudly() {
        try (Basalt engine = Basalt.create()) {
            Tensor x = engine.tensor(new float[]{1f, 2f}, 2);
            Device cuda = Device.cuda(0);

            assertFalse(cuda.implemented(), "本构建没有 GPU 后端");
            assertEquals("cuda:0", cuda.toString());
            UnsupportedOperationException failure = assertThrows(
                    UnsupportedOperationException.class, () -> x.to(cuda));
            assertTrue(failure.getMessage().contains("only cpu"), failure.getMessage());
        }
    }

    @Test
    void deviceTagsRoundTrip() {
        assertEquals(Device.CPU(), Device.fromTag(0));
        assertEquals(Device.cuda(3), Device.fromTag(1003));
        assertEquals(1003, Device.cuda(3).tag());
        assertSame(Device.CPU(), Device.CPU());
        assertThrows(IllegalArgumentException.class, () -> Device.fromTag(-1));
        assertThrows(IllegalArgumentException.class, () -> Device.fromTag(9999));
        assertThrows(IllegalArgumentException.class, () -> Device.cuda(-1));
    }
}
