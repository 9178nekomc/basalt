package basalt;

import basalt.nativebridge.NativeContext;
import basalt.nativebridge.NativeTensorHandle;
import basalt.torch.Tensor;

/**
 * 张量存储所在的设备。
 *
 * <p>运行时要求参与同一运算的张量位于同一设备，跨设备会直接报错而不是隐式搬运。
 * 目前只实现了 {@link #CPU()}；{@link #cuda(int)} 返回的设备可以被持有、被比较、
 * 被当作传输目标传入，但真正用它做计算会抛出 {@link UnsupportedOperationException}，
 * 因为本机没有可验证的 GPU 后端。</p>
 *
 * <p>这样做的好处是：设备校验、传输通道、Java API 现在就被测试覆盖，
 * 接 GPU 后端时只需要实现内核，不用再改调用方。</p>
 */
public final class Device {

    /** 标签：与原生侧 {@code Device::tag()} 一一对应。 */
    private static final int CPU_TAG = 0;
    private static final int CUDA_BASE_TAG = 1000;

    private static final Device CPU_DEVICE = new Device("cpu", CPU_TAG, true);

    private final String name;
    private final int tag;
    private final boolean implemented;

    private Device(String name, int tag, boolean implemented) {
        this.name = name;
        this.tag = tag;
        this.implemented = implemented;
    }

    /** 主机内存，唯一的已实现后端。 */
    public static Device CPU() {
        return CPU_DEVICE;
    }

    /**
     * 请求一块 CUDA 设备。
     *
     * <p>设备对象本身可以创建、比较、打印；把它作为计算目标使用时会失败。</p>
     */
    public static Device cuda(int index) {
        if (index < 0 || index >= 1000) {
            throw new IllegalArgumentException("cuda device index out of range: " + index);
        }
        return new Device("cuda:" + index, CUDA_BASE_TAG + index, false);
    }

    /** 从原生标签还原设备。 */
    public static Device fromTag(int tag) {
        if (tag == CPU_TAG) {
            return CPU_DEVICE;
        }
        if (tag >= CUDA_BASE_TAG && tag < CUDA_BASE_TAG + 1000) {
            return cuda(tag - CUDA_BASE_TAG);
        }
        throw new IllegalArgumentException("unknown device tag: " + tag);
    }

    public int tag() {
        return tag;
    }

    public String name() {
        return name;
    }

    /** 该设备是否已有可执行的后端。 */
    public boolean implemented() {
        return implemented;
    }

    public boolean isCpu() {
        return tag == CPU_TAG;
    }

    /**
     * 把张量复制到本设备。
     *
     * <p>同设备传输同样会复制一份新存储——这条路径就是 GPU 后端将来要用的那条，
     * 现在先被测试覆盖。</p>
     */
    public Tensor move(Tensor tensor) {
        if (tensor == null) {
            throw new IllegalArgumentException("tensor must not be null");
        }
        if (!implemented) {
            throw new UnsupportedOperationException(
                    "device " + name + " has no backend in this build; only cpu is available");
        }
        NativeContext context = tensor.handle().context();
        NativeTensorHandle moved = context.transferTensor(tensor.handle(), tag);
        return Tensor.wrap(context, moved);
    }

    @Override
    public String toString() {
        return name;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof Device device && device.tag == tag;
    }

    @Override
    public int hashCode() {
        return Integer.hashCode(tag);
    }
}
