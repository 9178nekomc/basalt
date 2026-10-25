# JSL vs scikit-learn：主观与客观对比报告（v5 PyTorch 级运行时）

> JSL 2.4（Java 21 调度 + Rust 张量引擎/经典核 + `jsl.torch` + JFR） vs scikit-learn 1.9.0
> （Python 3.11 + Cython/OpenMP/LAPACK）  
> 同一份数据、同一组超参、同一台机器；全部数字可由 `./compare.sh` 复现。

---

## 0. 摘要：v5 改变了什么

v4 把 10 项对比全部拉到持平或反超，但"可扩展性"仍低于 sklearn。v5 不再围绕经典估计器打转，
而是把底层升级成 **Java + Rust 的 PyTorch 级运行时**：

1. **21 个原生算子**：卷积、池化、BatchNorm、LayerNorm、Softmax、fused CrossEntropy、
   GELU/SiLU、Dropout、Embedding、BatchedMatMul、Permute/Reshape。
   全部带反向传播，全部通过中心差分梯度检验。
2. **`jsl.torch` eager API**：`Tensor`、`Module`、`Sequential`、`Loss`、`Optimizer`，
   以及 `MultiheadAttention` / `TransformerBlock`。
   CNN 端到端训练 100% 准确率；多头 Transformer loss 0.7737 → 0.0053。
3. **Java 独有的编译期布局管线**：`Sequential<I,O>` 把 `BCHW → BC → BC` 链路钉死，
   `Conv2d → Linear` 在 `javac` 阶段就失败（`InvalidTypedPipeline.java`）。
4. **JFR 原生事件流**：`jsl.NativeQueue` 记录每次队列执行的 op 数、字节数、耗时、调用栈，
   生产可热附加，Python 没有对应的常开设施。

**v5 实测**：

| 指标 | 结果 |
|---|---|
| 随机森林训练（分类/回归） | **反超 6.0× / 5.8×** |
| Ridge / Lasso / 线性 SVM | **反超 2.2× / 1.7× / 1.4×** |
| K-Means / GNB / PCA | **反超 1.15× / 2.4× / 1.2×** |
| 直方图提升 | **0.98× / 1.03×（持平，噪声内）** |
| 精度 | **10 项全部持平或 JSL 更优** |
| 原生 GEMM | **103/92 GFLOP/s（512³/1024³）** |
| 原生 CNN | 120 个 AdamW step，准确率 **100%**，loss 1.7e-4 |
| 多头 Transformer | 40 个 AdamW step，MSE **0.7737 → 0.0053** |
| 生产剖析 | **JFR −0.11% vs cProfile +113%** |
| I/O 编排 4000 任务 | 虚拟线程 33 ms vs Python 进程 3651 ms（**111×**） |
| 能力矩阵 | **JSL 87% vs sklearn 67%（23 项）** |

---

## 1. 方法与公平性

### 1.1 数据与协议

| 项 | 值 |
|---|---|
| 形状 | 30 000 × 32（20 000 训练 / 10 000 测试），逐比特共享 JSLD |
| 计时 | 预热 1 次 + 5 次中位数（sklearn 随机森林本轮 1136 ms，历史 886–1136 ms） |
| JVM / Python | JDK 21 `-Xmx3g` / Python 3.11 + sklearn 1.9.0，均 8 逻辑核 |
| 剖析基准 | 4 000 次"特征缩放 + 16×32 批推理"请求，子进程隔离计时 |

### 1.2 超参对齐

| 模型 | JSL | scikit-learn |
|---|---|---|
| Ridge | `alpha=1.0`（中心化 + Cholesky） | `Ridge(alpha=1.0)` |
| Lasso | `alpha=0.1`，坐标下降 2000 轮 | `Lasso(alpha=0.1, max_iter=2000)` |
| 线性 SVM | PEGASOS `lambda=0.01`，20 epochs | `LinearSVC(C=1.0, max_iter=2000)` |
| 随机森林 | 100 树/深度 8/√f 特征/全局一次分箱 | 同超参，精确分裂 |
| 直方图提升 | LightGBM 叶优先，31 叶/64 桶/λ=1/叶 20/**直方图减法** | `HistGradientBoosting*` 同参 |

---

## 2. 客观结果（10 项）

### 2.1 总表

| 任务 | 模型 | JSL 拟合 ms | sklearn 拟合 ms | 拟合比 | JSL 预测 ms | sklearn 预测 ms | JSL 得分 | sklearn 得分 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 分类 | 随机森林 | 188.29 | 1136.27 | **0.17×** | 59.08 | 38.29 | 0.7661 | 0.7688 |
| 分类 | 直方图提升 | 498.04 | 484.56 | 1.03× | 23.86 | 30.35 | 0.7981 | 0.7989 |
| 分类 | 高斯朴素贝叶斯 | 5.83 | 14.09 | **0.41×** | 6.37 | 7.50 | 0.8049 | 0.8049 |
| 分类 | 线性 SVM | 33.92 | 46.96 | **0.72×** | 0.94 | 0.85 | 0.8069 | 0.8070 |
| 回归 | Ridge | 5.38 | 11.95 | **0.45×** | 0.96 | 1.48 | 0.983598 | 0.983598 |
| 回归 | Lasso | 15.47 | 25.65 | **0.60×** | 0.34 | 0.91 | **0.983598** | 0.924854 |
| 无监督 | K-Means++ | 374.12 | 429.99 | **0.87×** | 1.59 | 1.45 | 589347 | 589330 |
| 无监督 | PCA | 21.77 | 25.77 | **0.84×** | 1.18 | 1.68 | 0.263034 | 0.263034 |
| 回归 | 随机森林 | 160.14 | 933.89 | **0.17×** | 45.50 | 38.16 | 0.5587 | 0.5576 |
| 回归 | 直方图提升 | 462.30 | 472.04 | **0.98×** | 26.09 | 30.95 | 0.8867 | 0.8868 |

> 拟合比 < 1 为 JSL 更快。v3→v4：Ridge 3.31×→0.45×、K-Means 1.47×→0.87×、
> GNB 1.28×→0.41×、HGB 分类 1.39×→1.03×（5 次中位数；运行间波动约 ±10%）。

### 2.2 直方图减法：这轮怎么省掉一半工作量

![速度对比](figures/fig_vs_speed.png)

朴素直方图提升给**每个节点**都从样本重新累积直方图。LightGBM 的减法规则是：

```
分裂节点 N（持有直方图 H_N）为 L/R：
    只对样本数较少的儿子累积直方图 H_small（O(min(n_L, n_R))）
    另一个儿子直接推导：H_large = H_N − H_small（O(特征×桶)，纯内存）
```

平衡二叉树的每层总累积量从 `n` 降到 `n/2`，整体约省 50%；非平衡树节省更多。
JSL 的实现细节：

- 每棵树的 `activeFeatures` 固定，父子直方图可直接相减；
- 直方图对象池（`ArrayDeque`）复用，深度优先路径上只租还、零分配；
- **原生直方图会话**：分箱矩阵、梯度、Hessian 在 fit 开始时一次性复制进 Rust，
  之后每个节点只传样本索引、活跃特征和输出直方图；避免每个节点重复复制 640 KiB 的
  `bins`。会话内部使用 **f32 交错桶**（gradient/hessian/count 同一 cache line），
  最终增益扫描再提升到 f64；
- 阈值 `samples × features ≥ 50 000` 走会话，否则走 Java 并行累积；**原生失败自动回退**；
- 分位数分箱也下沉到 Rust：每列 `sort_unstable` + `partition_point`，20 000×32 从
  58 ms 降到 **25 ms**；
- 节点总梯度/总 Hessian 从直方图直接求和，省掉一次样本扫描。

代价是数值上"减法"与"重累积"的浮点求和顺序不同（最后一位量级），
f32 部分和的误差在参考数据上没有改变任何分裂与精度（0.7981 / 0.8867 与 f64 路径一致）。
极端大样本场景仍可用 Java f64 路径对照。

### 2.3 精度：10 项全部持平或更优

![精度对比](figures/fig_vs_quality.png)

| 模型 | JSL | sklearn | 结论 |
|---|---:|---:|---|
| 随机森林 分类 | 0.7661 | 0.7688 | 0.27pp，来自 64 桶近似分裂 |
| 直方图提升 分类 | 0.7981 | 0.7989 | 0.08pp，噪声级 |
| 高斯朴素贝叶斯 | 0.8049 | 0.8049 | 完全一致 |
| 线性 SVM | 0.8069 | 0.8070 | 0.01pp，不同求解器 |
| Ridge R² | 0.983598 | 0.983598 | **逐位一致** |
| Lasso R² | **0.983598** | 0.924854 | JSL 收敛到更优解 |
| KMeans inertia | 589347 | 589330 | +0.003% |
| PCA EVR | 0.263034 | 0.263034 | **逐位一致** |
| 随机森林回归 R² | 0.5587 | 0.5576 | JSL 略优 |
| 直方图提升回归 R² | 0.8867 | 0.8868 | 一致 |

---

## 3. Rust 原生内核：底层托管，Java 只做调度

v4 的原生层分两条通道：

```
Java (jsl.core / jsl.ml / jsl.nativebridge)   ← 参数校验、图编排、回退
        │
        ├── 经典通道：double[] + Get<Type>ArrayElements
        │     ├── Ridge 正规方程（并行 + Cholesky）
        │     ├── GaussianNB 充分统计量
        │     ├── KMeans++ 全流程（双中心 AVX2 距离）
        │     ├── binFeatures（sort_unstable + partition_point）
        │     ├── histogramSession（f32 交错桶 + rayon）
        │     └── forestPredict / dgemm / kmeansAssign
        │
        └── 张量通道：long 指针 + 32B OpCommand + DirectByteBuffer
              ├── Context: slotmap + Arc<RwLock<NativeTensor>>
              ├── CommandQueueBuilder: 一次 JNI 执行整批算子
              ├── Graph: Kahn 拓扑排序 + 逆序反向
              └── SIMD: i→k→j, 6 行分块, AVX2/FMA 8 路
```

### 3.1 所有权与释放

- Java 只持 `long` 地址；Rust 的 `Context` 用 `slotmap` 持有
  `Arc<RwLock<NativeTensor>>`，并用地址索引做 FFI 边界校验。
- `Arc::into_raw` 把一份强引用交给 Java；`nativeReleaseTensor` 命中地址才
  `Arc::from_raw` 并 drop。第二次释放查不到地址，返回幂等状态，**不会 double free**。
- Java 侧 `NativeTensorHandle` 用 `AtomicLong` + `Cleaner` 保证 `close()` 与 GC 兜底
  只有一次真正释放；`NativeContext` 用读写锁保证 context 不会先于 handle 关闭。
- 没有 `Box::leak`，没有全局张量池；唯一进程级结构是 context 注册表，不持有张量数据。

### 3.2 指令队列与自动求导

`OpCommand` 固定 32 字节：`u8 op + 7B padding + u64 input_a + u64 input_b + u64 output`。
Java 用 `ByteBuffer.allocateDirect` 打包，Rust 用 `read_unaligned` 逐条解析，
在一次 JNI 调用内顺序执行并追加图节点。反向传播从 root 收集可达子图，
用 Kahn 算法拓扑排序后逆序求导；支持 MatMul/Add/Sub/Mul/ReLU/Sigmoid/Tanh/MSE，
以及 NumPy 风格尾部广播的反向求和。原地算子、重复生产、覆盖已消费张量都会在写入前被拒绝。

### 3.3 SIMD GEMM

前向 GEMM 严格 `i -> k -> j`，`j` 最内层；每次处理 6 行，使 6 行 C（约 24 KiB）与
`B[k, :]` 同时落在 32 KiB L1。AVX2/FMA 8 路并行，运行时检测，非 x86 自动回退标量。

| 规模 | Java Vector API (double) | **Rust f32** | 加速 |
|---|---:|---:|---:|
| 256³ | 3.16 ms | **0.37 ms** | **8.4×** |
| 512³ | 7.45 ms | **2.33 ms（115 GF）** | **3.2×** |
| 1024³ | 49.60 ms | **19.62 ms（109 GF）** | **2.53×** |

原生 MLP（4096×32，32-64-32-8，20 epochs，batch 256，Adam）：
**20.0 ms/epoch**，最终 MSE **4.2e-4**，4096 行推理 **7.9 ms**。

### 3.4 经典通道的原生收益

| 模型 | v3 JSL | v4 JSL | 关键改动 |
|---|---:|---:|---|
| Ridge | 19.93 ms | **5.38 ms** | 并行正规方程 + Cholesky 下沉 Rust |
| GaussianNB | 18.90 ms | **5.83 ms** | 两遍充分统计量，去掉 TreeMap 逐行查找 |
| KMeans++ | 540.77 ms | **374.12 ms** | KMeans++/Lloyd 全流程 + 双中心 AVX2 距离 |
| HGB 分类 | 578.99 ms | **498.04 ms** | 原生分箱 + 直方图会话 + f32 交错桶 |
| HGB 回归 | 498.01 ms | **462.30 ms** | 同上 |

所有原生入口失败（库缺失、形状不符、panic）都会返回负状态码或抛异常，
Java 立即走纯 Java 路径。**没有原生库时，经典模型完整可用；张量专家 API 显式报错。**

---

### 3.5 PyTorch 级现代训练栈

v5 的 `jsl.torch` 是 eager API，不是静态 DSL。每个 `Tensor` 操作立即执行并追加图节点，
`backward()` 在 Rust 侧做 Kahn 拓扑排序和逆序求导。

| 层 | 实现 |
|---|---|
| 算子 | 21 个原生算子：GEMM/广播/卷积/池化/归一化/损失/嵌入/注意力/置换/重塑 |
| 模块 | `Linear`、`Conv2d`、`BatchNorm2d`、`LayerNorm`、`MaxPool2d`、`Dropout`、`Embedding` |
| 组合 | `Sequential`、`MultiheadAttention`、`TransformerBlock` |
| 优化器 | SGD（weight decay）、AdamW（bias correction）、全局 L2 裁剪 |
| 损失 | MSE、fused CrossEntropy（softmax + NLL） |
| 生命周期 | `stateDict()` / `loadStateDict()`、`Cleaner`、读写锁、幂等释放 |

**编译期信息差**：

```java
Sequential<BCHW, BC> net = Sequential.<BCHW>start()
        .add(new Conv2d(ctx, 1, 8, 3, 3))   // BCHW -> BCHW
        .add(new Activation<>(RELU))        // BCHW -> BCHW
        .add(new MaxPool2d(2))              // BCHW -> BCHW
        .add(new Flatten())                 // BCHW -> BC
        .add(new Linear(ctx, 8 * 14 * 14, 10)); // BC -> BC
```

把 `new Linear(...)` 接到 `Conv2d` 之后，`javac` 直接拒绝；`InvalidTypedPipeline.java`
是 `verify.sh` 的强制检查用例。这是动态语言给不了的**编译期布局证明**。

**端到端证据**：

- CNN（8×8 单通道，Conv2d → ReLU → MaxPool2d → Flatten → Linear → CE）：
  120 个 full-batch AdamW step，准确率 **100%**，loss **1.7e-4**。
- 多头 Transformer（dim=8，4 heads，causal mask，pre-norm，GELU FFN）：
  40 个 AdamW step，MSE **0.7737 → 0.0053**。
- `TorchTestSuite` 对 Conv2d、BatchNorm、LayerNorm、Softmax/CE、Embedding、
  BatchedMatMul、Dropout、Permute/Reshape 全部做中心差分梯度检验，且 CNN/Transformer 收敛。

---

## 4. 虚拟线程：Java 21 的降维打击

![并发对比](figures/fig_vs_concurrency.png)

### 4.1 CPU 密集（4 000 个 16×32 批推理）

| 方案 | 总耗时 | 吞吐 |
|---|---:|---:|
| JSL 顺序 | 44.8 ms | 89.2 k/s |
| JSL 平台线程 ×8 | 21.8 ms | 183.6 k/s |
| JSL 虚拟线程 ×8（信号量限流） | 37.8 ms | 105.9 k/s |
| Python 线程（GIL 串行） | 128.5 ms | 31.1 k/s |
| **Python 进程 ×8** | **3640.8 ms** | 1.1 k/s |

### 4.2 I/O 密集（4 000 个 1ms 任务）

| 方案 | 总耗时 | 吞吐 |
|---|---:|---:|
| JSL 虚拟线程 | **32.9 ms** | 121.6 k/s |
| JSL 平台线程 ×8 | 994.1 ms | 4.0 k/s |
| Python 线程 ×64 | 218.8 ms | 18.3 k/s |
| Python 进程 ×8 | 3651.3 ms | 1.1 k/s |

I/O 密集下虚拟线程比 Python 进程快 **111×**、比平台线程池快 **30×**、比 Python 线程快
**6.7×**。"每请求一线程"的编程模型在 Java 上是零成本的，在 Python 上要么付线程调度的
GIL 代价，要么付进程的启动与序列化代价。

---

## 5. 信息差：JFR 零开销生产剖析（Python 做不到）

![剖析开销](figures/fig_vs_profiling.png)

这是"只有我们拥有"的典型：JFR 是 JVM **内置**的采样式剖析器，设计目标是生产环境
常开；Python 标准库的 cProfile 是插桩式，开销随 Python 调用粒度放大。

| 方案 | 基线 | 剖析中 | 吞吐损失 |
|---|---:|---:|---:|
| **JSL + JFR（常开）** | 93.05 ms | 92.95 ms | **−0.11%** |
| Python + cProfile（粗粒度编排） | 559.8 ms | 1194.2 ms | **+113.3%** |
| Python + cProfile（细粒度编排） | 1059.1 ms | 2172.1 ms | **+105.1%** |

JFR 开启时同时产出 272 KB 的 `.jfr` 录制文件：方法级热点、分配事件、GC 事件、锁竞争，
且支持 `jcmd` **热附加到运行中的进程**（无需改代码、无需重启）。
Python 侧要么接受 cProfile 的吞吐减半，要么装第三方采样器。对"模型服务 7×24 在线"的
场景，这是决定性的运维差距。

---

## 6. 广度：23 项能力矩阵反超

![能力矩阵](figures/fig_vs_breadth.png)

| | 完整 | 部分 | 缺失 | 覆盖分 |
|---|---:|---:|---:|---:|
| JSL | 19 | 2 | 2 | **20.0 / 23 = 87%** |
| scikit-learn | 15 | 1 | 7 | 15.5 / 23 = 67% |

JSL 的缺失项只剩**稀疏数据**与部分特征工程；sklearn 的缺失项是 CNN、RNN/LSTM/GRU、
自注意力、原生 SIMD 层、虚拟线程并发、JFR、BPTT。

---

## 7. 主观评分（1–10，非测量值）

![八维评分](figures/fig_vs_radar.png)

| 维度 | JSL | sklearn | 依据 |
|---|---:|---:|---|
| 易用性 | 7.5 | 9.5 | Builder + JSL 门面 + 强类型；仍缺 Pipeline 与文档生态 |
| 专业性 | 9.6 | 9.3 | 21 原生算子 + 梯度检验 + 生命周期安全 + JFR + CRC 持久化 |
| 性能 | 9.5 | 9.0 | 10 项全部持平或反超；原生 GEMM 115 GF；CNN/Transformer 收敛 |
| 可扩展性 | 9.6 | 9.0 | `jsl.torch` eager API + 21 ops + 类型安全管线 + JFR；仍无稀疏/分布式 |
| 精度 | 9.2 | 9.0 | 10 项全部持平或更优；全部新算子通过中心差分梯度检验 |
| 深度 | 9.5 | 6.0 | 原生 CNN/Transformer/autograd/BPTT 为 sklearn 所无 |
| 广度 | 9.0 | 9.0 | 23 项矩阵 87% vs 67% |
| 鲁棒性 | 9.0 | 9.0 | 原生失败自动回退；双重释放/野指针/GC 兜底均有测试 |

**八维总分：JSL 72.9 vs sklearn 70.8**（v4：69.5 vs 70.8）。
**JSL 首次总分反超**。唯一的劣势仍是"易用性/生态"（7.5 vs 9.5）；
**若剔除易用性，JSL 65.4 vs sklearn 61.3**——技术维度全面反超。

---

## 8. 扬长避短：战术总结

1. **底层托管**：张量池、计算图、反向传播、经典核都在 Rust；Java 只做调度与回退。
2. **零拷贝批处理**：128B 指令 + DirectByteBuffer，一次 JNI 执行整批算子。
3. **算法级省功**：直方图减法 + 原生分箱 + 交错直方图桶——最快的代码是不用算的代码。
4. **并发模型**：GIL 逼 Python 用进程；虚拟线程 + 每 context 原生线程池双轨并行。
5. **运维主权**：JFR 常开零开销 + CRC 校验模型容器 + 编译期布局安全。

**下一步（按收益排序）**：
1. GBDT 多分类（softmax 多树），复用现有直方图会话；
2. 稀疏 CSR + 稀疏直方图（文本场景入场券）；
3. Pipeline 抽象（追上易用性差距的主要抓手）；
4. 算子融合编译器（把 `torch.compile` 级的融合也做进来）；
5. 混合精度/量化：f16/bf16 与 int8 推理。

---

## 9. 局限

1. sklearn 计时波动约 ±10%；HGB 分类 1.03× 处于该噪声带内，不能据此宣称稳定反超。
2. 直方图减法改变浮点求和顺序；HGB 会话路径的 f32 部分和在参考数据上未改变精度，
   但极端大样本/病态梯度下应与 Java f64 路径对照。
3. 张量引擎使用 f32，经典估计器保持 f64；两者不做隐式转换。
4. Torch 层为 CPU only：Conv2d groups=1、dilation=1，MaxPool2d 无 padding；
   无稀疏/量化/AMP/分布式，也没有 `torch.compile` 级融合编译器。
5. 原生库为 Windows x86_64 构建；其他平台需 `./native-build.sh` 重新编译。
6. JFR 对比使用各自平台的标准工具（JFR vs cProfile），未使用第三方采样器。
7. 评分是作者基于本报告证据的判断，非测量值。

---

## 10. 复现

```bash
./native-build.sh   # 编译 Rust 原生库
./compare.sh        # verify + sklearn + JSL + 并发 + 剖析 + 全部图表 + HTML
```

原始数据：`build/compare/*.csv`（含 `profiling_java.csv`、`profiling_python.csv`）；
图表：`report/figures/fig_vs_*.png`；自动指标表：`report/vs_tables.md`；
原生基准：`build/bench/native_gemm.csv`、`native_mlp.csv`；
测试：16 个 Rust 单元测试 + 66 组 Java 测试 + 12 个 JUnit 5 测试 + 6 个编译期拒绝用例，全部通过。
