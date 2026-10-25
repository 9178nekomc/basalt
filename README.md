# Basalt — Java + Rust PyTorch-Class Runtime

> **Maven coordinates** — `io.github.9178nekomc:basalt:3.4.0`  
> **Runtime** — Java 21 (`--add-modules jdk.incubator.vector`) + bundled Rust native library  
> **One dependency, no Python, no GIL, no manual native build on Windows x86_64.**  
> **Docs** — <https://9178nekomc.github.io/basalt/>

## English

### Install

```xml
<dependency>
    <groupId>io.github.9178nekomc</groupId>
    <artifactId>basalt</artifactId>
    <version>3.4.0</version>
</dependency>
```

Build and install locally from source:

```bash
mvn clean install
```

The shaded, self-contained artifact is `basalt-3.4.0-all.jar` (classifier `all`).
It contains the Windows x86_64 native runtime; other platforms can add their own
`native/<os>-<arch>/` library or point `-Djsl.native.path` at a locally built one.

### QuickStart — 10 lines, forward + backward

```java
try (Basalt jsl = Basalt.create()) {
    Tensor x = basalt.randn(32, 8);
    Tensor y = basalt.randn(32, 1);
    Linear hidden = basalt.linear(8, 16);
    Linear output = basalt.linear(16, 1);
    Tensor loss = output.forward(hidden.forward(x).relu()).mse(y);
    loss.backward();
    System.out.println(loss.data()[0]);
}
```

### KNN classification

```java
Matrix x = Matrix.fromRows(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}});
double[] y = {0, 1, 1, 0};
Estimator knn = KNearestNeighbors.builder().k(3).build().fit(x, y);
double[] prediction = knn.predict(x);
```

### Random forest

```java
RandomForest forest = RandomForest.builder()
        .trees(100)
        .maxDepth(8)
        .seed(42)
        .build()
        .fit(x, y);
double[] prediction = forest.predict(x);
```

### Hand-written CNN

```java
try (Basalt jsl = Basalt.create()) {
    Tensor images = basalt.tensor(pixels, 64, 1, 28, 28);
    Tensor labels = basalt.tensor(targets, 64);
    Sequential<BCHW, BC> cnn = basalt.<BCHW>sequential()
            .add(basalt.conv2d(1, 8, 3, 1, 1))
            .add(basalt.activation(Activation.RELU))
            .add(new MaxPool2d(2))
            .add(new Flatten())
            .add(basalt.linear(8 * 14 * 14, 10));
    var lossFunction = basalt.crossEntropy();
    var optimizer = basalt.adamw(cnn.parameters(), 1e-3f);
    for (int step = 0; step < 100; step++) {
        Tensor loss = lossFunction.loss(cnn.forward(images), labels);
        loss.backward();
        optimizer.step();
        optimizer.zeroGrad();
        basalt.clearGraph();
    }
}
```

### Why Basalt

- **Rust at the bottom**: AVX2/FMA GEMM, im2col convolution, rayon parallelism. No Python interpreter, no GIL, no per-op FFI marshalling.
- **PyTorch-like API**: eager `Tensor`, `Module`, `Sequential`, `AdamW`, `CrossEntropyLoss`, `stateDict`, multi-head attention.
- **Compile-time safety Python cannot match**: `Sequential<I,O>` closes the layout chain at `javac` time; `Conv2d → Linear` is a compile error, not a runtime shape exception.
- **One dependency**: Java 21 + one Maven coordinate. The native library is extracted from the jar and loaded automatically.
- **Observability**: `basalt.NativeQueue` JFR events record every command-queue execution with op count, bytes, duration and stack trace.

> Runtime note: add `--add-modules jdk.incubator.vector` to your application JVM arguments.
> For Maven Surefire: `<argLine>--add-modules jdk.incubator.vector</argLine>`.

---

## 中文文档

Basalt 是一个**Java 调度、Rust 托管底层**的 PyTorch 级运行时：eager 张量、动态计算图、
反向传播、原生 CNN/Transformer 算子、AdamW、JFR 事件流，以及经典机器学习估计器。
Java 21 负责 API、校验、类型安全与编排；性能敏感的算子、张量生命周期和反向传播由 Rust 持有。
目标不是复刻 sklearn，而是把 Python/PyTorch 的动态易用性**抄过来**，再用 Java/Rust 独有的
编译期布局管线、JFR、虚拟线程和零拷贝 JNI 建立信息差。

- 语言/运行时：Java 21（`--release 21`，`jdk.incubator.vector`）+ **Rust 原生内核（JNI）**
- 依赖：Java 侧 **0**；原生侧 `jni` + `rayon` + `slotmap` + `matrixmultiply`（`./native-build.sh`）
- 构建：Maven 3.9+（`mvn clean install`）或 `javac` + `cargo` + `bash`；无 Gradle
- 验证：16 个 Rust 单元测试 + 66 组 legacy Java 套件 + 12 个 JUnit 5 SDK 测试
  + 6 个编译期拒绝用例 + `-Xlint:all`（除 JDK incubating 提示外零告警）
- 文档：[《Basalt 终极架构与性能基准测试报告》](report/Basalt_ARCHITECTURE_AND_BENCHMARK.md)
- API 参考：[docs/index.html](docs/index.html)（Sphinx 版式，离线可读）

---

## 下载

发布资产见 [Releases · v3.4.0](https://github.com/9178nekomc/basalt/releases/tag/v3.4.0)：

| 文件 | 说明 |
|---|---|
| `basalt-3.4.0-all.jar` | Fat Jar（classifier `all`），含 Windows x86_64 原生库，可直接 `java -cp` |
| `basalt-3.4.0.jar` | 标准 jar，原生库在 `native/windows-x86_64/` 资源目录内 |
| `basalt-3.4.0-sources.jar` | 源码包 |
| `basalt_native.dll` | Windows x86_64 原生库，独立分发用 |
| `basalt-docs-3.4.0.zip` | 离线文档站（13 页 HTML + 2 个基准报告），解压后双击 `index.html` |

---

## API 参考文档（HTML）

完整 API 参考是一个可直接双击打开的静态站点，版式对齐 docs.python.org：
顶部面包屑导航、左侧栏（下载 / 各版文档 / Other resources）、两栏文档章节网格、
`dl.py` 签名块、`参数 / 返回 / 抛出` 字段列表、页内永久链接、`Ctrl+K` 快速搜索，
以及**自动 / 浅色 / 深色**三档主题（默认跟随系统）。

入口：**[`docs/index.html`](docs/index.html)**

| 页面 | 内容 |
|---|---|
| `docs/index.html` | 首页：文档章节网格、安装、快速开始、三层 API 边界、模块索引 |
| `docs/api/index.html` | 模块索引（9 个包，附阅读顺序建议） |
| `docs/api/basalt.html` | `jsl` 门面与异常 |
| `docs/api/jsl_core.html` | 布局标记、`Matrix`、`MatMul`、`VectorMath`、`NativeKernels`、模型容器原语 |
| `docs/api/jsl_data.html` | `DataLoader`（BasaltD 流式 / 内存映射）、`Normalization` |
| `docs/api/jsl_io.html` | `ModelCodec`（`.bsl` v2）、`OnnxModel` / `OnnxConverter` |
| `docs/api/jsl_ml.html` | 经典估计器、树、梯度提升、聚类、降维、指标、切分 |
| `docs/api/jsl_nn.html` | 手写层、LSTM/GRU/RNN、注意力、序列训练、序列化 |
| `docs/api/basalt_nativebridge.html` | `NativeContext`、`NativeTensorHandle`、`CommandQueueBuilder`、`NativeMLP` |
| `docs/api/jsl_torch.html` | eager `Tensor`、`Module`/`Sequential`、CNN/Transformer 模块、优化器、损失 |
| `docs/genindex.html` | 总索引（按首字母） |
| `docs/readme.html` | 本 README 的 HTML 版（同一版式，静态托管下不会退化成纯文本） |

覆盖范围：**226 个公开类型、1463 个公开成员**，其中 1463 个成员全部带逐参数说明
（共 1423 条参数条目、1237 组 `参数/返回/抛出` 字段列表），另有 23 个包内可见类型
单列在各模块页末尾，并明确标注为非稳定 API。

### 文档是怎么生成的

签名不允许手写，说明不允许自动编造，因此拆成三步：

```bash
# 1. 扫描源码，提取精确签名、参数名、抛出子句 → build/api/api.json
python tools/api_dump.py --src src/main/java --out build/api/api.json

# 2. 按模块切出待填骨架（供撰写说明时对齐 key）
python tools/make_briefs.py --api build/api/api.json --out build/api/briefs

# 3. 合并骨架与说明，渲染 Sphinx 风格站点
python tools/build_api_docs.py --api build/api/api.json \
    --docs 'tools/docs_data/g*.json' --out docs

# 4. 冒烟测试：标签闭合、id 唯一、链接与锚点可达、无占位符
python tools/check_docs.py --root docs
```

- `tools/api_dump.py`：掩码扫描（注释/字符串置空但保留偏移）+ 深度成员切分，
  能识别嵌套类型（`RandomForest.Builder`）、枚举常量、record 组件、泛型方法与 `throws`。
- `tools/docs_data/g*.json`：说明数据层，契约见 `tools/docs_data/SCHEMA.md`。
  渲染器按 `(qualified, key)` 匹配，**匹配不上的说明直接丢弃并报错**，
  所以文档不可能描述一个不存在的参数。
- `tools/build_api_docs.py`：零外部依赖渲染器，输出 13 个 HTML 页（含 `readme.html`）。
  **每页自包含**：CSS、主题/搜索脚本、932 条搜索索引全部内联，不引用任何外部
  `_static/` 目录——静态托管（Jekyll、各类预览服务）会跳过下划线开头的目录，
  外部资源一旦 404，页面就会退化成无样式纯文本。
- `tools/md_render.py`：README 的 Markdown 渲染器（标题、围栏代码、GFM 表格、
  嵌套列表、引用块、行内标记），CJK 换行不插空格，`java` 代码块复用同一套高亮。
- `tools/check_docs.py`：退出码非 0 即视为文档损坏，可用于 CI 门禁。

当前 `check_docs.py` 结果：13 页、249 个类型块、1211 个方法块、1237 组字段列表、
0 个占位符、0 个断链。

---

## 0. Maven SDK 安装与 QuickStart（中文）

```xml
<dependency>
    <groupId>io.github.9178nekomc</groupId>
    <artifactId>basalt</artifactId>
    <version>3.4.0</version>
</dependency>
```

从源码安装：

```bash
mvn clean install
```

产物：
- `target/basalt-3.4.0.jar`：标准 jar（含 `native/windows-x86_64/basalt_native.dll` 资源）
- `target/basalt-3.4.0-all.jar`：Fat Jar（classifier `all`，可直接 `java -cp`）
- `target/basalt-3.4.0-sources.jar`：源码包

10 行完成一次前向 + 反向：

```java
try (Basalt jsl = Basalt.create()) {
    Tensor x = basalt.randn(32, 8);
    Tensor y = basalt.randn(32, 1);
    Linear hidden = basalt.linear(8, 16);
    Linear output = basalt.linear(16, 1);
    Tensor loss = output.forward(hidden.forward(x).relu()).mse(y);
    loss.backward();
    System.out.println(loss.data()[0]);
}
```

- 统一入口：`basalt.Basalt`；张量/模块：`basalt.torch.*`；经典模型：`basalt.ml.*`。
- 用户只创建 `Tensor`，不需要手动释放；`Cleaner` 兜底，严谨场景仍可 `try-with-resources`。
- 运行参数：`--add-modules jdk.incubator.vector`（Maven Surefire 已配置）。

---

## 1. 能力矩阵

| 领域 | 实现 | 关键点 |
|---|---|---|
| 线性模型 | `LinearRegression`、`LogisticRegression`、`Ridge`、`Lasso` | 梯度下降 / Cholesky / 坐标下降 |
| 近邻 | `KNearestNeighbors`、`KNeighborsRegressor` | SIMD 欧氏/曼哈顿距离 |
| SVM | `LinearSVM` | PEGASOS 亚梯度，二分类 |
| 树模型 | `DecisionTree`、`RandomForest` | 直方图分裂、分类/回归、OOB、特征重要性 |
| 梯度提升 | `HistogramGradientBoosting` | 二阶增益、XGBoost 深度优先 / LightGBM 叶优先、特征采样、验证集早停 |
| 聚类 | `KMeans`、`DBSCAN` | K-Means++、并行分配、重启、空簇修复；密度聚类 |
| 工具 | `Metrics`、`DataSplit`、`Basalt` | 指标/混淆矩阵/ROC-AUC、holdout+k-fold、一行门面与交叉验证 |
| 降维 | `PCA` | 协方差 Jacobi / Gram 对偶双求解器、白化、自动选择 |
| 贝叶斯 | `GaussianNaiveBayes` | 对数域 softmax、尺度不变方差平滑 |
| CNN | `Conv2D`、`MaxPooling2D`、`Flatten`、`Dense` | 分阶段 Builder，布局编译期检查 |
| 序列 | `LSTM`、`GRU`、`VanillaRNN` | 完整 BPTT、参数级梯度检验、工作区复用 |
| 注意力 | `SelfAttention` | Q/K/V 反向、Softmax 雅可比、causal mask |
| 序列训练 | `SequenceNetwork` | 池化头 + Dense 头 + 损失 + 全局范数裁剪 + 小批量 |
| 数据 | `DataLoader` | BasaltD 流式/内存映射、零拷贝批量、epoch 洗牌、标准化 |
| 并发 | `InferenceEngine` | Java 21 虚拟线程 + 信号量限流的批量推理编排 |
| 剖析 | JFR（JDK 内置） | 生产级常开剖析，零开销；对比见 `report/Basalt_VS_SKLEARN.md` 第 5 节 |
| 原生经典核 | `NativeKernels` (JNI → Rust) | AVX2/FMA f64 GEMM、KMeans 全流程、Ridge 正规方程、GNB 统计量、分箱、森林预测、直方图会话；失败自动回退 |
| 原生张量 | `NativeContext` / `NativeTensorHandle` | `Arc<RwLock<NativeTensor>>` + slotmap、惰性梯度、幂等释放、`Cleaner` 兜底 |
| 指令队列 | `CommandQueueBuilder` | 128 字节 `OpCommand`（5 输入 + 16 内联属性）、`ByteBuffer.allocateDirect` 零拷贝、单次 JNI 执行整批算子 |
| 自动求导 | `nativeBackward` | 计算图 + Kahn 拓扑排序、21 个算子（GEMM/广播/卷积/池化/归一化/注意力/嵌入）、广播反向 |
| 原生训练 | `NativeMLP` | 多层感知机、Adam、mini-batch、XOR/线性回归收敛测试 |
| SDK 入口 | `basalt.Basalt` | `create/randn/tensor/linear/conv2d/activation/optimizer/graph`，`AutoCloseable` |
| PyTorch API | `basalt.torch` | `Tensor` / `Module` / `Sequential` / `Loss` / `Optimizer`，eager 执行 + 自动求导 |
| 图引擎 | `basalt.torch.Graph` | `backward/clear/size/stats`，显式控制原生计算图 |
| 原生 CNN | `Conv2d` / `MaxPool2d` / `BatchNorm2d` | im2col + GEMM、反向传播、训练/推理模式、running stats |
| Transformer | `MultiheadAttention` / `LayerNorm` / `GELU` / `Embedding` | 多头、causal mask、残差、pre-norm block |
| 类型安全管线 | `Sequential<I,O>` | 布局链在**编译期**检查；`Conv2d → Linear` 直接编译失败 |
| 可观测性 | `basalt.NativeQueue` JFR event | 每次队列执行的 op 数、字节数、耗时、调用栈；Python 无对应常开设施 |
| 持久化 | `ModelCodec` / `stateDict` | `.bsl` v2、CRC32C、全模型 bit-exact 往返 |

---

## 2. 构建与验证

```bash
# SDK 构建：编译 + JUnit 5 + 标准 jar + Fat Jar + 源码包
mvn clean install

# 传统验证入口：编译 + legacy 套件 + 编译期拒绝用例（有 cargo 时先构建并测试 Rust 内核）
./verify.sh

# 只构建 Rust 内核：cargo test --release + cargo build --release，产物在 build/native/
./native-build.sh

# 完整基准（会先执行 verify），输出 build/bench/*.csv 并调用 Python 绘图
./benchmark.sh

# 没有 Rust 工具链时，仍可验证纯 Java 回退路径
BASALT_SKIP_NATIVE=1 ./verify.sh
```

报告同时提供 Markdown 与自包含 HTML：

- `report/Basalt_ARCHITECTURE_AND_BENCHMARK.md`
- `report/Basalt_ARCHITECTURE_AND_BENCHMARK.html`（图片内嵌 base64，可离线打开）
- `report/Basalt_VS_SKLEARN.md` / `.html`（与 scikit-learn 的主客观对比）

与 scikit-learn 的对比需要 Python + sklearn + matplotlib：

```bash
./compare.sh
```

该脚本用同一份 BasaltD 数据分别驱动 Basalt 与 sklearn（10 个模型/任务）与并发基准
（虚拟线程 vs Python 进程/线程），输出 `build/compare/*.csv`、
`report/figures/fig_vs_*.png` 与 `report/vs_tables.md`。
v4 结果（同一份 BasaltD，5 次中位数）：随机森林 **0.17×**、Ridge **0.45×**、Lasso **0.60×**、
线性 SVM **0.72×**、PCA **0.84×**、K-Means **0.87×**、高斯朴素贝叶斯 **0.41×**、
直方图提升回归 **0.98×**、直方图提升分类 **1.03×**（处于 ±10% 运行噪声内）。
10 项全部持平或反超；I/O 编排快 Python 进程 80×+，JFR 生产剖析零开销（cProfile +100% 以上），
能力矩阵 87% vs 67%。

`verify.sh` 的行为：

1. 若存在 `cargo` 且未设置 `BASALT_SKIP_NATIVE=1`，执行 `native-build.sh`：
   `cargo test --release`（16 个 Rust 单元测试）后构建动态库；
2. 以 `--release 21 --add-modules jdk.incubator.vector -Xlint:all` 编译主代码；
3. 扫描编译日志，**除 JDK 自身的 incubating module 提示外，任何 warning 都视为失败**；
4. 运行 `TestSuite`、`IndustrialTestSuite`、`SequenceTestSuite`、`BreadthTestSuite`、
   `NativeTestSuite`（14 组原生数值/生命周期/并发测试）；
5. 逐个编译 `src/test/compile-fail`，要求**必须编译失败**。

Python 图表需要 `matplotlib` 与 `numpy`；缺失时基准仍会完成，仅跳过绘图。

---

## 3. 快速开始

### 3.1 经典监督学习

```java
Matrix x = Matrix.fromRows(new double[][]{{-3}, {-2}, {-1}, {1}, {2}, {3}});
double[] y = {0, 0, 0, 1, 1, 1};

RandomForest forest = RandomForest.builder()
        .trees(200).maxDepth(8).threads(8).oob(true)
        .build().fit(x, y);

double[] labels = forest.predict(x);
Matrix probabilities = forest.predictProbability(x);
double oob = forest.oobScore();
```

### 3.2 直方图梯度提升（XGBoost / LightGBM 核心）

```java
HistogramGradientBoosting model = HistogramGradientBoosting.builder()
        .algorithm(BoostingAlgorithm.LIGHTGBM)   // 或 XGBOOST
        .objective(BoostingObjective.LOGISTIC)
        .iterations(300)
        .learningRate(0.05)
        .maxDepth(6)
        .maxLeaves(31)
        .bins(64)
        .featureFraction(0.8)
        .validationFraction(0.2)
        .earlyStoppingRounds(20)
        .threads(8)
        .build()
        .fit(trainX, trainY);

double[] probability = model.predictProbability(testX);
double[] importance = model.featureImportances();
```

### 3.3 聚类与降维

```java
KMeans kmeans = KMeans.builder()
        .clusters(8).restarts(4).threads(8).seed(42)
        .build().fit(x);

PCA pca = PCA.builder()
        .components(16)
        .solver(EigenSolver.AUTO)   // p > n 时自动走 Gram 对偶
        .whiten(true)
        .build().fit(x);
Matrix reduced = pca.transform(x);
Matrix restored = pca.inverseTransform(reduced);
```

### 3.4 CNN

```java
CNN cnn = NeuralNetwork.builder()
        .add(Conv2D.builder().filters(8).kernelSize(3, 3)
                .padding(Padding.SAME).activation(Activation.RELU).build())
        .add(MaxPooling2D.builder().poolSize(2, 2).build())
        .add(Flatten.builder().build())
        .add(Dense.builder().units(10).activation(Activation.SOFTMAX).build())
        .loss(Loss.CATEGORICAL_CROSSENTROPY)
        .optimizer(Optimizer.ADAM.withLearningRate(1e-3))
        .build();

cnn.fit(images, oneHotTargets, 20);
```

`NeuralNetwork.builder()` 返回 `ImageBuilder`，只有经过 `Flatten` 才能进入 `VectorBuilder`；
把 `Dense` 直接接在图像层之后**无法通过编译**。

### 3.5 序列模型与注意力

```java
SequenceNetwork network = SequenceNetwork.builder()
        .add(LSTM.builder().hiddenSize(64).seed(1).build())
        .add(SelfAttention.builder().dimension(64).causal(false).build())
        .pooling(SequencePool.Pooling.LAST)
        .head(Dense.builder().units(3).activation(Activation.SOFTMAX).build())
        .loss(Loss.CATEGORICAL_CROSSENTROPY)
        .optimizer(Optimizer.ADAM.withLearningRate(0.005))
        .gradientClip(5.0)
        .build();

TrainingHistory history = network.fit(features, targets, 100, 32); // epochs, batchSize
Tensor<BC> probability = network.predict(features);
```

`Tensor<BSF>` 的布局由泛型固定，`LSTM.forward(Tensor<BCHW>)` 是编译错误，而不是运行时异常。

### 3.6 数据管线

```java
DataLoader.write(path, features, labels);                 // BasaltD v1
Normalization norm = DataLoader.estimateNormalization(path, featureCount);

try (DataLoader.MappedBatchStream stream = DataLoader.builder(path)
        .featureCount(featureCount)
        .batchSize(1024)
        .shuffle(true)
        .normalization(norm)
        .openMapped()) {
    DataLoader.Batch batch = stream.batch();             // 复用同一对象
    while (stream.next()) {
        double[] x = batch.features();                    // 复用同一数组
        double[] y = batch.labels();
        // train...
    }
    stream.rewind();                                     // 下一 epoch，重新洗牌
}
```

`next()` 不分配 Batch、特征数组、标签数组或字节缓冲区；映射流通过绝对
`DoubleBuffer.get(index, dst, off, len)` 聚合行，一个 epoch 内零系统调用。

### 3.7 原生张量与自动求导（专家层）

```java
try (NativeContext ctx = NativeContext.create(8)) {
    NativeTensorHandle x = ctx.of(features, false, 256, 32);   // [batch, features]
    NativeTensorHandle w = ctx.random(new int[]{32, 64}, 42, 0.15f, true);
    NativeTensorHandle b = ctx.parameter(64);
    NativeTensorHandle z = ctx.zeros(256, 64);
    NativeTensorHandle a = ctx.zeros(256, 64);
    NativeTensorHandle h = ctx.zeros(256, 64);
    NativeTensorHandle target = ctx.of(labels, false, 256, 64);
    NativeTensorHandle loss = ctx.zeros(1);

    CommandQueueBuilder queue = new CommandQueueBuilder(ctx);
    queue.matMul(x, w, z)       // z = x W
         .add(z, b, a)          // a = z + b（b 按行广播）
         .relu(a, h)            // h = relu(a)
         .mse(h, target, loss)  // loss = mean((h - target)^2)
         .execute();            // 4 个算子，1 次 JNI

    ctx.backward(loss);         // 拓扑排序 + 逆序求导
    w.adam(1e-3f, 0.9f, 0.999f, 1e-8f, 0f, 1);
    b.adam(1e-3f, 0.9f, 0.999f, 1e-8f, 0f, 1);
    w.zeroGrad();
    b.zeroGrad();
    ctx.clearGraph();           // 下一轮独立前向之前必须清图
}
```

`NativeMLP` 是基于同一套 API 的完整参考实现：XOR（2-12-1）与线性回归都在
`NativeTestSuite` 中做收敛断言，梯度用中心差分逐参数校验。

三层 API 的边界是刻意设计的：

| 层级 | 入口 | 适用人群 |
|---|---|---|
| 整体 | `Basalt` 门面、`Estimator.fit/predict` | 新手：一行完成训练/预测/交叉验证 |
| 模块 | `RandomForest.builder()`、`HistogramGradientBoosting.builder()` | 老手：按模块调参、替换求解器、控制线程 |
| 原子 | `NativeContext` + `CommandQueueBuilder` + `nativeBackward` | 大师：手写算子序列、自定义损失与优化器 |

### 3.8 PyTorch 风格 eager API（`basalt.torch`）

```java
try (NativeContext ctx = NativeContext.create(8)) {
    Sequential<BCHW, BC> net = Sequential.<BCHW>start()
            .add(new Conv2d(ctx, 1, 8, 3, 3, 1, 1, 1, 1, true, 42))
            .add(new Activation<BCHW>(Activation.Kind.RELU))
            .add(new MaxPool2d(2))
            .add(new Flatten())
            .add(new Linear(ctx, 8 * 14 * 14, 10));

    CrossEntropyLoss lossFunction = new CrossEntropyLoss();
    AdamW optimizer = new AdamW(net.parameters(), 1e-3f);

    for (int step = 0; step < 100; step++) {
        Tensor logits = net.forward(images);          // [N, 10]
        Tensor loss = lossFunction.loss(logits, labels);
        loss.backward();                              // Rust 拓扑排序 + 逆序求导
        optimizer.step();                             // AdamW，原生 bias correction
        optimizer.zeroGrad();
        ctx.clearGraph();                             // 下一轮独立图
    }
}
```

- `Tensor` 是动态形状的 eager 张量：`matMul`、`conv2d`、`maxPool2d`、`batchNorm2d`、
  `layerNorm`、`softmax`、`crossEntropy`、`embedding`、`permute`、`reshape`、`dropout`。
- `Module<I,O>` 的布局标签是**编译期**的：`Conv2d` 是 `BCHW → BCHW`，`Flatten` 是
  `BCHW → BC`，`Linear` 是 `BC → BC`。`Sequential.<BCHW>start().add(new Linear(...))`
  **无法通过编译**（`InvalidTypedPipeline.java` 是强制检查用例）。
- `MultiheadAttention` / `TransformerBlock` 由同一套公开 API 组合：多头 reshape/permute、
  batched matmul、causal mask、softmax、LayerNorm、GELU、残差。
- `stateDict()` / `loadStateDict()` 提供权重往返；`Optimizer.clipGradNorm()` 在 Rust 侧做
  全局 L2 裁剪。

### 3.9 模型持久化

```java
ModelCodec.save(Path.of("model.bsl"), network);
SequenceNetwork loaded = ModelCodec.load(Path.of("model.bsl"), SequenceNetwork.class);
double[] prediction = loaded.predict(features).data();
```

支持 `CNN`、`HistogramGradientBoosting`、`RandomForest`、`KMeans`、`PCA`、
`GaussianNaiveBayes`、`LSTM`、`GRU`、`VanillaRNN`、`SelfAttention`、`SequenceNetwork`。
加载后的预测与保存前**逐比特一致**（测试覆盖）。

---

## 4. 原生 CPU 内核架构

### 4.1 所有权与生命周期

```
Java: NativeTensorHandle ──long ptr──┐
                                     │  JNI
Rust: Context                         ▼
        ├── SlotMap<TensorKey, TensorSlot>   ← 唯一所有者
        │       └── Arc<RwLock<NativeTensor>> ← Java 指针指向的分配
        └── Graph (Mutex)
```

- 张量内存是 `Vec<f32>`，梯度 `Vec<f32>` **惰性分配**；推理张量永远不付第二份缓冲区。
- `Arc::into_raw` 把一份强引用交给 Java；Context 的 slotmap 保留另一份用于查找与兜底释放。
- `nativeReleaseTensor` 先按地址查 slotmap，命中才 `Arc::from_raw` 并 drop。
  第二次释放查不到地址，返回 `1`（幂等），**不会 double free**。
- Java 侧 `NativeTensorHandle` 只持 `long`，实现 `AutoCloseable`；`Cleaner` 注册的 `State`
  用 `AtomicLong` 保证 `close()` 与 GC 兜底只会有一次真正释放。
- `NativeContext` 用读写锁划定生命周期：所有 JNI 调用持读锁，`close()` 持写锁。
  Tensor handle 强引用 context，因此 context 不可能先于 handle 被 GC。
- 释放图内张量会清空当前计算图，避免地址复用后旧节点仍然指向新张量。

没有 `Box::leak`，没有全局张量池。唯一的进程级结构是 context 注册表，
用于在 FFI 边界验证指针存活；它不持有任何张量数据。

### 4.2 指令队列

`CommandQueueBuilder` 把一批算子写进 `ByteBuffer.allocateDirect`，每条 32 字节：

| 偏移 | 类型 | 字段 |
|---:|---|---|
| 0 | `u8` | op code（1=MatMul, 2=Add, 3=ReLU, 4=Sub, 5=Mul, 6=Sigmoid, 7=Tanh, 8=MSE） |
| 1 | `u8[7]` | 对齐填充，写零 |
| 8 | `u64` | input A 的 native 指针 |
| 16 | `u64` | input B 的 native 指针（一元算子为 0） |
| 24 | `u64` | output 的 native 指针 |

Rust 用 `read_unaligned` 逐条解析，在**一次 JNI 调用**内顺序执行并追加计算图节点。
Java 不循环调 JNI，也不把张量数据搬进 `float[]`；只有显式 `data()` / `grad()` 才发生拷贝。

### 4.3 SIMD 矩阵乘

前向 GEMM 严格采用 `i -> k -> j`，`j` 最内层：

```text
for i in 0..M          // 每次处理 6 行，6 行 C 常驻 L1
  for k in 0..K        // 广播 a[i,k]
    for j in 0..N      // _mm256_loadu_ps / _mm256_fmadd_ps，8 路并行
      C[i,j] += a[i,k] * B[k,j]
```

- `B[k, :]` 在 6 行之间复用；`C` 的 6 行约 24 KiB，和 `B` 的一行一起落在 32 KiB L1 内。
- AVX2/FMA 在运行时检测；非 x86 或缺少 FMA 时自动走标量路径。
- 512³ f32 实测 **115 GFLOP/s**、1024³ **109 GFLOP/s**；相对 Java Vector API 的
  double GEMM 分别为 **3.2× / 2.53×**，256³ 为 **8.4×**。
- 反向传播使用 `gemm_nt`（`dA = dC·Bᵀ`）与 `gemm_tn`（`dB = Aᵀ·dC`），同样按行/列分块并行。

张量引擎使用 **f32**；经典估计器（Ridge/KMeans/GBDT/森林）仍使用 **f64**。
这是刻意的分工：神经网络训练对 f32 足够，经典模型的数值证明与持久化保持 f64。

### 4.4 自动求导

- 每个队列算子追加一个 `Node { op, inputs, output, attrs }`；
  `attrs` 记录 MatMul 的 `(m,k,n)` 或广播的输入形状。
- `nativeBackward(root)` 从 root 反向收集可达子图，用 **Kahn 算法**做拓扑排序；
  逆序遍历，按 `op_code` 调用对应导数，梯度累加到每个输入的 `grad`。
- 支持 `MatMul`、`Add`、`Sub`、`Mul`、`ReLU`、`Sigmoid`、`Tanh`、`MSE`；
  `Add/Sub/Mul` 支持 NumPy 风格的尾部广播，反向时对广播维求和。
- 图在写入前校验：禁止 `output == input`、禁止重复生产同一张量、禁止覆盖已被消费的张量。
  这三条把“原地算子”和“地址复用”这类静默错误变成带上下文的异常。
- 一轮训练结束后 Java 调 `clearGraph()`；`NativeMLP` 在每个 mini-batch 后执行
  `backward -> Adam -> zeroGrad -> clearGraph`。

### 4.5 安全与回退

| 风险 | 处理 |
|---|---|
| 野指针 / 已释放 handle | context 注册表 + slotmap 双重查找，查不到抛 `IllegalStateException` |
| 双重释放 | Java `AtomicLong` + Rust 幂等 remove；两层都只允许一次真正 drop |
| GC 漏释放 | `Cleaner` 兜底；`NativeTestSuite` 创建 200 个 handle 后强制 GC 并断言归零 |
| context 先关 | 读写锁 + handle 强引用 context；关闭后 handle 操作立即失败 |
| Rust panic 跨 FFI | 每个入口 `catch_unwind`，转成 Java 异常，绝不让 unwind 穿过 JNI |
| 原生库缺失 | `NativeLibrary.available()==false`，经典模型走纯 Java 路径；张量 API 显式报错 |
| 形状错误 | 在写输出前校验，失败不污染目标张量 |

### 4.6 原生基准

`build/bench/native_gemm.csv` 与 `native_mlp.csv`（`./benchmark.sh native`）：

| 规模 | Java Vector API (double) | Rust f32 | 加速 |
|---|---:|---:|---:|
| 256³ | 3.16 ms | 0.37 ms | **8.4×** |
| 512³ | 7.45 ms | 2.33 ms | **3.2×** |
| 1024³ | 49.60 ms | 19.62 ms | **2.53×** |

`NativeMLP`（4096×32，32-64-32-8，20 epochs，batch 256，Adam）：
**20.0 ms/epoch**，最终 MSE **4.2e-4**，4096 行推理 **7.9 ms**。

---

## 5. 现代训练栈：把 PyTorch 的优势抄过来

### 5.1 原生算子矩阵（21 个）

| 类别 | 算子 | 反向 |
|---|---|---|
| 线性代数 | `MatMul`、`BatchedMatMul`、`Permute`、`Reshape` | ✅ |
| 逐元素 | `Add`、`Sub`、`Mul`（广播）、`ReLU`、`Sigmoid`、`Tanh`、`GELU`、`SiLU` | ✅ |
| 归一化 | `LayerNorm`、`BatchNorm2d`（running stats / train-eval） | ✅ |
| 卷积/池化 | `Conv2d`（im2col + GEMM，stride/padding）、`MaxPool2d` | ✅ |
| 损失 | `MSE`、`CrossEntropy`（fused softmax + NLL） | ✅ |
| 正则/索引 | `Dropout`（确定性 mask）、`Embedding`（scatter-add） | ✅ |
| 注意力 | `Softmax`（末轴）、`MultiheadAttention`（组合算子） | ✅ |

所有算子共享同一张计算图、同一套 `Arc<RwLock<NativeTensor>>` 存储和同一个 128 字节指令格式。

### 5.2 两个端到端例子

**CNN**（8×8 单通道，Conv2d → ReLU → MaxPool2d → Flatten → Linear → CrossEntropy）：

```text
120 个 full-batch AdamW step，64 个样本，训练准确率 100%，loss 1.7e-4
```

**Transformer**（2 层 pre-norm block，dim=8，4 heads，causal mask，GELU FFN）：

```text
40 个 AdamW step，MSE 0.7737 → 0.0053
```

两者都只用 `basalt.torch` 的公开 API，没有任何测试专用捷径。

### 5.3 信息差：Python 追不了的四件事

1. **编译期布局管线**：`Sequential<I,O>` 用 Java 泛型把 `BCHW → BC → BC` 的链路钉死。
   PyTorch 只能在运行时抛 shape mismatch；Basalt 在 `javac` 阶段就拒绝。
   强制用例：`src/test/compile-fail/jsl/tests/InvalidTypedPipeline.java`。
2. **JFR 常开事件流**：`basalt.NativeQueue` 记录每次队列执行的 op 数、字节数、耗时和完整调用栈。
   JFR 是 JVM 内置的采样式剖析器，生产可热附加；Python 的 cProfile 是插桩式，吞吐减半。
3. **零拷贝 + 无 GIL**：Java 只持 `long`，32/128 字节指令直接读 `ByteBuffer.allocateDirect`；
   Rust 用 `rayon` 在原生线程池并行。没有 Python 解释器、没有 GIL、没有 per-op FFI 编组。
4. **Rust 所有权 + Java 生命周期**：`Arc<RwLock<>>` + slotmap + `Cleaner` + 读写锁，
   双重释放/野指针/GC 漏释放都有测试。动态语言给不了这种内存安全保证。

### 5.4 我们抄了 PyTorch 什么

- eager execution：每个 `Tensor` 操作立即执行并记录图节点；
- 动态计算图 + 反向传播：Kahn 拓扑排序、梯度累加、广播反向；
- `nn.Module` 组合：`Sequential`、`Linear`、`Conv2d`、`BatchNorm2d`、`LayerNorm`、
  `Dropout`、`Embedding`、`MultiheadAttention`、`TransformerBlock`；
- 优化器：SGD（weight decay）、AdamW（bias correction + decoupled weight decay）、
  全局 L2 梯度裁剪；
- 损失：MSE、fused CrossEntropy；
- `stateDict()` / `loadStateDict()` 权重往返；
- 动态 shape、NumPy 风格广播、`reshape`/`permute`/`flatten`。

### 5.5 还没抄的（诚实边界）

- **CPU only**：没有 CUDA/Metal/分布式；`rayon` 只吃本机核心。
- **无稀疏/量化/AMP**：CSR、int8、f16/bf16 未实现；张量引擎统一 f32。
- **Conv2d 简化**：groups=1、dilation=1；MaxPool2d 无 padding。
- **无 `torch.compile` 级融合**：有指令批处理和图捕获，但没有算子融合编译器；
  `Tensor` 每个 op 一次 JNI，专家可用 `CommandQueueBuilder` 手工批处理。
- **单流图**：同一 context 的图由锁串行化；并发训练用独立 context。

这些边界不影响“现代 PyTorch 级 API + 原生 autograd + CNN/Transformer”的定位，
但决定了它现在不是 CUDA 训练框架的替代品。

---

## 6. `.bsl` 容器格式（v2）

```
偏移    类型      字段
0       int       magic = 0x42534C54 ("BSLT")
4       int       version = 2
8       int       kind (稳定枚举码，非 ordinal)
12      int       payload length
16      long      CRC32C(payload)
24      byte[]    payload
```

- 保存：一次性构造 `ByteBuffer`，单次 `FileChannel.write`；
- 加载：`FileChannel.map(READ_ONLY)`，CRC32C 直接在映射区上计算，**不复制 payload**；
- 校验：magic、版本、长度与文件大小一致、CRC、枚举范围、数组长度上界、树深度上界；
- 失败统一抛出 `ModelFormatException`（非受检），不会把损坏文件解释为合法模型。

---

## 7. 类型安全与失败快速

| 机制 | 表现 |
|---|---|
| `Tensor<S extends Layout>` | `BCHW` / `BC` / `BSF` / `BHW` 布局在编译期绑定 |
| 分阶段 Builder | `ImageBuilder.add(Dense)` 不存在该方法 |
| `Layer<I, O>` | `LSTM.forward(BCHW)` 类型不匹配 |
| `Estimator` | 所有经典模型共享 `fit(Matrix, double[])` / `predict(Matrix)` |
| 枚举 | `Activation`、`Loss`、`Optimizer`、`EigenSolver`、`TreeTask`、`BoostingAlgorithm` |
| 参数校验 | 维度、有限值、正数、枚举组合在构造/拟合入口检查 |
| 状态机 | `backward` 前必须有 `forward`；`update` 前必须有 `backward` |

编译期拒绝用例位于 `src/test/compile-fail/`，由 `verify.sh` 强制检查。

---

## 8. 目录结构

```
src/main/java/jsl/
  core/   Tensor, Matrix, MatMul, VectorMath, 模型 I/O 原语
  ml/     经典估计器、直方图树、提升、聚类、PCA、朴素贝叶斯
  nn/     Layer, Dense, Conv2D, LSTM, GRU, RNN, SelfAttention, SequenceNetwork
  data/   DataLoader (BasaltD), Normalization
  io/     ModelCodec (.bsl v2)
  nativebridge/  NativeContext, NativeTensorHandle, CommandQueueBuilder, NativeMLP
  torch/       Tensor, Module, Sequential, Conv2d, MultiheadAttention, AdamW, JFR event
native/src/
  lib.rs       模块声明
  api.rs       JNI 张量/队列/反向传播入口
  engine.rs    128B 指令执行 + 全算子反向传播
  tensor.rs    NativeTensor + slotmap 存储
  graph.rs     Op/Node/Kahn 拓扑排序
  ops.rs       GEMM/广播/一元算子/MSE/优化器
  nn_ops.rs    Conv2d/MaxPool/LayerNorm/BatchNorm/Softmax/CE/Embedding 等
  simd.rs      AVX2/FMA f32 核
  context.rs   Context、线程池、Adam 状态、梯度裁剪
  classic.rs   原有 f64 经典核 + 交错直方图桶
  fastmath.rs  Ridge/GNB/KMeans/双中心距离
  histogram.rs 持久化直方图会话
  binning.rs   原生分位数分箱
src/test/java/jsl/tests/
  TestSuite            经典数值/收敛
  IndustrialTestSuite  树模型/聚类/降维/数据/持久化
  SequenceTestSuite    BPTT/注意力/序列训练/梯度消失
  BreadthTestSuite     广度/门面/持久化
  NativeTestSuite      张量数值/梯度检验/生命周期/并发
  TorchTestSuite       eager 算子梯度检验 + CNN/Transformer 收敛
  BasaltBenchmark         CSV 基准
src/test/compile-fail/  必须编译失败的用例（含 InvalidTypedPipeline）
report/                 报告、图表生成脚本、figures/
```

---

## 9. 设计约束与已知边界

1. **稠密内存模型**：`Matrix`/`Tensor` 使用单个 `double[]`，上限约 20 亿元素；稀疏场景未优化。
2. **PCA 复杂度**：Jacobi 为 `O(p^3)` 级旋转，Gram 为 `O(n^2 p)` 且限制 `n ≤ 20 000`；
   超大特征空间应改用随机化 SVD（未实现）。
3. **GBDT 目标**：当前支持平方误差与二分类逻辑损失；多分类 softmax 未实现。
4. **序列训练**：`SequenceNetwork` 为分类头设计；序列标注（逐时间步输出）需要额外的输出层。
5. **注意力**：单头、无位置编码、无 dropout；`causal` 仅做掩码，不改变参数形状。
6. **持久化**：v2 不兼容 v1；`ModelCodec` 一次性构造 payload，超大模型（>1 GiB）会受数组上限约束。
7. **基准**：非 JMH，使用中位数与预热；结果只对当前机器/JDK 有效，运行间波动约 ±10%。
8. **精度分层**：原生张量引擎为 f32；经典估计器保持 f64。跨层比较需显式转换，不做隐式降精度。
9. **计算图生命周期**：一轮独立前向/反向结束后必须 `clearGraph()`；同一张量在同一图中只能被生产一次，
   原地算子会被拒绝。这是为了保证反向传播的 DAG 语义。
10. **原生可选**：经典模型在动态库缺失时完整回退 Java；`basalt.nativebridge` 的专家 API 需要原生库。
11. **HGB 直方图**：会话路径使用 f32 交错桶累加部分和，最终增益扫描提升到 f64；参考数据上
    精度与 f64 路径一致（0.7981 / 0.8867），极端大样本场景可用 Java f64 路径对照。
12. **线程模型**：每个 `NativeContext` 持有按并行度缓存的 rayon 线程池；多个 context 并发时
    线程数会叠加，生产环境应共享 context 或限制线程数。
13. **单流图**：同一 context 的计算图是单流（队列执行与 backward 由图锁串行化）；
    并发训练/推理请为每个线程创建独立 context，或只做无图推理。
14. **队列非事务**：一批指令中途失败时，已执行的算子与图节点不会回滚；
    捕获异常后应 `clearGraph()` 并丢弃该批输出。
15. **Torch 边界**：CPU only；Conv2d 目前 groups=1、dilation=1，MaxPool2d 无 padding；
    无稀疏/量化/混合精度/分布式，也没有 `torch.compile` 级算子融合编译器。

---

## 10. 许可

本仓库为教学/研究用途的独立实现，未复制任何第三方库代码。

---

## 11. v3.4.0 三期演进全景（Tribuo / DL4J / scikit-learn 对标升级）

### 第一阶段：夯实工程底座（Phase 1）
1. **JNI / Rust 底层核心工程强化**
   - **双重内存安全 (`Cleaner` + `AutoCloseable`)**：`NativeContext`、`NativeTensorHandle` 与 `NativeKernels.HistogramSessionHandle` 均支持 `try-with-resources` 确定性释放与 `Cleaner` 兜底回收。
   - **代数句柄校验（Generation-Tagged Handle）**：原生层句柄采用 `(generation << 32) | slot_index` 编码，释放后 slot 递增代数，彻底杜绝悬垂指针与 ABA 复用引发的 JVM 崩溃。
   - **跨平台预编译与 CI/CD**：`NativeKernels` 自动识别 `windows-x86_64`、`linux-x86_64`、`linux-aarch64`、`macos-x86_64`、`macos-aarch64` 并从 Classpath 提取原生库；配备 `.github/workflows/ci-release.yml` 多平台自动构建流水线。
   - **线程安全与异常体系**：`CommandQueueBuilder` 全方法同步化保护；JNI 边界捕获所有 panic 并映射为结构化异常体系（`BasaltExceptionCode`、`BasaltParameterException`、`BasaltShapeException`、`BasaltMemoryException`、`BasaltUnsupportedOperationException`）。
2. **ML 基础设施工具（对标 `sklearn.preprocessing` / `model_selection` / `pipeline`）**
   - **数据预处理**：`Transformer` 接口、`StandardScaler`、`MinMaxScaler`、`OneHotEncoder`、`LabelEncoder`、`SimpleImputer`（支持 `MEAN` / `MEDIAN` / `MOST_FREQUENT` / `CONSTANT`），全部支持 `inverseTransform` 与 `.bsl` 序列化。
   - **模型选择与超参搜索**：`DataSplit`（`Basalt.trainTestSplit` 含分层抽样）、`KFold`、`StratifiedKFold`、`GridSearchCV`、`RandomizedSearchCV`。
   - **Pipeline 管线**：`basalt.ml.Pipeline` 支持任意 `Transformer` 序列串联终端 `Estimator`，一键完成训练、推理、评分及 `.bsl` 原子二进制持久化。
3. **质量保障体系**
   - 9 大测试套件（含 `Phase1TestSuite`、`Phase2TestSuite`、`Phase3TestSuite`）+ 6 个编译期安全拒绝测试 + 2,000 次循环张量分配/释放零泄漏验证 + `./benchmark.sh` 可复现基准。

### 第二阶段：补齐核心能力（Phase 2）
1. **经典 ML 算法扩展**
   - **核方法**：`Kernel`（`LINEAR` / `RBF` / `POLY` / `SIGMOID`）、非线性 `SVC`（SMO 求解器 + Platt 概率校准）、`KernelRidge`（Cholesky 对偶求解）、`KernelPCA`（双中心核特征分解）。
   - **无监督学习**：`GaussianMixture`（全协方差 EM + log-likelihood / AIC / BIC）、`AgglomerativeClustering`（`WARD` / `COMPLETE` / `AVERAGE` / `SINGLE`）、`IsolationForest`（异常分数与离群点检测）、`KernelDensity`（Gaussian / Tophat / Epanechnikov 核密度估计）。
   - **特征工程**：`SelectKBest`（`ANOVA_F` / `PEARSON_CORR` / `MUTUAL_INFO`）、`RFE`（递归特征消除）、`PolynomialFeatures`（任意阶多项式与交互项）、`KBinsDiscretizer`（等宽 / 分位数分箱，支持序号与独热输出）。
2. **深度学习能力升级**
   - **算子与层**：`Conv2d` 支持分组卷积（`groups`）与空洞卷积（`dilation`）；新增 `DepthwiseSeparableConv2d`、`GroupNorm`；新增 `L1Loss`、`SmoothL1Loss`、`BCEWithLogitsLoss` 及 `LEAKY_RELU`、`ELU` 激活函数；完善 `Dropout` / `BatchNorm2d` 的 `train()` / `eval()` 状态切换。
   - **DAG 计算图 (`ComputationGraph`)**：突破 `Sequential` 线性限制，支持残差连接（`addAddVertex` / `addSubVertex` / `addMulVertex`）及多输入多输出拓扑排序前向与反向传播。
   - **训练基础设施**：`LRScheduler`（`CosineAnnealingLR`、`StepLR`）、全局梯度裁剪 `clipGradNorm`、梯度累积 `Optimizer.stepAccumulated(accumulationSteps)`、以及包含模型权重+优化器动量+调度器状态的 `Checkpoint` 断点续训。
3. **生态互通（ONNX 导入与导出）**
   - `OnnxModel` 与 `OnnxConverter` 提供零外部依赖的 `onnx.proto3`（IR v8, Opset 17）解析与编码器，支持导出 Basalt 经典模型/管线与深度学习模块至标准 `.onnx` 文件（可直接被 Python `onnxruntime` 加载推理，误差 `< 2e-6`），并支持将外部 `.onnx` 模型直接加载进 JVM 原生执行。

### 第三阶段：打造差异化竞争力（Phase 3）
1. **模型可解释性（Explainable AI）**
   - `FeatureImportance`：树模型杂质重要度、线性模型归一化权重重要度、以及模型无关的置换特征重要度（`permutationImportance`）。
   - `ShapExplainer`：精确 LinearSHAP 与基于联盟采样的 TreeSHAP / KernelSHAP，严格满足局部可加性公理 $\sum_{j=1}^p \phi_j(x) + \phi_0 = f(x)$。
   - `LimeExplainer`：局部指数核加权 Ridge 代理模型，输出局部特征归因权重与局部拟合优度 $R^2$。
2. **可观测性增强（JFR + 标准指标输出）**
   - 新增 JFR 事件：`NativeQueueEvent`、`OperatorExecutionEvent`、`NativeMemoryEvent`。
   - `BasaltMetrics`：无锁记录算子调用次数、耗时分位数（p50 / p95 / p99）、吞吐量（ops/s、elements/s）与堆外/JVM 内存占用，支持一键导出 JSON 与 Prometheus 文本格式。
