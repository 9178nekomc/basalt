# 文档数据契约（docs_data schema）

`tools/build_api_docs.py` 渲染 HTML 时只做一件事：把**机械提取的签名**（`build/api/api.json`）
与**人工撰写的说明**（`tools/docs_data/gNN.json`）合并。签名永远来自源码扫描，说明永远来自
`docs_data`。这样文档不可能出现"签名写错"或"文档了一个不存在的参数"。

## 文件形态

每个分组一个 JSON，结构与 `build/api/briefs/gNN.json` **完全同构**，只是给每个
type 和每个 member 增加一个 `doc` 对象。

```json
{
  "group": "g01",
  "title": "门面与并发",
  "modules": [
    {
      "name": "jsl",
      "intro": ["模块级导语第一段", "第二段"],
      "types": [
        {
          "qualified": "jsl.JSL",
          "doc": {
            "summary": "一句话定位（≤60 字）",
            "body": ["第一段", "第二段"],
            "notes": ["使用约束 / 边界条件 / 性能特征"],
            "example": "try (JSL jsl = JSL.create()) {\n    ...\n}",
            "see": ["jsl.ml.JSL", "jsl.torch.Tensor"]
          },
          "members": [
            {
              "key": "create(int)",
              "doc": {
                "summary": "一句话（≤40 字）",
                "body": ["补充语义，可选"],
                "params": {"threads": "并行度；≤0 时取 CPU 核数"},
                "returns": "持有独立 NativeContext 的门面实例",
                "throws": {"IllegalArgumentException": "threads 为负"},
                "notes": ["关闭门面会级联释放其上下文内所有张量"]
              }
            }
          ]
        }
      ]
    }
  ]
}
```

## 硬性规则

1. `qualified`、`key`、`signature` 必须与骨架逐字节一致，不得新增或删除成员。
   渲染器按 `key` 匹配，匹配不上的说明会被丢弃并报错。
2. 只描述**源码里真实存在**的行为。参数含义从方法体、校验分支、调用方推断；
   推断不出来就写结构性事实（例如"见 {@link ...}"），不要编造取值范围。
3. 抛出的异常只能来自源码中真实出现的 `throw` / 显式校验；`throws` 子句为空时不要写。
4. 示例代码必须能对着当前 API 编译：构造器参数、方法名、枚举常量都要在骨架里存在。
5. 中文行文，术语统一：张量（tensor）、句柄（handle）、上下文（context）、估计器（estimator）、
   计算图（graph）、指令队列（command queue）。类型名、方法名、参数名保持英文原文。
6. 不在正文里写"本文档""我们"；用客观陈述句。

## 字段语义

| 字段 | 用途 | 渲染位置 |
|---|---|---|
| `summary` | 类型/成员的一句话定位 | 标题下第一行 |
| `body` | 段落数组 | 摘要之后的正文 |
| `notes` | 注意事项数组 | 蓝色提示框 |
| `example` | Java 代码 | 代码块 |
| `see` | 相关类型/方法 | "参见" |
| `params` | 参数名 → 说明 | 参数表第三列 |
| `returns` | 返回值说明 | 返回值行 |
| `throws` | 异常类型 → 触发条件 | 抛出异常表 |
