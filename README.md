# FactorBarNet

FactorBarNet 是一个面向加密货币现货与永续合约的多周期 K 线直接预测项目。

它不沿用 Kronos 的 tokenizer / token generation 路线，而是直接使用历史 OHLCV 与外生因子预测未来 K 线路径：

```text
历史 OHLCV + 历史因子
        ↓
Bar Encoder + Factor Encoder
        ↓
Gated / FiLM Fusion
        ↓
Multi-Horizon K-line Decoder
        ↓
未来 1/5/15/30 根 K 线与完整 30-step path
```

## 当前状态

当前项目只包含设计文档和最小工程骨架，尚未实现训练代码。

建议先阅读：

| 目标 | 文档 |
| --- | --- |
| 理解完整架构设计 | [docs/FACTORBARNET_DESIGN_SPEC.md](docs/FACTORBARNET_DESIGN_SPEC.md) |
| 理解 v1 因子 schema | [docs/FACTOR_SCHEMA_V1.md](docs/FACTOR_SCHEMA_V1.md) |
| 理解实验和评测计划 | [docs/EXPERIMENT_AND_EVALUATION_PLAN.md](docs/EXPERIMENT_AND_EVALUATION_PLAN.md) |

## 核心目标

- 直接预测未来 30 根 1min K 线，而不是预测 token 或单一收益值。
- 从预测路径中同时得到 h1 / h5 / h15 / h30 结果。
- 使用统一的 5/15/30 技术因子窗口，并支持现货与永续合约特有因子。
- 用 K 线误差、方向命中率、IC / Rank-IC / ICIR 共同评估。
- 第一版优先采用 TCN + Gated Fusion，保证训练速度和可解释性。

## 暂定目录

```text
FactorBarNet/
├── docs/
│   ├── FACTORBARNET_DESIGN_SPEC.md
│   ├── FACTOR_SCHEMA_V1.md
│   └── EXPERIMENT_AND_EVALUATION_PLAN.md
├── src/
│   └── factorbarnet/
│       └── __init__.py
├── AGENTS.md
├── pyproject.toml
└── README.md
```
