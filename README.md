# SuperPnL

SuperPnL 是一个 PnL-first 预测模型研究项目。当前阶段先聚焦加密货币现货市场，后续可以扩展到永续、组合配置、执行成本建模和其他直接面向 PnL 的预测任务。

项目目标不再是单纯预测未来 K 线是否准确，而是学习一个可以被交易执行的净 edge：

```text
历史 OHLCV + 历史因子
        ↓
Bar Encoder + Factor Encoder
        ↓
Gated / FiLM Fusion
        ↓
Path Head + Alpha Head + Position Head
        ↓
现货目标仓位
        ↓
按固定成本假设计算后的 PnL
```

## 当前状态

当前项目已经包含从 OKX 拉取 1min 现货 K 线、生成训练特征、训练 baseline / 有因子模型、测试评估和回测报告的最小可运行流程。

建议先阅读：

| 目标 | 文档 |
| --- | --- |
| 理解完整 PnL-first 架构 | [docs/SUPERPNL_DESIGN_SPEC.md](docs/SUPERPNL_DESIGN_SPEC.md) |
| 理解特征 schema | [docs/FEATURE_SCHEMA.md](docs/FEATURE_SCHEMA.md) |
| 理解回测和评测计划 | [docs/BACKTEST_AND_EVALUATION_PLAN.md](docs/BACKTEST_AND_EVALUATION_PLAN.md) |
| 理解下游如何配置和使用模型 | [docs/DOWNSTREAM_USAGE.md](docs/DOWNSTREAM_USAGE.md) |
| 查看 Top20 / 12 个月实验结果 | [docs/TOP20_12M_EXPERIMENT_REPORT.md](docs/TOP20_12M_EXPERIMENT_REPORT.md) |
| 复现实验并理解 Top20 币池 | [docs/REPRODUCIBILITY_AND_TOP20_UNIVERSE.md](docs/REPRODUCIBILITY_AND_TOP20_UNIVERSE.md) |
| 给 BitPro 项目实现策略的提示词 | [docs/BITPRO_STRATEGY_PROMPT.md](docs/BITPRO_STRATEGY_PROMPT.md) |

## 核心目标

- 第一版只做现货，不做永续、杠杆、借币做空。
- 数据层只获取 1min K 线；策略 horizon 可配置，例如 `strategy_horizons=["5m", "15m"]`。
- 预测策略 horizon 上的可交易 edge，而不是只预测价格。
- 输出 `0..1` 的现货目标仓位，支持空仓、半仓、满仓。
- 当前阶段不把盘口、成本、流动性作为训练特征；回测使用固定成本配置，若设为 0 必须标注。
- 评测必须同时包含 no-trade、buy-and-hold、OHLCV-only 和有因子模型。
- 所有特征只能使用决策时刻 `t` 及以前的信息。

## 实验入口

下载数据在服务器执行：

```bash
python3 scripts/download_okx_spot_1m.py \
  --out /opt/bitpro/data/superpnl/okx_spot_1m_top20_365d \
  --top 20 \
  --days 365 \
  --workers 4 \
  --rate 8
```

本地训练与回测：

```bash
PYTHONPATH=src python3 scripts/run_superpnl_experiment.py \
  --raw-dir data/okx_spot_1m_top20_365d \
  --cache-dir data/cache/okx_spot_1m_top20_365d_l256_h5_15 \
  --out-dir outputs/superpnl_top20_365d \
  --lookback 256 \
  --horizons 5,15 \
  --feature-windows 5,15,30 \
  --epochs 5 \
  --samples-per-epoch 200000 \
  --batch-size 256 \
  --hidden-dim 128
```

`data/`、`outputs/`、checkpoint 和模型权重不会提交进仓库。

## 目录

```text
SuperPnL/
├── docs/
│   ├── SUPERPNL_DESIGN_SPEC.md
│   ├── FEATURE_SCHEMA.md
│   ├── BACKTEST_AND_EVALUATION_PLAN.md
│   ├── DOWNSTREAM_USAGE.md
│   ├── TOP20_12M_EXPERIMENT_REPORT.md
│   ├── REPRODUCIBILITY_AND_TOP20_UNIVERSE.md
│   └── BITPRO_STRATEGY_PROMPT.md
├── scripts/
│   ├── download_okx_spot_1m.py
│   ├── run_superpnl_experiment.py
│   └── plot_horizon_comparison.py
├── src/
│   └── superpnl/
│       ├── data.py
│       ├── metrics.py
│       ├── model.py
│       └── training.py
├── AGENTS.md
├── pyproject.toml
└── README.md
```
