# Reproducibility And Top20 Universe

本文档用于让别人从零复现 SuperPnL 第一轮 OKX 现货 Top20 / 12 个月实验，并理解这次 Top20 币池的构成。

术语说明：本实验里的 Top20 指“下载当时 OKX 现货 `*-USDT` 中，剔除稳定币 base 后，按 24h 成交额排序并且 1min 历史覆盖满 365 天的前 20 个标的”。它不是固定不变的市场 Top20，也不是市值 Top20。

## 1. 复现实验摘要

| 项 | 值 |
| --- | --- |
| project | SuperPnL |
| exchange | OKX |
| market | spot |
| quote | USDT |
| bar_size | 1m |
| raw window | `2025-04-30 15:00:00 UTC` -> `2026-04-30 15:00:00 UTC` |
| symbols | 20 |
| rows per symbol | 525,601 |
| common timestamps | 525,596 |
| lookback | 256 |
| feature_windows | 5m, 15m, 30m |
| model variants | `ohlcv_tcn`, `full_feature_tcn` |
| baselines | `no_trade`, `buy_and_hold_equal_weight`, `naive_momentum` |
| cost assumption | 主实验为零成本：`fee=0bps`, `slippage=0bps` |

实验产物目录：

```text
/Users/jie.feng/wlb/SuperPnL/outputs/superpnl_top20_365d_l256_h5_15_hd64_e3/
/Users/jie.feng/wlb/SuperPnL/outputs/superpnl_top20_365d_l256_h30_hd64_e3/
```

原始数据和训练产物不提交进 git，均在 `.gitignore` 覆盖的目录中：

```text
data/
outputs/
artifacts/
checkpoints/
```

## 2. 环境准备

本地代码路径：

```bash
cd /Users/jie.feng/wlb/SuperPnL
```

Python 依赖来自 `pyproject.toml`：

```text
numpy
pandas
torch
```

语法检查命令：

```bash
PYTHONPATH=src python3 -m py_compile \
  scripts/download_okx_spot_1m.py \
  scripts/run_superpnl_experiment.py \
  src/superpnl/*.py
```

本次本地训练环境使用 Apple MPS。脚本会按顺序选择 `cuda`、`mps`、`cpu`，也可以用 `--device` 强制指定。

## 3. 数据下载复现

本地无法稳定连接 OKX，因此下载在服务器执行：

```bash
ssh root@47.79.36.92
```

服务器上执行：

```bash
python3 /opt/bitpro/superpnl/scripts/download_okx_spot_1m.py \
  --out /opt/bitpro/data/superpnl/okx_spot_1m_top20_365d \
  --top 20 \
  --days 365 \
  --workers 4 \
  --rate 8 \
  --sleep 0
```

脚本行为：

1. 调 OKX `/api/v5/public/instruments` 获取 live spot instruments。
2. 调 OKX `/api/v5/market/tickers` 获取 spot tickers。
3. 只保留 `*-USDT` 现货。
4. 剔除稳定币 base：`USDT, USDC, DAI, USDG, PYUSD, TUSD, FDUSD, EURT, RLUSD, USDE, USD1`。
5. 按 `volCcy24h` 降序排序。
6. 用 OKX `history-candles` 检查是否能覆盖 365 天 1min 历史。
7. 选满 Top20 后并发下载每个 symbol 的确认 K 线。
8. 每个 symbol 写成一个 `csv.gz` 文件，最后写 `metadata.json`。

本次下载完成后的固定币池：

```text
BTC-USDT, ETH-USDT, DOGE-USDT, SOL-USDT, XRP-USDT,
PEPE-USDT, TRX-USDT, XAUT-USDT, BIO-USDT, PENGU-USDT,
PI-USDT, ZKJ-USDT, TRUMP-USDT, SUI-USDT, FIL-USDT,
ADA-USDT, APE-USDT, CHZ-USDT, LINK-USDT, LTC-USDT
```

下载耗时约 73 分钟。服务器目录：

```text
/opt/bitpro/data/superpnl/okx_spot_1m_top20_365d/
├── metadata.json
└── csv/
```

同步回本地：

```bash
rsync -az --delete \
  root@47.79.36.92:/opt/bitpro/data/superpnl/okx_spot_1m_top20_365d/ \
  data/okx_spot_1m_top20_365d/
```

完整性检查：

```bash
python3 - <<'PY'
import gzip, json, os

root = "data/okx_spot_1m_top20_365d"
meta = json.load(open(os.path.join(root, "metadata.json")))
for sym in meta["symbols"]:
    path = os.path.join(root, "csv", f"{sym}.csv.gz")
    with gzip.open(path, "rt") as f:
        rows = sum(1 for _ in f) - 1
    print(sym, rows)
PY
```

期望每个 symbol 都输出：

```text
525601
```

## 4. 特征与标签复现

数据准备入口在：

```text
src/superpnl/data.py
```

默认输入：

```text
bar_inputs:
open_rel, high_rel, low_rel, close_rel, volume_z_30m, amount_z_30m

feature_inputs:
ret_5m/15m/30m
rsi_5m/15m/30m
vol_std_5m/15m/30m
ma_dev_5m/15m/30m
boll_z_5m/15m/30m
macd_5m_15m
macd_15m_30m
market_ret_5m/15m/30m
market_vol_5m/15m/30m
cross_section_ret_rank_5m/15m/30m
cross_section_vol_rank_5m/15m/30m
hour_sin/hour_cos
dayofweek_sin/dayofweek_cos
```

标签定义：

```text
label_h = log(open_{t+h+1} / open_{t+1})
```

含义：在 `t` 时刻决策，下一根 K 线开盘进入，在 horizon 后的下一根开盘退出。这样避免使用当前 bar 内不可成交的 close 作为入场价。

泄漏约束：

- 所有 rolling / EMA / rank 特征只用 `<= t` 的历史数据。
- train / val / test 按时间切分。
- 标准化均值和标准差只在 train split 上拟合。
- 截面 rank 只用同一时刻已经可见的 Top20 数据。
- Top20 币池由本次下载时刻确定，复现时应使用 `metadata.json` 固定币池，避免重新按未来成交额排序。

## 5. 训练复现

5m/15m 实验：

```bash
PYTHONPATH=src python3 scripts/run_superpnl_experiment.py \
  --raw-dir data/okx_spot_1m_top20_365d \
  --cache-dir data/cache/okx_spot_1m_top20_365d_l256_h5_15 \
  --out-dir outputs/superpnl_top20_365d_l256_h5_15_hd64_e3 \
  --lookback 256 \
  --horizons 5,15 \
  --feature-windows 5,15,30 \
  --epochs 3 \
  --samples-per-epoch 200000 \
  --batch-size 1024 \
  --hidden-dim 64 \
  --validation-samples 100000 \
  --fixed-fee-bps 0 \
  --fixed-slippage-bps 0 \
  --rebuild-cache
```

30m 实验：

```bash
PYTHONPATH=src python3 scripts/run_superpnl_experiment.py \
  --raw-dir data/okx_spot_1m_top20_365d \
  --cache-dir data/cache/okx_spot_1m_top20_365d_l256_h30 \
  --out-dir outputs/superpnl_top20_365d_l256_h30_hd64_e3 \
  --lookback 256 \
  --horizons 30 \
  --feature-windows 5,15,30 \
  --epochs 3 \
  --samples-per-epoch 200000 \
  --batch-size 1024 \
  --hidden-dim 64 \
  --validation-samples 100000 \
  --fixed-fee-bps 0 \
  --fixed-slippage-bps 0 \
  --rebuild-cache
```

输出目录结构：

```text
outputs/<run_name>/
├── REPORT.md
├── metrics.json
├── run_config.json
├── ohlcv_tcn.pt
├── ohlcv_tcn_history.json
├── ohlcv_tcn_test_predictions.npz
├── full_feature_tcn.pt
├── full_feature_tcn_history.json
└── full_feature_tcn_test_predictions.npz
```

本次产物大小：

| path | size |
| --- | ---: |
| `data/okx_spot_1m_top20_365d` | 234M |
| `data/cache/okx_spot_1m_top20_365d_l256_h5_15` | 1.7G |
| `data/cache/okx_spot_1m_top20_365d_l256_h30` | 1.7G |
| `outputs/superpnl_top20_365d_l256_h5_15_hd64_e3` | 62M |
| `outputs/superpnl_top20_365d_l256_h30_hd64_e3` | 32M |

## 6. 结果复现

5m/15m 主结果：

| model | horizon | total_return | sharpe | max_drawdown | turnover | trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_trade | - | 0.0000 | 0.000 | 0.0000 | 0.0000 | 0 |
| buy_and_hold_equal_weight | - | 0.0657 | 0.894 | -0.1734 | 0.0000 | 20 |
| naive_momentum | 5m/15m | -0.2661 | -6.653 | -0.2824 | 0.0851 | 134121 |
| ohlcv_tcn | 5m | -0.1216 | -2.233 | -0.1904 | 0.2370 | 373578 |
| ohlcv_tcn | 15m | -0.0457 | -0.883 | -0.1430 | 0.2973 | 468544 |
| full_feature_tcn | 5m | -0.1449 | -3.249 | -0.1933 | 0.3167 | 499082 |
| full_feature_tcn | 15m | 0.6246 | 9.099 | -0.0579 | 0.2472 | 389591 |

30m 补充结果：

| model | horizon | total_return | sharpe | max_drawdown | turnover | trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| buy_and_hold_equal_weight | - | 0.0653 | 0.889 | -0.1734 | 0.0000 | 20 |
| naive_momentum | 30m | -0.2656 | -6.638 | -0.2824 | 0.0851 | 134130 |
| ohlcv_tcn | 30m | -0.0451 | -1.169 | -0.1218 | 0.2547 | 401456 |
| full_feature_tcn | 30m | -0.0580 | -1.532 | -0.0795 | 0.1594 | 251256 |

结论：

- 当前唯一正向模型结果是 `full_feature_tcn_15m`。
- `5m` 不可用，`30m` 有一点预测指标优势但没有转成正 PnL。
- 当前回测是零成本；加入 maker/taker 成本后，15m 也会因为高换手被成本打穿。
- 下游不应直接使用 `pred_ret > 0` 的逐分钟翻仓规则，必须先加入换手约束、持仓时间约束、阈值选择或成本感知训练。

## 7. 成本敏感性复现

5m/15m 成本敏感性文件：

```text
outputs/superpnl_top20_365d_l256_h5_15_hd64_e3/cost_threshold_sensitivity.json
```

30m 成本敏感性文件：

```text
outputs/superpnl_top20_365d_l256_h30_hd64_e3/cost_threshold_sensitivity.json
```

复现脚本可以直接调用 `superpnl.training.backtest_scores`，对同一份 predictions 改 `threshold_bps`、`fixed_fee_bps` 和 `fixed_slippage_bps`。

核心结论：

- `full_feature_tcn_15m` 零成本下从 `0bps` 到 `20bps` 阈值仍为正。
- 加入 `maker_8bps` 或 `taker_12bps` 后，所有阈值基本不可用。
- `full_feature_tcn_30m` 零成本也为负，只有极高阈值下接近不交易。

## 8. Top20 币种简介

下面的类别说明只用于理解币池，不作为模型输入特征。新增这类人工基本面标签进入模型前，必须单独评估数据来源时间戳，否则可能产生未来信息泄漏或幸存者偏差。

| symbol | 类别 | 简介 | 主要风险 |
| --- | --- | --- | --- |
| BTC-USDT | PoW / 数字黄金 | Bitcoin 是最早的加密资产，以固定供给、PoW 安全性和高流动性为核心叙事。 | 宏观流动性、监管、矿工经济、链上活动周期。 |
| ETH-USDT | L1 / 智能合约 | Ethereum 是最大智能合约生态之一，承载 DeFi、NFT、L2 结算和大量链上应用。 | L2 分流、手续费收入波动、监管和生态竞争。 |
| DOGE-USDT | PoW / Meme / 支付 | Dogecoin 是老牌 meme 币，社区和社交传播强，交易属性强于基本面现金流。 | 情绪驱动强、波动高、缺少明确收入模型。 |
| SOL-USDT | 高性能 L1 | Solana 主打高吞吐、低费用和消费级链上应用，生态包括 DeFi、meme、NFT 和支付。 | 网络稳定性、生态拥挤、L1 竞争和高 beta。 |
| XRP-USDT | 支付 / 结算网络 | XRP 关联 XRP Ledger 和跨境支付叙事，历史流动性深。 | 监管事件、机构采用不确定性、供给释放。 |
| PEPE-USDT | Meme | PEPE 是以互联网 meme 文化为核心的高 beta 代币，价格主要受情绪、流动性和社群交易驱动。 | 极高波动、叙事退潮、流动性踩踏。 |
| TRX-USDT | L1 / 稳定币转账生态 | TRON 是高频稳定币转账网络之一，USDT 转账和低费用是核心使用场景。 | 生态集中度、监管、链上活跃结构变化。 |
| XAUT-USDT | RWA / 代币化黄金 | Tether Gold 代表链上黄金敞口，每个 XAUt 通常对应实物黄金权益。 | 托管和发行方风险、赎回机制、金价和流动性。 |
| BIO-USDT | DeSci / BioDAO | BIO Protocol 面向去中心化科学和生物技术融资，代币用于协议治理和生态激励。 | 新叙事资产、流动性薄、项目落地周期长。 |
| PENGU-USDT | NFT IP / Meme / 社群 | PENGU 是 Pudgy Penguins 生态相关代币，核心是 NFT IP、消费品牌和社区传播。 | 品牌热度、NFT 周期、meme 情绪和代币效用不确定。 |
| PI-USDT | 移动端社交网络 / Web3 App | Pi Network 强调移动端参与、身份验证社交网络和 Web3 应用生态。 | 真实使用闭环、流通供给、交易所定价和社区预期。 |
| ZKJ-USDT | ZK / 互操作基础设施 | ZKJ 关联 Polyhedra Network，叙事集中在零知识证明、跨链互操作和基础设施。 | 技术落地、代币释放、低流动性和高波动。 |
| TRUMP-USDT | 政治 Meme | TRUMP 是政治人物 IP 相关 meme 资产，交易高度依赖新闻、社交媒体和事件驱动。 | 事件跳空、监管/声誉风险、筹码集中和流动性冲击。 |
| SUI-USDT | Move L1 | Sui 是基于 Move 的高性能 L1，强调对象模型、并行执行和面向大众应用。 | L1 竞争、生态留存、解锁和高 beta。 |
| FIL-USDT | 去中心化存储 | Filecoin 提供去中心化存储网络，叙事包括存储、数据可用性和去中心化基础设施。 | 存储需求增长、矿工经济、代币释放和竞品。 |
| ADA-USDT | PoS L1 | Cardano 是 PoS 智能合约公链，强调研究驱动、形式化方法和长期路线图。 | 生态增长速度、开发者活跃、L1 竞争。 |
| APE-USDT | NFT / DAO / IP | ApeCoin 与 BAYC / Yuga Labs 生态相关，偏 NFT IP、社区治理和游戏/元宇宙叙事。 | NFT 周期、生态落地、治理价值捕获不确定。 |
| CHZ-USDT | 体育粉丝代币基础设施 | Chiliz 面向体育和娱乐粉丝代币生态，与俱乐部粉丝互动、权益和交易相关。 | 体育 IP 合作持续性、粉丝代币需求、监管。 |
| LINK-USDT | Oracle / 数据基础设施 | Chainlink 是主流预言机和链下数据基础设施，服务 DeFi、RWA 和跨链数据。 | 协议收入捕获、竞争、代币价值累积路径。 |
| LTC-USDT | PoW / 支付 | Litecoin 是老牌 PoW 支付型资产，常被视作 Bitcoin 的轻量化支付版本。 | 叙事老化、生态增长有限、PoW 资产整体周期。 |

## 9. Top20 样本统计

以下统计直接来自 `data/okx_spot_1m_top20_365d/csv/*.csv.gz`，不是外部行情网站数据。

| symbol | rows | raw_12m_return | ann_vol_1m | avg_daily_usdt_amount |
| --- | ---: | ---: | ---: | ---: |
| BTC-USDT | 525601 | -18.61% | 43.89% | 633,462,355 |
| ETH-USDT | 525601 | 28.30% | 68.98% | 594,018,468 |
| DOGE-USDT | 525601 | -37.46% | 97.70% | 122,311,431 |
| SOL-USDT | 525601 | -41.48% | 77.12% | 160,610,015 |
| XRP-USDT | 525601 | -36.75% | 79.13% | 74,192,255 |
| PEPE-USDT | 525601 | -55.25% | 117.69% | 31,964,714 |
| TRX-USDT | 525601 | 33.32% | 29.73% | 11,994,302 |
| XAUT-USDT | 525601 | 39.03% | 25.93% | 14,298,079 |
| BIO-USDT | 525601 | -42.92% | 172.55% | 3,557,433 |
| PENGU-USDT | 525601 | -7.44% | 156.01% | 12,734,478 |
| PI-USDT | 525601 | -69.65% | 125.66% | 22,182,827 |
| ZKJ-USDT | 525601 | -99.29% | 212.31% | 1,361,938 |
| TRUMP-USDT | 525601 | -81.12% | 140.33% | 23,189,463 |
| SUI-USDT | 525601 | -73.44% | 123.62% | 37,916,679 |
| FIL-USDT | 525601 | -66.50% | 122.85% | 12,460,228 |
| ADA-USDT | 525601 | -63.67% | 97.06% | 19,404,847 |
| APE-USDT | 525601 | -70.90% | 120.51% | 1,050,945 |
| CHZ-USDT | 525601 | 1.01% | 98.36% | 1,355,496 |
| LINK-USDT | 525601 | -35.27% | 93.09% | 12,520,266 |
| LTC-USDT | 525601 | -32.96% | 86.81% | 19,015,173 |

读取方式：

```bash
python3 - <<'PY'
import json, math
from pathlib import Path
import pandas as pd

root = Path("data/okx_spot_1m_top20_365d")
meta = json.load(open(root / "metadata.json"))
for sym in meta["symbols"]:
    df = pd.read_csv(root / "csv" / f"{sym}.csv.gz", usecols=["timestamp", "close", "amount"])
    close = df["close"].astype(float)
    raw_return = close.iloc[-1] / close.iloc[0] - 1
    logret = close.apply(math.log).diff().dropna()
    ann_vol = logret.std() * math.sqrt(365 * 24 * 60)
    avg_daily_amount = df["amount"].astype(float).mean() * 1440
    print(sym, len(df), raw_return, ann_vol, avg_daily_amount)
PY
```

## 10. 按 symbol 的模型表现

`full_feature_tcn_15m` test split：

| symbol | total_return | avg_position |
| --- | ---: | ---: |
| ZKJ-USDT | 6.8019 | 0.5914 |
| BIO-USDT | 3.6249 | 0.5672 |
| APE-USDT | 1.3716 | 0.5808 |
| PI-USDT | 1.3389 | 0.6106 |
| PEPE-USDT | 1.0835 | 0.6146 |
| PENGU-USDT | 0.6493 | 0.5682 |
| TRUMP-USDT | 0.5221 | 0.6683 |
| CHZ-USDT | 0.4987 | 0.6037 |
| ADA-USDT | 0.4710 | 0.6873 |
| DOGE-USDT | 0.4196 | 0.7537 |
| LTC-USDT | 0.4021 | 0.8091 |
| ETH-USDT | 0.3264 | 0.7914 |
| FIL-USDT | 0.3151 | 0.6613 |
| LINK-USDT | 0.2099 | 0.7769 |
| SOL-USDT | 0.1743 | 0.7623 |
| BTC-USDT | 0.1697 | 0.8497 |
| TRX-USDT | 0.1392 | 0.8581 |
| SUI-USDT | 0.1285 | 0.7019 |
| XRP-USDT | 0.0464 | 0.8084 |
| XAUT-USDT | -0.0849 | 0.8166 |

`full_feature_tcn_30m` test split：

| symbol | total_return | avg_position |
| --- | ---: | ---: |
| BTC-USDT | 0.2328 | 0.5136 |
| DOGE-USDT | 0.1106 | 0.4497 |
| TRX-USDT | 0.0967 | 0.5307 |
| SUI-USDT | 0.0901 | 0.3991 |
| PEPE-USDT | 0.0819 | 0.3442 |
| LTC-USDT | 0.0694 | 0.5142 |
| ETH-USDT | 0.0523 | 0.4513 |
| XRP-USDT | 0.0149 | 0.5000 |
| PI-USDT | -0.0087 | 0.3648 |
| PENGU-USDT | -0.0098 | 0.3101 |
| LINK-USDT | -0.0395 | 0.4548 |
| APE-USDT | -0.0637 | 0.3306 |
| SOL-USDT | -0.0658 | 0.4355 |
| CHZ-USDT | -0.0835 | 0.3627 |
| ADA-USDT | -0.1419 | 0.3934 |
| FIL-USDT | -0.1453 | 0.3818 |
| BIO-USDT | -0.2038 | 0.2990 |
| XAUT-USDT | -0.2281 | 0.5253 |
| TRUMP-USDT | -0.2874 | 0.3951 |
| ZKJ-USDT | -0.3832 | 0.3192 |

观察：

- 15m 的收益集中在高波动小币，尤其 ZKJ、BIO、APE、PI、PEPE。
- 30m 对 BTC、DOGE、TRX 等大流动性标的更友好，但组合整体仍为负。
- XAUT 在 15m 和 30m 下都不是主要收益来源，说明黄金类 RWA 的短线模式可能和高 beta 加密资产不同。
- 15m 模型对高波动下跌资产的 timing 有明显贡献，但这也意味着样本外风险更高。

## 11. 复现失败排查

常见问题：

| 问题 | 检查点 |
| --- | --- |
| Top20 不一致 | 不要重新按当前 24h 成交额选币，使用 `metadata.json` 固化的 symbols。 |
| 行数不一致 | 检查 OKX 是否返回未确认 K 线；脚本只保留 `confirm=1`。 |
| 训练结果不完全一致 | 检查 `seed=17`、MPS/CUDA/CPU 差异、PyTorch 版本差异。 |
| test 时间段不同 | 检查 `lookback`、`horizons`、`feature_windows` 是否一致。 |
| PnL 被成本打穿 | 当前换手过高，属于已知问题，不是复现错误。 |
| 数据文件被误提交 | 检查 `.gitignore` 是否包含 `data/`, `outputs/`, `*.pt`, `*.npz`。 |

## 12. 下一步建议

下一轮要把零成本 PnL 转成可交易 PnL，优先做：

1. 在 val split 上搜索 `threshold_bps`、最短持仓时间和冷却时间。
2. 在 loss 或 label 中加入成本和换手惩罚。
3. 把 `target_pos` 从二值仓位改成平滑仓位。
4. 加入 symbol-level cap，降低对 ZKJ/BIO 等小币的集中依赖。
5. 在完全不同时间段重新下载一段 out-of-sample 数据，验证 15m 是否仍有效。

## 13. 参考链接

这些链接用于理解币种背景，不参与模型训练：

- Bitcoin: https://bitcoin.org/
- Ethereum: https://ethereum.org/
- Dogecoin: https://dogecoin.com/
- Solana: https://solana.com/
- XRP Ledger: https://xrpl.org/
- TRON: https://tron.network/
- Tether Gold: https://gold.tether.to/
- BIO Protocol: https://docs.bio.xyz/bio/introduction/bio-token
- Pudgy Penguins PENGU: https://www.pudgypenguins.com/pengu
- Pi Network: https://minepi.com/about/
- Polyhedra Network: https://polyhedra.network/
- Sui: https://sui.io/
- Filecoin: https://filecoin.io/
- Cardano: https://cardano.org/
- ApeCoin: https://apecoin.com/
- Chiliz: https://www.chiliz.com/
- Chainlink: https://chain.link/
- Litecoin: https://litecoin.org/
