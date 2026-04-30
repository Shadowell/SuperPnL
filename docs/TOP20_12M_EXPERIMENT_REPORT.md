# Top20 12M Experiment Report

本报告记录 SuperPnL 第一轮 OKX 现货 Top20 / 12 个月实验结果。完整产物在：

```text
/Users/jie.feng/wlb/SuperPnL/outputs/superpnl_top20_365d_l256_h5_15_hd64_e3/
```

## 1. 数据

| 项 | 值 |
| --- | --- |
| source | OKX public API |
| market | spot `*-USDT` |
| universe | 非稳定币成交额 Top20 |
| bar_size | `1m` |
| raw window | `2025-04-30 15:00:00 UTC` -> `2026-04-30 15:00:00 UTC` |
| rows per symbol | `525,601` |
| common timestamps after feature/label filtering | `525,596` |
| features | `33` |
| lookback | `256` |
| horizons | `5m, 15m` |

Universe:

```text
BTC-USDT, ETH-USDT, DOGE-USDT, SOL-USDT, XRP-USDT,
PEPE-USDT, TRX-USDT, XAUT-USDT, BIO-USDT, PENGU-USDT,
PI-USDT, ZKJ-USDT, TRUMP-USDT, SUI-USDT, FIL-USDT,
ADA-USDT, APE-USDT, CHZ-USDT, LINK-USDT, LTC-USDT
```

时间切分：

| split | range |
| --- | --- |
| train | `2025-04-30 19:16 UTC` -> `2026-01-11 04:01 UTC` |
| val | `2026-01-11 04:02 UTC` -> `2026-03-06 21:19 UTC` |
| test | `2026-03-06 21:20 UTC` -> `2026-04-30 14:38 UTC` |

## 2. 训练配置

```text
lookback=256
horizons=5,15
feature_windows=5,15,30
hidden_dim=64
epochs=3
samples_per_epoch=200000
batch_size=1024
validation_samples=100000
fixed_fee_bps=0
fixed_slippage_bps=0
threshold_bps=0
```

训练过程：

| model | epoch 1 val_mae | epoch 2 val_mae | epoch 3 val_mae | epoch 3 val_rank_ic |
| --- | ---: | ---: | ---: | ---: |
| ohlcv_tcn | 0.0105 | 0.0047 | 0.0030 | -0.0101 |
| full_feature_tcn | 0.0127 | 0.0052 | 0.0030 | 0.0048 |

过程结论：两个模型的验证 MAE 都稳定下降，但 rank IC 较弱，说明模型确实学到了收益尺度，但截面排序能力还不强。

## 3. Test 回测对比

本表是零成本回测，即 `fee=0bps`、`slippage=0bps`。当前阶段暂不使用盘口、成本、流动性特征。

| model | horizon | total_return | sharpe | max_drawdown | turnover | trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_trade | - | 0.0000 | 0.000 | 0.0000 | 0.0000 | 0 |
| buy_and_hold_equal_weight | - | 0.0657 | 0.894 | -0.1734 | 0.0000 | 20 |
| naive_momentum | 5m/15m | -0.2661 | -6.653 | -0.2824 | 0.0851 | 134121 |
| ohlcv_tcn | 5m | -0.1216 | -2.233 | -0.1904 | 0.2370 | 373578 |
| ohlcv_tcn | 15m | -0.0457 | -0.883 | -0.1430 | 0.2973 | 468544 |
| full_feature_tcn | 5m | -0.1449 | -3.249 | -0.1933 | 0.3167 | 499082 |
| full_feature_tcn | 15m | 0.6246 | 9.099 | -0.0579 | 0.2472 | 389591 |

主要结论：

- 当前唯一有效结果是 `full_feature_tcn_15m`，零成本 test return 为 `+62.46%`，显著高于 buy-and-hold 的 `+6.57%`。
- `5m` horizon 不可用；OHLCV-only 和 full-feature 的 5m 都亏损。
- 有因子模型在 15m 上明显优于 OHLCV-only，说明当前因子集合对 15m PnL 有帮助。
- 换手极高，当前输出不能直接视为可实盘收益。

## 4. 预测指标

| model | horizon | mae | hit_rate | rank_ic |
| --- | ---: | ---: | ---: | ---: |
| ohlcv_tcn | 5m | 0.00246 | 0.4630 | -0.0003 |
| ohlcv_tcn | 15m | 0.00277 | 0.4770 | -0.0052 |
| full_feature_tcn | 5m | 0.00230 | 0.4610 | -0.0047 |
| full_feature_tcn | 15m | 0.00301 | 0.4899 | 0.0200 |

`full_feature_tcn_15m` 的 rank IC 为 `0.0200`，方向命中率仍低于 50%。当前收益更像是仓位筛选/风险暴露结构带来的 PnL，而不是一个强方向分类器。

## 5. 稳定性

`full_feature_tcn_15m` 月度收益：

| month | total_return |
| --- | ---: |
| 2026-03 | 0.1933 |
| 2026-04 | 0.3614 |

`full_feature_tcn_15m` test split 表现最好的 symbol：

| symbol | total_return | avg_position |
| --- | ---: | ---: |
| ZKJ-USDT | 6.8019 | 0.5914 |
| BIO-USDT | 3.6249 | 0.5672 |
| APE-USDT | 1.3716 | 0.5808 |
| PI-USDT | 1.3389 | 0.6106 |
| PEPE-USDT | 1.0835 | 0.6146 |

稳定性判断：

- 月度上 2026-03 和 2026-04 都为正，不是单月孤立收益。
- symbol 上收益集中在高波动小币，特别是 ZKJ/BIO；这会带来样本外稳定性风险。
- `full_feature_tcn_5m` 两个月都亏损，不应作为下游策略 horizon。

## 6. 成本敏感性

额外输出文件：

```text
/Users/jie.feng/wlb/SuperPnL/outputs/superpnl_top20_365d_l256_h5_15_hd64_e3/cost_threshold_sensitivity.json
```

`full_feature_tcn_15m` 在零成本下，阈值从 `0bps` 到 `20bps` 仍为正；但只要加入固定 maker `8bps` 或 taker `10bps + 2bps slippage`，所有阈值几乎归零亏完。

原因是当前仓位规则 `pred_ret > threshold` 产生了很高换手，15m 零成本下 `turnover=0.2472`，test trades 为 `389,591`。

因此当前结论是：

```text
当前模型证明了 15m 因子信号在零成本条件下有 PnL，但还不能直接用于含真实手续费/滑点的实盘。
```

下一步要把 PnL 变成可交易收益，必须做至少一项：

- 降低换手：加入最短持仓时间、仓位平滑、冷却时间、top-k 调仓。
- 成本感知训练：label 改成扣成本后的 net edge，或者 loss 里加入 turnover penalty。
- 阈值策略搜索：在 val split 上选择 threshold / holding rule，再只在 test 上报告一次。
- 分币种约束：限制过度依赖 ZKJ/BIO 等小币的收益贡献。

## 7. 下游使用

通用下游配置见：

```text
/Users/jie.feng/wlb/SuperPnL/docs/DOWNSTREAM_USAGE.md
```

当前可用模型：

```text
/Users/jie.feng/wlb/SuperPnL/outputs/superpnl_top20_365d_l256_h5_15_hd64_e3/full_feature_tcn.pt
```

当前建议下游只使用：

```text
horizon=15m
feature_windows=5,15,30
lookback=256
```

不建议直接使用默认 `pred_ret > 0` 逐分钟翻仓规则。下游策略层必须先加换手约束或成本约束，再做新的 out-of-sample 回测。
