# Long Horizon Experiment Report

> **2026-10-05 评测修复后：** 本文数字是修复前历史记录。成交时点、组合记账、标签隔离和因子数值口径已修正，以下指标需重新训练和评测；本轮未重跑真实历史实验。

本文档记录 `1h / 4h / 1d` horizon 的第一轮收益验证。结论先行：长周期的 rank IC 比 15m 更好，base factor model 明显强于 OHLCV-only；但经过 validation-only 低换手选参后，测试集 PnL 仍未跑赢 no-trade 或 buy-and-hold，暂不能视为可交易策略。

## 实验设置

数据仍使用 OKX 现货 Top20 / 1min K 线：

```text
train: 2025-05-01 15:05 UTC -> 2026-01-10 17:20 UTC
val:   2026-01-10 17:21 UTC -> 2026-03-06 04:06 UTC
test:  2026-03-06 04:07 UTC -> 2026-04-29 14:53 UTC
```

配置：

```text
lookback = 256
horizons = 60,240,1440
feature_windows = 5,15,30,60,240,1440
factor_set = base
position_loss_weight = 0
model_selection_metric = val_rank_ic_mean
fixed_fee_bps = 8
fixed_slippage_bps = 0
```

本轮没有新增特征，不引入新的未来信息泄漏风险。标签仍是 `t` 时刻决策、下一根 open 入场、`horizon` 后 open 出场的未来收益，只作为监督目标使用。

## 训练结果

| model | best_epoch | best_val_rank_ic_mean |
| --- | ---: | ---: |
| ohlcv_tcn | 5 | 0.0407 |
| full_feature_tcn | 3 | 0.0606 |

长周期上，base factor model 的验证 rank IC 明显高于 OHLCV-only。

测试集 rank IC：

| model | 1h rank_ic | 4h rank_ic | 1d rank_ic |
| --- | ---: | ---: | ---: |
| ohlcv_tcn | 0.0233 | 0.0192 | 0.0425 |
| full_feature_tcn | 0.0324 | 0.0512 | 0.0479 |

## Validation-Only 低换手选参

下面参数均只在 validation split 上选择，然后应用到 test split 一次。

| horizon | model | val_net | test_net | test_sharpe | test_max_dd | test_trades | selected_config |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1h | ohlcv_tcn | 5.69% | -24.27% | -5.20 | -25.29% | 72 | `top_k=3, rebalance=60m, holding=120m, threshold=0bps` |
| 1h | full_feature_tcn | 0.24% | -9.52% | -2.28 | -17.39% | 315 | `top_k=3, rebalance=60m, holding=60m, threshold=20bps` |
| 4h | ohlcv_tcn | -14.45% | 1.49% | 0.38 | -9.06% | 595 | `top_k=1, rebalance=240m, holding=240m, threshold=-20bps` |
| 4h | full_feature_tcn | 3.68% | -2.90% | -0.83 | -11.49% | 123 | `top_k=1, rebalance=60m, holding=240m, threshold=40bps` |
| 1d | ohlcv_tcn | 2.31% | -5.12% | -2.47 | -7.47% | 49 | `top_k=1, rebalance=1440m, holding=4320m, threshold=-50bps` |
| 1d | full_feature_tcn | 3.15% | -3.63% | -0.85 | -8.98% | 36 | `top_k=1, rebalance=720m, holding=2880m, threshold=25bps` |

同期测试集基准：

| baseline | test_net | test_sharpe | test_max_dd |
| --- | ---: | ---: | ---: |
| no_trade | 0.00% | 0.00 | 0.00% |
| buy_and_hold_equal_weight | 3.24% | 0.45 | -17.34% |

## 判断

- `1h` 不值得继续作为当前主周期：验证选出来的 OHLCV 参数测试大幅转负，factor model 测试也为负。
- `4h` 的 factor rank IC 最强，但 PnL 转化仍失败；OHLCV 测试小正但 validation 为负，不能作为有效选参结果。
- `1d` 的换手和成本压力最低，但测试仍为负；当前策略层没有把 1d 排序信号稳定转成收益。
- 当前最有研究价值的是 `4h / 1d` 的排序信号，而不是直接上 TCN 仓位规则。

## Expanded 因子状态

尝试直接构建 `factor_set=expanded` 的 `60/240/1440` 缓存时，当前机器在前处理阶段被系统终止，未产出完整 cache 和报告。原因是 expanded 数据准备会在内存中拼接完整特征矩阵，峰值过高。

后续如果继续 expanded 长周期实验，应先改数据准备流程：

```text
按 symbol / block 分块写入 memmap
避免一次性 np.stack 全部 feature_inputs
复用已计算的 horizon-independent features
只对 label / split 重新生成 horizon-dependent cache
```

在完成低内存数据构建前，不建议把 Top100 和 expanded 长周期训练同时推进。
