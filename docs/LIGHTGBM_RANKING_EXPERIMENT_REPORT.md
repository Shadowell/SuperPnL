# LightGBM Ranking Experiment Report

本文档记录 LightGBM 用于 `4h / 1d` 截面排序的第一轮验证。目标不是预测收益绝对值，而是在每个 timestamp 的 20 个币之间学习相对强弱排序，再用低换手 Top-K 策略验证是否能转成含成本 PnL。

## 实现方式

新增 LightGBM tabular 路线：

```text
feature_inputs[t, symbol] -> LightGBM score -> cross-sectional ranking -> low-turnover Top-K
```

本轮使用 `factor_set=base`，输入为历史 OHLCV 派生因子和同一时刻截面因子，不使用盘口、成本、流动性作为训练特征。

已验证两个目标：

| objective | 说明 |
| --- | --- |
| `lambdarank` | 将同一 timestamp 内未来收益分桶成 relevance，用 LightGBM ranking objective 训练 |
| `rank_regression` | 将同一 timestamp 内未来收益转成连续排名分位 `[-0.5, 0.5]`，用 LightGBM 回归学习排序分数 |

实践结果是 `rank_regression` 明显优于 `lambdarank`。后续默认优先使用 `rank_regression`。

注意：LightGBM 输出是排序分数，不是收益 bps。低换手选参脚本里的 `threshold_bps` 只是复用已有 backtester 字段；例如 `-6000` 表示阈值 `-0.6`，基本等价于每次再平衡都选 Top-K。

## 训练设置

```text
dataset = Top20 / 1min / 365d
cache = data/cache/okx_spot_1m_top20_365d_l256_h60_240_1440_base
horizons = 60,240,1440
focus = 240m,1440m
train samples = 2,000,000 rows = 100,000 timestamp groups
validation samples = 400,000 rows = 20,000 timestamp groups
fee = 8bps
slippage = 0bps
```

新增特征泄漏判断：本轮没有新增数据特征。LightGBM 标签使用未来收益或未来收益的截面排名，只作为监督目标，不进入输入特征；不会产生新的未来信息泄漏。主要风险仍是用 test split 调参，因此下方策略参数只在 validation split 上选择。

## 排序能力

测试集 rank IC 对比：

| model | 4h rank_ic | 1d rank_ic |
| --- | ---: | ---: |
| TCN OHLCV-only | 0.0192 | 0.0425 |
| TCN base factor | 0.0512 | 0.0479 |
| LightGBM `lambdarank` | -0.0134 | 0.0159 |
| LightGBM `rank_regression` | 0.0625 | 0.0673 |

结论：LightGBM `rank_regression` 在 `4h / 1d` 的截面排序能力已经超过 TCN，是目前最强的 factor-only 排序 baseline。

## Validation-Only 低换手结果

参数只在 validation split 上搜索，然后应用到 test split 一次。

| horizon | model | val_net | test_net | test_sharpe | test_max_dd | test_trades |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4h | LightGBM rank_regression | -15.80% | -8.27% | -2.38 | -10.37% | 521 |
| 1d | LightGBM rank_regression | -4.30% | -3.52% | -2.10 | -4.02% | 47 |

同期基准：

| split | baseline | net | sharpe | max_dd |
| --- | --- | ---: | ---: | ---: |
| validation | no_trade | 0.00% | 0.00 | 0.00% |
| validation | buy_and_hold | -32.54% | -3.65 | -46.11% |
| test | no_trade | 0.00% | 0.00 | 0.00% |
| test | buy_and_hold | 3.24% | 0.45 | -17.34% |

判断：

- LightGBM 的排序信号是真实存在的，尤其 `4h / 1d` rank IC 明显增强。
- 但当前 long-only Top-K 策略仍没有稳定转成正的绝对 PnL。
- validation 期间市场下跌，1d 策略显著少亏于 buy-and-hold，但仍输给 no-trade。
- test 期间 buy-and-hold 转正，LightGBM long-only 选币没有跟上，说明模型更像相对强弱排序器，不是独立做多信号。

## 下一步

继续 LightGBM 路线，但不要只做 long-only Top-K：

```text
1. 增加 market regime gate：只在市场趋势/广度允许时开 long-only 仓位。
2. 对 LightGBM 分数做 validation calibration：按分位而不是固定阈值交易。
3. 评估 market-neutral top-minus-bottom 研究指标，但不作为当前现货策略上线目标。
4. 解决 expanded 长周期 cache 的低内存构建，再验证 expanded factors 对 LightGBM 排序是否继续增益。
```

当前不建议扩 Top100。先把 `4h / 1d + rank_regression + regime gate` 做清楚。
