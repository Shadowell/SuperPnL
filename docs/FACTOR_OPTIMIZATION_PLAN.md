# Factor Optimization Plan

本文档记录 SuperPnL 下一轮因子优化方向。目标不是追求零成本回测更漂亮，而是验证多周期趋势、流动性、波动状态和残差强弱能否在低换手、含成本约束下改善 PnL。

## 1. 当前问题

当前 `full_feature_tcn_15m` 只证明了零成本条件下有毛收益。加入 maker/taker 成本后，逐分钟 `pred_ret > threshold` 翻仓会被换手打穿。

因此下一轮优化必须同时解决：

```text
信号质量
交易频率
流动性过滤
市场状态识别
样本外稳定性
```

## 2. 参考方向

公开研究中相对稳定的 crypto 因子方向包括：

- 多周期趋势：CTREND 这类趋势因子聚合不同 horizon 的价格和成交量技术信号，并强调大流动性币和交易成本后的鲁棒性。
- 动量 / 反转切换：crypto 在不同周期和不同流动性分组中可能从 momentum 切到 reversal。
- 流动性条件：大流动性币更偏 momentum，长尾低流动性币更容易表现出 reversal 或不可交易噪声。
- 波动率与流动性周期：crypto 的波动和成交量存在小时、星期和交易机制相关的周期性。

参考链接：

- `A Trend Factor for the Cross-Section of Cryptocurrency Returns`: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4601972
- `Momentum and liquidity in cryptocurrencies`: https://arxiv.org/abs/1904.00890
- `Up or down? Short-term reversal, momentum, and liquidity effects in cryptocurrency markets`: https://www.sciencedirect.com/science/article/pii/S1057521921002349
- `Cryptocurrency Momentum and Reversal`: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3913263
- `Periodicity in Cryptocurrency Volatility and Liquidity`: https://arxiv.org/abs/2109.12142

## 3. 已实现实验开关

新增数据配置：

```text
factor_set = base | expanded
```

`base` 保持旧实验复现。`expanded` 增加：

```text
trend_score / ema_slope / breakout_pos / volume_confirmed_ret
log_amount_mean / amount_z / amihud
downside_vol / range_mean / jump_intensity
btc_beta / btc_resid_ret / eth_beta / eth_resid_ret
market_dispersion / market_breadth_pos
cross_section_amount_rank / cross_section_amihud_rank / cross_section_btc_resid_ret_rank
```

新增低换手回测：

```text
--low-turnover-backtest
--low-turnover-top-k
--rebalance-interval-bars
--min-holding-bars
--cooldown-bars
--max-position-per-symbol
--max-total-position
--min-liquidity-rank
--max-illiquidity-rank
```

## 4. 推荐第一轮实验

```bash
PYTHONPATH=src python3 scripts/run_superpnl_experiment.py \
  --raw-dir data/okx_spot_1m_top20_365d \
  --cache-dir data/cache/okx_spot_1m_top20_365d_l256_h15_expanded \
  --out-dir outputs/superpnl_top20_365d_l256_h15_expanded_lowturnover \
  --lookback 256 \
  --horizons 15 \
  --feature-windows 5,15,30,60,240,1440 \
  --factor-set expanded \
  --epochs 3 \
  --samples-per-epoch 200000 \
  --batch-size 512 \
  --hidden-dim 96 \
  --validation-samples 100000 \
  --threshold-bps 10 \
  --fixed-fee-bps 8 \
  --fixed-slippage-bps 0 \
  --low-turnover-backtest \
  --low-turnover-top-k 3 \
  --rebalance-interval-bars 15 \
  --min-holding-bars 30 \
  --cooldown-bars 30 \
  --max-position-per-symbol 0.2 \
  --max-total-position 0.6 \
  --min-liquidity-rank -0.2 \
  --max-illiquidity-rank 0.2 \
  --rebuild-cache
```

这条命令不是最终生产参数，只是第一轮研究起点。正式报告必须在 validation split 上选参数，并对 test split 只报告一次。

正式选参使用：

```bash
PYTHONPATH=src python3 scripts/select_low_turnover_params.py \
  --cache-dir data/cache/okx_spot_1m_top20_365d_l256_h15_expanded \
  --out-dir outputs/superpnl_top20_365d_l256_h15_expanded_alphaonly_lowturnover \
  --fixed-fee-bps 8 \
  --fixed-slippage-bps 0
```

该脚本只用 validation split 搜索 `threshold/top_k/rebalance/holding/filter`，然后把选出的参数应用到 test split 一次。

## 5. 泄漏判断

本轮新增特征不使用盘口、成本、未来成交量或未来滑点。所有 rolling、EMA、beta、rank 只使用 `<= t` 的历史或同一时刻截面数据。

主要风险：

- 用未来 24h 成交额重新选择历史 universe。
- 用未来上市状态筛掉历史上不可交易的标的。
- 用 test split 反复搜索 `threshold/top_k/holding/cooldown`。
- 用 val/test/live 数据重新拟合标准化参数。

这些风险必须通过固定 `metadata.json`、train-only 标准化和 validation-only 参数选择规避。

## 6. 成功标准

下一轮认为值得继续研究，至少要满足：

```text
含 maker_8bps 的 net_total_return > 0
Sharpe 高于 buy-and-hold
max_drawdown 不显著劣于 buy-and-hold
turnover 明显低于逐分钟二值翻仓
收益不集中在单个 symbol 或单个月份
full_feature_tcn 显著优于 ohlcv_tcn
```

如果 expanded 因子只提高零成本收益，但含成本仍失败，则说明方向仍停留在毛 edge，不应下游上线。

## 7. 2026-05-03 阶段结果

已完成两轮 Top20 / 15m / expanded 因子实验。

第一轮保留 `position_loss_weight=0.15`，full-feature 模型预测明显正偏：

```text
pred_mean ~= +17.21bps
pred > 10bps ratio ~= 84.9%
```

含 `8bps` 固定手续费的低换手测试结果仍为负，说明 position head 当前会把模型推向过度乐观和高暴露，暂时不适合作为主训练目标。

第二轮改为 alpha-only：

```text
position_loss_weight = 0
model_selection_metric = val_rank_ic_mean
```

验证集 best epoch：

| model | best_epoch | val_rank_ic_mean |
| --- | ---: | ---: |
| ohlcv_tcn | 5 | 0.0259 |
| full_feature_tcn | 4 | 0.0495 |

full-feature 的 rank IC 明显高于 OHLCV-only，说明 expanded 因子确实提高了截面排序信号。但验证集选参后应用到测试集，PnL 转化并不稳定：

| model | selected on validation | val_net | test_net | test_sharpe | test_max_dd | test_trades |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ohlcv_tcn | threshold=15bps, top_k=2, rebalance=30m, holding=30m, no filter | 4.17% | 3.36% | 1.03 | -8.53% | 150 |
| full_feature_tcn | threshold=2bps, top_k=3, rebalance=30m, holding=120m, liquid_mid filter | 2.34% | -6.02% | -1.42 | -14.76% | 66 |

同期测试集基准：

| baseline | test_net | test_sharpe | test_max_dd |
| --- | ---: | ---: | ---: |
| no_trade | 0.00% | 0.00 | 0.00% |
| buy_and_hold_equal_weight | 5.92% | 0.81 | -17.34% |

当前判断：

- `alpha-only + best epoch` 是正确方向，已修复第一轮的预测正偏。
- expanded 因子提高 rank IC，但还没有稳定转化为含成本 PnL。
- OHLCV-only 低换手在这次验证选择下测试为正，但净收益仍低于 buy-and-hold。
- OHLCV-only 的测试收益主要来自 `BIO-USDT`、`APE-USDT`、`PI-USDT`，同时被 `ZKJ-USDT` 明显拖累；收益不够分散。
- full-feature 的测试亏损集中在 `ZKJ-USDT`、`APE-USDT`、`TRUMP-USDT`，且 `2026-03` 和 `2026-04` 两个月都为负，不能解释为单月异常。
- 暂时不应扩到 Top100 训练；先解决分数校准、持仓规则和收益归因。
- 下一步重点不是再堆因子，而是把 `pred_ret` 转成仓位的策略层从固定阈值改成 validation-calibrated rank/quantile policy，并加入 symbol-level 风险约束。
