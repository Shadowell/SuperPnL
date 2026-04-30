# BitPro Strategy Prompt

本文档是一份可直接复制给 BitPro 项目编码代理的策略实现提示词。

目标：在 BitPro 当前 `BaseStrategy` 框架内，把 SuperPnL 的 `full_feature_tcn_15m` 预测结果落成一个现货 long-only、低换手、含成本可评估的策略层。

注意：这份提示词面向 **BitPro 项目**，不是 SuperPnL 训练仓库本身。

## 使用方式

把下面整段 prompt 粘贴到 BitPro 项目的新任务中执行。

````text
你现在在 BitPro 项目中实现一个 SuperPnL 低换手现货策略。请先阅读并遵守：

1. AGENTS.md
2. README.md
3. docs/spec.md
4. docs/progress.md
5. docs/contracts/module_map_v2.md
6. docs/strategy_development_guide.md
7. backend/app/core/execution/base_strategy.py
8. backend/app/strategies/kairos_30m_horizon_dca_strategy.py
9. backend/app/services/strategy_registry.py
10. data/seed/strategies.json

目标：
基于 SuperPnL 已训练好的 `full_feature_tcn_15m` 预测输出，在 BitPro 的 `BaseStrategy` 框架里实现一个现货 long-only 低换手策略层。不要重新训练模型。不要写旧函数式 `strategy(ctx)` / `setup(ctx)`。不要直接使用 Backtrader、CCXT、数据库裸调用或交易所 API。策略必须是一套 `BaseStrategy` 代码，能被 BitPro 回测、模拟盘、实盘同构运行。

背景：
- SuperPnL 结果显示：
  - `full_feature_tcn_15m` 零成本 test total_return = +62.46%
  - `5m` 不可用
  - `30m` 当前 PnL 为负
  - 主要问题是逐分钟 `pred_ret > 0` 翻仓导致换手极高，加入手续费后收益被成本打穿
- 当前只做 OKX spot，不做永续、不做杠杆、不做借币做空
- 目标不是追求零成本最高收益，而是降低换手，让策略在含成本下仍可能可交易

请实现一个新策略：

文件建议：
- `backend/app/services/superpnl_signal_service.py`
- `backend/app/strategies/superpnl_15m_low_turnover_strategy.py`
- 更新 `backend/app/services/strategy_registry.py`
- 更新 `data/seed/strategies.json`
- 更新 `docs/progress.md`
- 必要时更新 `docs/spec.md`

策略类要求：
- 类名：`SuperPnL15mLowTurnoverStrategy`
- 必须继承 `BaseStrategy`
- 必须实现 `async def on_init(self) -> None`
- 必须实现 `async def on_bar(self, bar: BarData) -> None`
- 可选实现 `async def on_warmup_bar(self, bar: BarData) -> None`
- 只允许一个 `BaseStrategy` 子类
- 不允许旧函数式 API
- 不允许在策略类里直接读文件、调网络、调交易所、调数据库
- 如果预测不可用，必须显式跳过交易并输出诊断日志，不能 mock、dummy、random、momentum-template 或 synthetic fallback

策略 key：

```text
superpnl_15m_low_turnover
```

注册方式：

```python
from app.strategies.superpnl_15m_low_turnover_strategy import SuperPnL15mLowTurnoverStrategy

_BASE_STRATEGY_REGISTRY["superpnl_15m_low_turnover"] = SuperPnL15mLowTurnoverStrategy
```

种子配置写入 `data/seed/strategies.json`，示例：

```json
{
  "name": "SuperPnL 15m 低换手现货策略",
  "description": "使用 SuperPnL full_feature_tcn_15m 预测收益，按阈值、Top-K、最短持仓、冷却时间和再平衡间隔降低换手。现货 long-only，不做杠杆或做空。",
  "strategy_key": "superpnl_15m_low_turnover",
  "config": {
    "strategy_key": "superpnl_15m_low_turnover",
    "timeframe": "1m",
    "horizon": "15m",
    "warmup_bars": 300,

    "threshold_bps": 10,
    "top_k": 3,
    "rebalance_interval_bars": 15,
    "min_holding_bars": 30,
    "cooldown_bars": 30,

    "max_position_per_symbol": 0.2,
    "max_total_position": 0.6,
    "allow_cash": true,

    "fee_bps": 8,
    "slippage_bps": 0,

    "strategy_diagnostic_ws": true,
    "strategy_diagnostic_every_n_bars": 1,

    "signal_provider": "superpnl",
    "signal_horizon": "15m"
  },
  "exchange": "okx",
  "symbols": [
    "BTC/USDT",
    "ETH/USDT",
    "DOGE/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "PEPE/USDT",
    "TRX/USDT",
    "XAUT/USDT",
    "BIO/USDT",
    "PENGU/USDT",
    "PI/USDT",
    "ZKJ/USDT",
    "TRUMP/USDT",
    "SUI/USDT",
    "FIL/USDT",
    "ADA/USDT",
    "APE/USDT",
    "CHZ/USDT",
    "LINK/USDT",
    "LTC/USDT"
  ],
  "script_file": null
}
```

SuperPnL 信号服务：

实现 `backend/app/services/superpnl_signal_service.py`，提供统一接口：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SuperPnLSignal:
    symbol: str
    timestamp: int
    horizon: str
    pred_ret: float
    score_bps: float
    source: str


class SuperPnLSignalService:
    async def get_signal(
        self,
        symbol: str,
        timestamp_ms: int,
        horizon: str = "15m",
    ) -> SuperPnLSignal | None:
        ...
```

要求：
- `symbol` 需要兼容 BitPro 的 `BTC/USDT` 和 SuperPnL 的 `BTC-USDT`
- `timestamp_ms` 用 `BarData.timestamp`
- 如果找不到对应信号，返回 `None`
- 不允许生成假信号
- 预测不可用时策略跳过该 bar
- 第一版可以实现本地 artifact provider，读取 SuperPnL 已导出的 `npz/npy/json`，但读取逻辑必须封装在 service 中，不要放进策略类
- 后续 live provider 可以替换为实时 SuperPnL 推理服务，策略代码不应修改

策略逻辑：

1. 只处理 `timeframe="1m"` 的已收盘 K 线。
2. 多 symbol 运行时，`on_bar` 会按 symbol 被调用。策略需要维护每个 symbol 的最新 bar、最新信号、持仓状态、持仓开始 bar、冷却结束 bar。
3. 每 `rebalance_interval_bars` 才允许做一次组合级再平衡。
4. 再平衡时：
   - 对所有已有最新信号的 symbol 排序
   - 只保留 `pred_ret > threshold_bps / 10000`
   - 选择 `pred_ret` 最高的 `top_k`
   - long-only，不做空
   - 目标仓位范围 `[0, 1]`
   - 单币目标仓位不超过 `max_position_per_symbol`
   - 组合总仓位不超过 `max_total_position`
   - 若候选少于 top_k，剩余资金保持 cash
5. 最短持仓：
   - 已持仓但未达到 `min_holding_bars` 时，不允许平仓或降低仓位
6. 冷却：
   - 某 symbol 平仓后进入 `cooldown_bars`
   - 冷却期内不能重新买入该 symbol
7. 下单：
   - 使用 `await self.buy(symbol, qty)`
   - 使用 `await self.sell(symbol, qty)` 或 `await self.close_position(symbol)`
   - 不直接调用 broker 以外接口
   - 买入数量由目标 USDT 名义 / 当前 close 计算
   - 估算账户权益时优先使用 `self.state.positions["_capital"]` + 当前持仓市值
8. 成本：
   - BitPro broker 会处理真实/模拟成交成本时，策略层不用重复扣 PnL
   - 但策略诊断日志中必须记录本次调仓估算成本：

```text
estimated_cost_bps = fee_bps + slippage_bps
estimated_turnover = abs(target_position - current_position)
```

诊断日志：

参考 `Kairos30mHorizonDcaStrategy._maybe_emit_bar_diagnostic`，通过：

```python
await self.broadcast_strategy_channel(payload)
```

输出结构化诊断。

至少包含：

```json
{
  "type": "bar_diag",
  "decision": "buy_filled / sell_filled / skip_no_signal / skip_below_threshold / skip_rebalance_interval / skip_min_holding / skip_cooldown / rebalance",
  "decision_label": "中文说明",
  "summary": "人能直接看懂的一句话",
  "symbol": "BTC/USDT",
  "bar_ts_ms": 1234567890,
  "close": 123.45,
  "pred_ret": 0.0012,
  "pred_ret_bps": 12.0,
  "threshold_bps": 10,
  "rank": 1,
  "target_position": 0.2,
  "current_position": 0.0,
  "max_total_position": 0.6,
  "top_k": 3,
  "rebalance_interval_bars": 15,
  "min_holding_bars": 30,
  "cooldown_bars": 30
}
```

必须实现的决策枚举：

```python
DECISION_LABELS = {
    "warm_up_history": "历史K线不足，继续预热",
    "skip_no_signal": "未交易：SuperPnL 信号不可用",
    "skip_below_threshold": "未交易：预测收益低于阈值",
    "skip_rebalance_interval": "未交易：未到再平衡时间",
    "skip_min_holding": "未卖出：未达到最短持仓时间",
    "skip_cooldown": "未买入：仍在冷却期",
    "skip_qty_zero": "未交易：下单数量为0",
    "rebalance": "组合再平衡",
    "buy_filled": "买入成交",
    "sell_filled": "卖出成交",
    "broker_error": "下单失败",
}
```

参数搜索：

请不要把参数搜索写进策略类。策略类只执行给定 config。

如果要做搜索，请新增独立脚本：

```text
scripts/sweep_superpnl_strategy.py
```

搜索要求：
- 只用 val split 调参
- test split 只评估一次

搜索参数：

```text
threshold_bps: [0, 2, 5, 10, 20, 30, 50]
top_k: [1, 2, 3, 5, 10]
rebalance_interval_bars: [5, 15, 30, 60]
min_holding_bars: [15, 30, 60, 120]
cooldown_bars: [0, 15, 30, 60]
max_position_per_symbol: [0.1, 0.2, 0.3, 0.5]
max_total_position: [0.3, 0.5, 0.8, 1.0]
fee_bps: [8]
slippage_bps: [0, 1, 2]
```

评估指标：

必须输出：
- total_return
- annualized_return
- sharpe
- sortino
- max_drawdown
- calmar
- turnover
- trade_count
- average_position
- average_holding_minutes
- win_rate
- profit_factor
- monthly_returns
- by_symbol_pnl
- by_symbol_trade_count
- by_symbol_avg_position

对照组：

必须同时输出：
- no_trade
- buy_and_hold_equal_weight
- 原始 naive：每分钟 `pred_ret_15m > 0` 翻仓
- 新 BitPro 策略层

成功标准：
- 在 `fee_bps=8`, `slippage_bps=0` 下 test total_return > 0
- trade_count 相比原始 naive 下降至少 80%
- max_drawdown 不高于 buy-and-hold
- by_symbol_pnl 不能只靠 1-2 个 symbol
- 参数必须来自 val split，test 只评估一次

需要改的地方：

1. 新增 `backend/app/services/superpnl_signal_service.py`
2. 新增 `backend/app/strategies/superpnl_15m_low_turnover_strategy.py`
3. 更新 `backend/app/services/strategy_registry.py`
4. 更新 `data/seed/strategies.json`
5. 更新 `docs/progress.md`
6. 如新增配置或架构说明，更新 `docs/spec.md`
7. 如该策略需要上线展示，说明需要执行远端 seed sync：

```bash
export BITPRO_SEED_SSH=user@production-host
./scripts/seed_strategies_remote.sh
```

如果当前没有 SSH / 生产权限，请在 `docs/progress.md` 标记“生产 strategies 表待同步”，不要声称线上已生效。

验证：

必须至少运行：

```bash
python3 -m compileall -q backend/app
./scripts/check.sh
```

如果 `./scripts/check.sh` 因既有前端 lint 或环境问题失败，需要在最终说明中明确失败原因，并说明后端编译是否通过。

交付要求：
- 小步提交
- 修改完成后提交并 push 到 GitHub
- 最终说明包含：
  - 新增文件
  - 策略 key
  - seed 配置
  - 如何运行回测/模拟
  - 验证命令结果
  - 若生产 DB 未同步，明确待执行命令
````

## 设计约束摘要

给执行方的关键提醒：

- 不要重新训练 SuperPnL。
- 不要用 `test` 调参。
- 不要让策略类直接读模型文件或预测文件，读预测必须封装在 signal service。
- 不要在 BitPro 策略里直接调用 OKX / CCXT / DB。
- 不要生成 mock 预测。
- 先证明 `15m` 信号在 `maker 8bps` 成本下还能不能活。
- 先解决换手，再谈实盘。

## 关联 SuperPnL 结果

主结果来自：

```text
outputs/superpnl_top20_365d_l256_h5_15_hd64_e3/REPORT.md
docs/TOP20_12M_EXPERIMENT_REPORT.md
docs/REPRODUCIBILITY_AND_TOP20_UNIVERSE.md
```

核心结果：

| model | horizon | zero-cost test return | conclusion |
| --- | ---: | ---: | --- |
| full_feature_tcn | 5m | -14.49% | 不可用 |
| full_feature_tcn | 15m | +62.46% | 当前唯一主线 |
| full_feature_tcn | 30m | -5.80% | 暂不作为主线 |

当前问题不是模型完全没有信号，而是默认逐分钟翻仓规则换手过高。BitPro 策略层的第一目标是通过阈值、Top-K、再平衡间隔、最短持仓、冷却时间和仓位上限把换手压下来。
