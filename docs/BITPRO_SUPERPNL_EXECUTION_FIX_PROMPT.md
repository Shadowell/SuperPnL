# BitPro SuperPnL Execution Fix Prompt

本文档是一份可直接复制给 BitPro 项目编码代理的修复提示词。

目标：修复 `SuperPnL15mLowTurnoverStrategy` 的执行层问题。当前优先级不是重新训练模型，而是确保策略的真实仓位读取、`top_k`、`max_total_position`、旧仓位清理、下单金额和诊断日志正确生效。

```text
你现在在 BitPro 项目中修复 SuperPnL 15m 实时推理低换手现货策略的执行逻辑。当前问题不是先优化模型，而是策略层仓位、top-k、max_total_position、下单金额和诊断日志明显不正确。

请先阅读并遵守：

1. AGENTS.md
2. README.md
3. docs/spec.md
4. docs/progress.md
5. docs/contracts/module_map_v2.md
6. docs/strategy_development_guide.md
7. backend/app/core/execution/base_strategy.py
8. backend/app/strategies/superpnl_15m_low_turnover_strategy.py
9. backend/app/services/superpnl_model_inference_service.py
10. backend/app/services/superpnl_feature_builder.py
11. backend/app/services/strategy_registry.py
12. data/seed/strategies.json

背景：

当前线上/模拟盘观察到这些异常：

1. `max_total_position` 没有生效。
   - 账户约 9873 USDT，但策略持仓接近满仓。
   - 配置里应限制总仓位，例如 `max_total_position=0.6`，保守测试可先用 `0.1`。

2. `top_k` 没有生效，或旧仓位没有被清掉。
   - 如果 `top_k=3`，理论上最多主仓 3 个币。
   - 当前实际持仓接近 10 个币，说明非 top-k 仓位没有在 rebalance 中正确减仓/清仓。

3. 策略内部当前仓位和 broker 实际持仓不一致。
   - 诊断日志显示 LTC 当前仓位为 0%，但持仓面板实际有 LTC/USDT 仓位。
   - 必须以 broker / account state 的真实持仓为唯一仓位来源，策略内部缓存只能作为辅助状态，不能覆盖真实持仓。

4. 下单金额异常小。
   - 成交明细出现 0.01 USDT、0.03 USDT、0.04 USDT 订单。
   - 这说明 target_position 到 order qty 的计算可能错了，或者缺少最小下单金额过滤。

本次目标：

修复 `SuperPnL15mLowTurnoverStrategy` 的执行层逻辑，让它成为一个真正受控的 long-only top-k 组合策略。

严禁事项：

- 不要重新训练 SuperPnL 模型。
- 不要修改 BitPro 策略引擎核心框架，除非明确发现框架无法支持，且必须先说明。
- 不要用 mock/random/momentum/synthetic 信号替代 SuperPnL 实时推理。
- 不要读取历史 prediction `.npz` 作为模拟盘或实盘信号。
- 不要绕过 broker / BaseStrategy 直接调用交易所 API。
- 不要只改日志不改实际下单逻辑。

必须修复的核心逻辑：

一、真实仓位读取

实现或修复一个统一方法，例如：

```python
def _get_broker_position_snapshot(self) -> dict[str, PositionSnapshot]:
    ...
```

要求：

- 从 broker / self.state.positions / BitPro 框架认可的账户状态读取真实持仓。
- symbol 使用 BitPro 格式，例如 `BTC/USDT`。
- 每个 symbol 至少包含：
  - quantity
  - mark_price / last_price / close
  - notional_usdt
  - avg_entry_price
  - unrealized_pnl
- 如果策略内部维护 position state，必须用 broker snapshot 校准。
- 诊断日志中同时输出 broker_position 和 strategy_cached_position，方便发现不同步。

二、仓位比例计算

用账户权益计算真实仓位比例：

```python
account_equity = cash_usdt + sum(position_notional_usdt)
current_position_ratio = position_notional_usdt / account_equity
```

要求：

- 不允许把 quantity 当成仓位比例。
- 不允许把 target_position 直接当成下单金额。
- 如果无法取得 account_equity，必须跳过交易并输出 `skip_account_equity_unavailable`。

三、top-k 目标组合生成

每次 rebalance 时：

1. 批量拿到所有 universe symbol 的 SuperPnL 信号。
2. 只保留：

```python
pred_ret > threshold_bps / 10000
```

3. 按 `pred_ret` 降序排序。
4. 选择前 `top_k`。
5. 对候选生成目标仓位。
6. 非候选 symbol 的目标仓位必须是 0，除非该 symbol 仍处于最短持仓保护期。

目标仓位必须满足：

```python
sum(target_position.values()) <= max_total_position
target_position[symbol] <= max_position_per_symbol
```

建议实现：

```python
selected = candidates[:top_k]
slot = min(max_position_per_symbol, max_total_position / max(1, len(selected)))
target_positions = {symbol: slot for symbol in selected}
```

如果候选为空，目标组合应为空，即全部目标仓位为 0，但仍要遵守最短持仓保护。

四、清理非 top-k 旧仓位

这是本次重点。

每次 rebalance 后，对所有当前实际持仓 symbol：

- 如果 symbol 不在目标 top-k 中；
- 且没有处于 `min_holding_bars` 保护期；
- 则必须卖出到目标仓位 0。

不能只买新的 top-k，而不处理旧仓位。

如果处于最短持仓保护期：

- 不平仓；
- 诊断输出 `skip_min_holding`；
- 但到期后的下一次 rebalance 必须允许清仓。

五、max_total_position 强制约束

下单前必须二次校验：

```python
target_total = sum(target_positions.values())
if target_total > max_total_position:
    scale = max_total_position / target_total
    target_positions = {s: p * scale for s, p in target_positions.items()}
```

并在诊断日志中输出：

- target_total_before_cap
- target_total_after_cap
- max_total_position
- cap_applied

六、下单数量计算

买入：

```python
target_notional = account_equity * target_position_ratio
current_notional = current_position_ratio * account_equity
delta_notional = target_notional - current_notional
qty = delta_notional / close
```

卖出：

```python
target_notional = account_equity * target_position_ratio
current_notional = current_position_ratio * account_equity
delta_notional = current_notional - target_notional
qty = min(current_quantity, delta_notional / close)
```

要求：

- 买卖都基于 notional 差值。
- qty 不能为负。
- qty 不能超过当前持仓数量。
- close <= 0 时跳过并输出 `skip_invalid_price`。
- account_equity <= 0 时跳过并输出 `skip_account_equity_unavailable`。

七、最小下单金额过滤

新增配置：

```json
"min_order_notional_usdt": 5
```

默认至少 5 USDT，也可以按 OKX 规则调大。

下单前：

```python
if abs(delta_notional) < min_order_notional_usdt:
    skip_qty_too_small
```

不要再产生 0.01 USDT、0.03 USDT 这种订单。

八、保守默认配置

先把 seed 里的 SuperPnL 策略配置改保守，用于验证执行逻辑：

```json
{
  "threshold_bps": 30,
  "top_k": 1,
  "rebalance_interval_bars": 30,
  "min_holding_bars": 60,
  "cooldown_bars": 60,
  "max_position_per_symbol": 0.1,
  "max_total_position": 0.1,
  "min_order_notional_usdt": 5,
  "fee_bps": 8,
  "slippage_bps": 0
}
```

先用 10% 总仓位验证策略，不要直接满仓 Top20。

九、诊断日志增强

每次 bar 或 rebalance 诊断至少输出：

```json
{
  "type": "bar_diag",
  "decision": "...",
  "decision_label": "...",
  "symbol": "BTC/USDT",
  "bar_ts_ms": 1234567890,
  "close": 123.45,

  "pred_ret": 0.0012,
  "pred_ret_bps": 12.0,
  "threshold_bps": 30,
  "rank": 1,

  "account_equity": 9873.39,
  "cash_usdt": 3900.12,

  "broker_quantity": 0.123,
  "broker_notional": 100.0,
  "broker_position_ratio": 0.0101,

  "strategy_cached_position_ratio": 0.0,
  "target_position_ratio": 0.1,

  "target_notional": 987.33,
  "current_notional": 100.0,
  "delta_notional": 887.33,
  "order_qty": 0.0116,
  "order_notional": 887.33,

  "target_total_before_cap": 0.2,
  "target_total_after_cap": 0.1,
  "max_total_position": 0.1,
  "cap_applied": true,

  "top_k": 1,
  "selected_symbols": ["ETH/USDT"],
  "current_holding_symbols": ["ETH/USDT", "LTC/USDT"],
  "symbols_to_close": ["LTC/USDT"],

  "rebalance_interval_bars": 30,
  "min_holding_bars": 60,
  "cooldown_bars": 60,
  "min_order_notional_usdt": 5
}
```

必须新增或确认这些 decision：

```python
DECISION_LABELS = {
    "skip_account_equity_unavailable": "未交易：账户权益不可用",
    "skip_invalid_price": "未交易：价格无效",
    "skip_qty_too_small": "未交易：下单金额低于最小限制",
    "skip_no_signal": "未交易：SuperPnL 实时信号不可用",
    "skip_below_threshold": "未交易：预测收益低于阈值",
    "skip_rebalance_interval": "未交易：未到再平衡时间",
    "skip_min_holding": "未卖出：未达到最短持仓时间",
    "skip_cooldown": "未买入：仍在冷却期",
    "rebalance": "组合再平衡",
    "buy_filled": "买入成交",
    "sell_filled": "卖出成交",
    "close_non_topk": "清理非Top-K旧仓位",
    "broker_error": "下单失败"
}
```

十、测试要求

必须新增或更新单元测试，覆盖：

1. `top_k=1` 时最多只有 1 个非保护期目标持仓。
2. `max_total_position=0.1` 时目标总仓位不超过 10%。
3. 已有旧仓位不在 top-k，且超过 min_holding，必须生成清仓动作。
4. 已有旧仓位不在 top-k，但未超过 min_holding，不清仓并输出 `skip_min_holding`。
5. `delta_notional < min_order_notional_usdt` 时不下单。
6. broker 实际持仓和 strategy cache 不一致时，以 broker 实际持仓为准。
7. target_position -> qty 的计算使用 account_equity 和 close，不能产生 0.01 USDT 级别异常订单。
8. 候选为空时，目标组合为空，并清理非保护期旧仓位。

验证命令：

```bash
python3 -m compileall -q backend/app
./scripts/check.sh
```

如果 `./scripts/check.sh` 因已有前端 lint 或环境问题失败，最终说明必须明确失败原因，并说明后端编译和新增测试是否通过。

交付要求：

- 小步提交并 push。
- 最终说明包含：
  - 修改了哪些文件
  - 修复了哪些执行逻辑
  - 新的保守 seed 参数
  - 如何验证 max_total_position/top_k/min_order_notional 生效
  - 验证命令结果
  - 如果生产 DB seed 未同步，明确需要执行的同步命令
```
