# SuperPnL Constitution

## Core Principles

### I. 可交易 PnL 与现货边界
必须以现货 long-only 目标仓位 0..1 研究 PnL、风险、回撤和换手；不扩展到杠杆、永续或实盘。

### II. 因果数据与可追溯性
输入仅使用决策时刻及以前数据，标准化仅在训练集拟合。
新增特征必须解释泄漏风险；缓存与权重必须验证数据契约。

### III. 公平评测
必须同时报告 no-trade、buy-and-hold、OHLCV-only 和有因子模型，明确成本假设。
回测价格、收益与资金流必须通过小型确定性案例核算。

### IV. 小步修复与验证
行为缺陷先增加能复现失败的回归测试，再修复并验证；只提交任务相关文件。
每个 issue 对应独立提交，测试结果与历史实验结论分别陈述。

### V. 数据与授权保护
不得提交原始数据、权重、artifacts、虚拟环境或秘密；不得覆盖用户已有修改。
没有明确授权不得推送、PR、部署、付费任务或真实交易。

## Additional Constraints
使用现有 Python、NumPy、Pandas、PyTorch 技术栈。不加入盘口、成本、流动性训练特征。

## Development Workflow
采用 specify → clarify → plan → tasks → analyze → implement → converge。
仅保留必需规格与长期文档；过程日志留在对话。遵守 AGENTS.md。

## Governance
用户当前指令优先；原则变更须注明日期及版本，新增原则升 minor，破坏性变更升 major。
合入前核对回归测试、授权范围与上述原则。

**Version**: 1.0.0 | **Ratified**: 2026-10-04 | **Last Amended**: 2026-10-04
