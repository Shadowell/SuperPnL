# Implementation Plan: 审查问题修复

**Branch**: `codex/superpnl-audit-fixes` | **Date**: 2026-10-04 | **Spec**: [spec.md](spec.md)

## Summary
按 issue 建立失败回归、最小修复和独立提交；数据→评测→模型契约→打包集成验收。

## Technical Context
Python >=3.10；沿用 NumPy/Pandas/PyTorch，增加 matplotlib 运行依赖和 pytest 测试 extra。
本地文件存储，测试使用临时合成数据，小型 CPU 模型；不下载原始数据或 checkpoint。

## Constitution Check
通过：仅现货研究，不新增特征；四类基准保留；成本来自配置，标签不跨分区。
数据和模型产物留在临时目录；GitHub issues 已获授权，代码不推送、不部署。

## Project Structure
- src/superpnl/data.py：1m 网格、下一开盘收益、purge、缓存版本及原始数据指纹。
- src/superpnl/training.py：资产/现金账本、原始动量、checkpoint 数据契约。
- src/superpnl/metrics.py：初始净值峰值。
- scripts/run_superpnl_experiment.py：缓存校验及一致的分币/月归因。
- scripts/package_superpnl_model.py：数据契约/实际架构校验、失败保留旧包。
- tests/：按领域组织的确定性回归和 CPU CLI 集成测试。
- README.md、docs/ 现有入口：修改后的口径、旧指标失效、重建和测试命令。

## Design Decisions
- next_returns 仍存对数收益，但来自 open[t+2]/open[t+1]。
- 目标权重为 signal/N，使用扣费后净值求可自融资的目标金额；余额为现金。
  净值逐期按真实简单收益推进后再转换为组合 log return，供现有指标函数使用。
- buy-and-hold 初始等权买入一次，不再平衡，期末市值计价。
- 严格拒绝缺失币种、分钟缺口和无效 OHLCV；不填未来值，不把分钟压缩成行。
- train/val 样本尾部按 max_horizon+1 purge，确保标签结束价格早于下一分区。
- 缓存保存 schema version、配置、原始文件 SHA256。数据契约包含这些身份、
  特征及 horizon 顺序、分区/形状、标准化统计量哈希；checkpoint 保存同一契约。
- 打包在临时同级目录完成，验证完成再替换；替换异常恢复旧目录和 tarball。
- 旧缓存/无契约 checkpoint 明确拒绝并提示重建或重训，不猜测来源。

## Verification
逐 issue 先红后绿；全量 pytest；9 个源文件语法检查；小型 CPU 两模型训练、
四类基准、两类模型打包及加载；specify integration status、git diff --check。
无真实原始数据，不宣称修复前报告收益仍成立。

## Dependencies & Execution
数据 #1→#3→#4；指标 #6→账本 #2→动量 #7；打包 #9→#10 可独立进行。
各领域每完成一个 issue 暂停，由根 agent 核对并串行提交，再开始下一个。
缓存 #8 在数据修复完成后实施；契约 #5 在缓存、训练和打包前置完成后实施。
最后 #11、文档与整体审查。分析：FR-001..012 均有任务，无未决澄清或冲突。
