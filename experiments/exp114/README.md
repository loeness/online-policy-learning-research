# E114 · 来源剂量 pilot 及聚合失败修复

- 归属：[H020](../../research/records/H020.md)；角色：`pilot`；执行状态：`COMPLETED`。
- 归档方式：2026-10-20 历史回溯归类；不表示当日运行、预注册或重新验证。

## 设计与已有事实

首 pilot 在最终聚合失败，真实交互保留；v2 为独立重试。

精确种子、比较条件、指标和配置以原协议及有效运行配置为准；科学解释见 H 记录和原始 review。

## 执行与原始证据

- run001 · `FAILED` · [outputs/source_dose_pilot](../../outputs/source_dose_pilot) · [回溯来源记录](runs/run001/provenance.json)
- run002 · `COMPLETED` · [outputs/source_dose_pilot_v2](../../outputs/source_dose_pilot_v2) · [回溯来源记录](runs/run002/provenance.json)

## 成本与来源限制

原记录的成本摘要（不同字段口径不相加）：

```json
{
  "failed_attempt": {
    "environment_steps": 39936,
    "cpu_s": 29.031,
    "wall_s": 31.817
  },
  "successful_attempt": {
    "environment_steps": 39936,
    "cpu_s": 25.016,
    "wall_s": 27.708
  }
}
```

执行 commit/完整命令本次未证实，不能用导入时 HEAD 补造。源码快照、输入和协议哈希保留在原位置。复现说明中的命令是复现入口，不自动当成原执行命令。

## 协议、分析与完整修订历史

- [docs/SOURCE_DOSE_PROTOCOL.md](../../docs/SOURCE_DOSE_PROTOCOL.md)
- [docs/SOURCE_FOLLOWUP_REVIEW.md](../../docs/SOURCE_FOLLOWUP_REVIEW.md)
- [configs/source_dose.json](../../configs/source_dose.json)
- [outputs/source_followup_audit](../../outputs/source_followup_audit)

[复现说明](../../docs/REPRODUCIBILITY.md)。报告/审计仅链接原版本；通过完整性核对不代表假设已获支持。

## 归属与分析版本导航

主生命周期：[H020](../../research/records/H020.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
