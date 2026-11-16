# E115 · 来源剂量与候选内容正式比较

- 归属：[H020](../../research/records/H020.md)；角色：`formal`；执行状态：`COMPLETED`。
- 归档方式：2026-10-20 历史回溯归类；不表示当日运行、预注册或重新验证。

## 设计与已有事实

保留全部报告修订与最终审计；512 逻辑分支不等于512次实际付费续程。

精确种子、比较条件、指标和配置以原协议及有效运行配置为准；科学解释见 H 记录和原始 review。

## 执行与原始证据

- run001 · `COMPLETED` · [outputs/source_dose_main](../../outputs/source_dose_main) · [回溯来源记录](runs/run001/provenance.json)

## 成本与来源限制

原记录的成本摘要（不同字段口径不相加）：

```json
{
  "environment_steps": 5111808,
  "campaign_cpu_s": 3764.359,
  "campaign_wall_s": 956.842
}
```

执行 commit/完整命令本次未证实，不能用导入时 HEAD 补造。源码快照、输入和协议哈希保留在原位置。复现说明中的命令是复现入口，不自动当成原执行命令。

## 协议、分析与完整修订历史

- [docs/SOURCE_DOSE_PROTOCOL.md](../../docs/SOURCE_DOSE_PROTOCOL.md)
- [docs/SOURCE_FOLLOWUP_REVIEW.md](../../docs/SOURCE_FOLLOWUP_REVIEW.md)
- [configs/source_dose.json](../../configs/source_dose.json)
- [outputs/source_dose_main_report](../../outputs/source_dose_main_report)
- [outputs/source_followup_main_report](../../outputs/source_followup_main_report)
- [outputs/source_followup_report](../../outputs/source_followup_report)
- [outputs/source_followup_report_combo](../../outputs/source_followup_report_combo)
- [outputs/source_followup_report_combo_v2](../../outputs/source_followup_report_combo_v2)
- [outputs/source_followup_report_combo_v3](../../outputs/source_followup_report_combo_v3)
- [outputs/source_followup_report_combo_v4](../../outputs/source_followup_report_combo_v4)
- [outputs/source_followup_report_combo_v5](../../outputs/source_followup_report_combo_v5)
- [outputs/source_followup_report_combo_v6](../../outputs/source_followup_report_combo_v6)
- [outputs/source_followup_audit](../../outputs/source_followup_audit)

[复现说明](../../docs/REPRODUCIBILITY.md)。报告/审计仅链接原版本；通过完整性核对不代表假设已获支持。

## 归属与分析版本导航

主生命周期：[H020](../../research/records/H020.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
- [analysis003](analyses/analysis003/README.md)
- [analysis004](analyses/analysis004/README.md)
- [analysis005](analyses/analysis005/README.md)
- [analysis006](analyses/analysis006/README.md)
- [analysis007](analyses/analysis007/README.md)
- [analysis008](analyses/analysis008/README.md)
- [analysis009](analyses/analysis009/README.md)
- [analysis010](analyses/analysis010/README.md)
- [analysis011](analyses/analysis011/README.md)
