# E130 · 行为候选正式设计：中断与完整重跑

- 归属：[H026](../../research/records/H026.md)；角色：`formal`；执行状态：`COMPLETED`。
- 归档方式：2026-10-20 历史回溯归类；不表示当日运行、预注册或重新验证。

## 设计与已有事实

run001中断仅有355148落盘步下界；run002完整3391488步，不能合成额外独立样本。

精确种子、比较条件、指标和配置以原协议及有效运行配置为准；科学解释见 H 记录和原始 review。

## 执行与原始证据

- run001 · `CANCELLED` · [outputs/retention_candidate_main](../../outputs/retention_candidate_main) · [回溯来源记录](runs/run001/provenance.json)
- run002 · `COMPLETED` · [outputs/retention_candidate_completed](../../outputs/retention_candidate_completed) · [回溯来源记录](runs/run002/provenance.json)

## 成本与来源限制

原记录的成本摘要（不同字段口径不相加）：

```json
{
  "completed_environment_steps": 3391488,
  "completed_cpu_s": 1957.625,
  "completed_wall_s": 1025.3635592,
  "interrupted_persisted_step_lower_bound": 355148,
  "interrupted_total_steps": null
}
```

执行 commit/完整命令本次未证实，不能用导入时 HEAD 补造。源码快照、输入和协议哈希保留在原位置。复现说明中的命令是复现入口，不自动当成原执行命令。

## 协议、分析与完整修订历史

- [docs/RETENTION_CANDIDATE_PROTOCOL.md](../../docs/RETENTION_CANDIDATE_PROTOCOL.md)
- [docs/RETENTION_CANDIDATE_REVIEW.md](../../docs/RETENTION_CANDIDATE_REVIEW.md)
- [configs/retention_candidate.json](../../configs/retention_candidate.json)
- [outputs/retention_candidate_completed_report](../../outputs/retention_candidate_completed_report)
- [outputs/retention_candidate_audit](../../outputs/retention_candidate_audit)
- [outputs/retention_candidate_interrupted_inventory.json](../../outputs/retention_candidate_interrupted_inventory.json)
- [outputs/retention_candidate_pause_receipt.json](../../outputs/retention_candidate_pause_receipt.json)
- [outputs/retention_candidate_environment.json](../../outputs/retention_candidate_environment.json)
- [outputs/retention_candidate_protocol_freeze.json](../../outputs/retention_candidate_protocol_freeze.json)
- [outputs/retention_candidate_completed_console.log](../../outputs/retention_candidate_completed_console.log)
- [outputs/retention_candidate_preflight_checks.xml](../../outputs/retention_candidate_preflight_checks.xml)
- [outputs/retention_candidate_producer_checks.xml](../../outputs/retention_candidate_producer_checks.xml)
- [outputs/retention_candidate_producer_checks_v2.xml](../../outputs/retention_candidate_producer_checks_v2.xml)
- [outputs/retention_candidate_producer_checks_final.xml](../../outputs/retention_candidate_producer_checks_final.xml)
- [outputs/retention_candidate_software_checks.xml](../../outputs/retention_candidate_software_checks.xml)
- [outputs/retention_candidate_publication_preflight.json](../../outputs/retention_candidate_publication_preflight.json)
- [outputs/retention_candidate_publication_preflight_v2.json](../../outputs/retention_candidate_publication_preflight_v2.json)

[复现说明](../../docs/REPRODUCIBILITY.md)。报告/审计仅链接原版本；通过完整性核对不代表假设已获支持。

## 归属与分析版本导航

主生命周期：[H026](../../research/records/H026.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
- [analysis003](analyses/analysis003/README.md)
