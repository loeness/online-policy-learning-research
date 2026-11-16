# E122 · 随机流所选反馈预测校准

- 归属：[H023](../../research/records/H023.md)；角色：`formal`；执行状态：`COMPLETED`。
- 归档方式：2026-10-20 历史回溯归类；不表示当日运行、预注册或重新验证。

## 设计与已有事实

六种子×两关系；768所选标签。预测改善不是写入策略价值增量。

精确种子、比较条件、指标和配置以原协议及有效运行配置为准；科学解释见 H 记录和原始 review。

## 执行与原始证据

- run001 · `COMPLETED` · [outputs/retention_value_calibration_main](../../outputs/retention_value_calibration_main) · [回溯来源记录](runs/run001/provenance.json)

## 成本与来源限制

原记录的成本摘要（不同字段口径不相加）：

```json
{
  "environment_steps": 1216512,
  "cpu_s": 1376.031,
  "wall_s": 472.916
}
```

执行 commit/完整命令本次未证实，不能用导入时 HEAD 补造。源码快照、输入和协议哈希保留在原位置。复现说明中的命令是复现入口，不自动当成原执行命令。

## 协议、分析与完整修订历史

- [docs/ORDERED_RETENTION_VALIDATION_REVIEW.md](../../docs/ORDERED_RETENTION_VALIDATION_REVIEW.md)
- [docs/RETENTION_VALUE_STREAM_PROTOCOL.md](../../docs/RETENTION_VALUE_STREAM_PROTOCOL.md)
- [docs/RETENTION_VALUE_STREAM_METHOD.md](../../docs/RETENTION_VALUE_STREAM_METHOD.md)
- [configs/retention_value_stream.json](../../configs/retention_value_stream.json)
- [outputs/retention_value_calibration_report](../../outputs/retention_value_calibration_report)
- [outputs/retention_value_calibration_main_audit.json](../../outputs/retention_value_calibration_main_audit.json)

[复现说明](../../docs/REPRODUCIBILITY.md)。报告/审计仅链接原版本；通过完整性核对不代表假设已获支持。

## 归属与分析版本导航

主生命周期：[H023](../../research/records/H023.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
- [analysis003](analyses/analysis003/README.md)
