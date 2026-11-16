# E067：S3 离线 support 与 ambiguity diagnostic

**H：** [H013](../../research/records/H013.md) · **性质：** 对已保存探测记录的离线科学诊断。**状态：** 原分析已完成；2026-10-20 回顾性整理。

## 设计与决策

从 S1b 三个正式种子的 independent `probe_steps.csv` 离线重建 16/32/64/128/256/512 行 profile，并检查动作 0/1 censoring 与人工同分 prototype 的 abstention。Grouper 只接收 numeric profile；family/context 名只在分组后用于 observer audit。目的在于辨明 probe 支持不足、局部身份歧义和平局处理，不估计策略收益。

## 结果、成本与限制

读取历史 9,216 行，新增环境交互为 0；动作 0/1 censorship 使用 4,512 行。平均分配 coverage 在 16/32 行时为 27.8%，64 行 50.0%、128 行 88.9%，256/512 行 100%。在低 coverage 下已解析 pair 的零错误覆盖很少样本。只看动作 0/1 仍能恢复 family，但 A0/A1/A2 均可与三个 A references 匹配，区别藏在未观测动作 2/3。三个人工同分压力查询都返回 `-1 / ambiguous_match`。结论仅限保存的三条 probe 流和手工 tie state，不证明安全共享或通用识别性。

## 产物与复现

- 方法：[SHARED_WRITE_SUPPORT_DIAGNOSTIC.md](../../docs/SHARED_WRITE_SUPPORT_DIAGNOSTIC.md)；固定阈值配置：[shared_write_auto.json](../../configs/shared_write_auto.json)；分析入口：[diagnose_shared_write_support.py](../../scripts/diagnose_shared_write_support.py)。历史协议中给出的命令仅为重建参考，实际 shell 命令未验证。
- [Summary](../../outputs/knowledge_shared_write_support_diagnostic/summary.json)、[assignments](../../outputs/knowledge_shared_write_support_diagnostic/assignments.csv)、[pairwise audit](../../outputs/knowledge_shared_write_support_diagnostic/pairwise_audit.csv)、[tie checks](../../outputs/knowledge_shared_write_support_diagnostic/tied_prototype_checks.json)、[coverage figure](../../outputs/knowledge_shared_write_support_diagnostic/support_coverage_error.png)。原始输入仍在 S1b outputs。
- `summary.json` 内记录的输入绝对路径仍指向旧的 `D:\MovedFromC:\ChatGPT\online` 根；此导入按协议和当前仓库结构记下方精确相对路径。它不改变输入文件，也不表示本次读取过原始 trace。
- 单个离线 diagnostic root [outputs/knowledge_shared_write_support_diagnostic](../../outputs/knowledge_shared_write_support_diagnostic) 的输出、输入路径、代码路径和来源状态见 [runs/run001/provenance.json](runs/run001/provenance.json)；这是一个分析 cohort，不是新增环境交互。

**成本限制：** 无新的环境交互；CPU/墙钟分析成本未在该摘要核验。执行 Git commit 与独立分析 source snapshot 未找到，相关 code provenance 存在缺口。

## 归属与分析版本导航

主生命周期：[H013](../../research/records/H013.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
