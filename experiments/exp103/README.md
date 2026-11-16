# E103 — Retention stream Stage 2 exploratory main cohort

- **H:** H017 ([lifecycle](../../research/records/H017.md)).
- **Kind / status:** exploratory main cohort / COMPLETED.
- **Import:** retrospective classification dated 2026-10-20; not a new run or preregistration.

Three deployment seeds (19001, 19013, 19031) × low/wide capacities × five methods (independent, direct_shared, mean, mean_reuse_gate, retention_gate): 30 branches, each 24,576 raw steps. Seeds share a fixed W0 and are not independent pretraining replicates.

- **Metrics:** full-stream total reward/reward per raw step and natural-eviction recovery, higher is better.
- **Results:** mean exceeded independent in all six seed×budget comparisons; direct_shared exceeded mean in all six. The prior-use gate lowered all six full-stream comparisons against mean. Retention gate showed no stable advantage over mean_reuse_gate (low worse, wide better) and approved 7/127 write tests. Only low capacity had natural evictions; wide results do not establish eviction recovery.
- **Actual cost:** 737,280 interactions; CPU 892.06 s; wall 461.48 s. Including pilot E101 and E102: 829,440 interactions, CPU 998.95 s, sequential cohort wall 531.49 s. Historical approved budget: UNKNOWN.

- **Config/protocol:** [configs/retention_stream_stage2.json](../../configs/retention_stream_stage2.json); [docs/RETENTION_STREAM_STAGE2_METHOD.md](../../docs/RETENTION_STREAM_STAGE2_METHOD.md), `..._PROTOCOL.md`, `..._DIAGNOSTICS.md`, `..._REVIEW.md`.
- **Raw output / source manifest:** [outputs/retention_stream_stage2_exploratory](../../outputs/retention_stream_stage2_exploratory); [outputs/retention_stream_stage2_exploratory/source_snapshot/manifest.json](../../outputs/retention_stream_stage2_exploratory/source_snapshot/manifest.json).
- **Final report/analysis/figures:** [outputs/retention_stream_stage2_report_v2/report.md](../../outputs/retention_stream_stage2_report_v2/report.md), `analysis.json`, `figure.png`, `figure.pdf`; earlier report retained at [outputs/retention_stream_stage2_report/](../../outputs/retention_stream_stage2_report).
- **Audit:** [outputs/retention_stream_stage2_audit/artifact_audit.json](../../outputs/retention_stream_stage2_audit/artifact_audit.json).
- **Historical command:** exact argv not recovered. Reproduction recipe (not historical command): invoke [scripts/run_retention_stream.py](../../scripts/run_retention_stream.py) with frozen main config and a new output directory.
- **Execution/data gaps:** no full Git execution SHA; source snapshot has file hashes only. Frozen config and snapshot do not bind raw dataset path/hash.
- **Run provenance:** [run001](runs/run001/provenance.json).

This is an exploratory C1 mechanism study, not evidence about unknown boundaries, reward-conflict safety, parent replacement, or strict total-byte-optimal replay.


## 归属与分析版本导航

主生命周期：[H017](../../research/records/H017.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
- [analysis003](analyses/analysis003/README.md)
