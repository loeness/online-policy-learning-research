# E100 — Consolidation retention Stage 1 exploratory cohort

- **H:** H016 ([lifecycle](../../research/records/H016.md)).
- **Kind / status:** exploratory main cohort / COMPLETED.
- **Import:** retrospective classification dated 2026-10-20; not a new run or preregistration.

Seeds 1009, 1013, 1019; local_only, mean, weighted, and random extraction arms; targets A0+, A2+, B2+, A0−. Frozen observer comparisons used paired starts/actions; no new training/probe occurred. Primary metric is signed episode return (higher is better), 8 episodes per target/arm/source, at most 64 steps.

**Results:** Mean minus deleted-W0 was +0.480 on A0+ and +0.557 on exploratory A2+ (seed directions mixed); mean was −0.887 on B2+ and −0.883 on opposite-goal A0−. Weighted was below simple mean for every A0+ and A2+ seed. These are descriptive results from reused exploratory source seeds, not independent confirmation.

**Actual cost:** 22,746 new observer/evaluation interactions; 0 new training steps. The run reused 98,304 historical source-training steps (110,207 relevant prior interactions including probe/evaluation), not rerun or counted as new cost. CPU 10.50 s; wall 10.89 s. Historical approved budget: UNKNOWN.

- **Config / protocol:** [configs/consolidation_retention_stage1.json](../../configs/consolidation_retention_stage1.json); [docs/CONSOLIDATION_RETENTION_STAGE1_PROTOCOL.md](../../docs/CONSOLIDATION_RETENTION_STAGE1_PROTOCOL.md); [docs/CONSOLIDATION_RETENTION_STAGE1_REVIEW.md](../../docs/CONSOLIDATION_RETENTION_STAGE1_REVIEW.md).
- **Raw output / source manifest:** [outputs/consolidation_retention_stage1](../../outputs/consolidation_retention_stage1); [outputs/consolidation_retention_stage1/source_snapshot/manifest.json](../../outputs/consolidation_retention_stage1/source_snapshot/manifest.json).
- **Source checkpoints:** `outputs/write_decision_causal_control/seed_{1009,1013,1019}/trial_rollback/checkpoints/checkpoint_step_00032768.pt`. Source audit hashes are in report v3; foundation SHA-256 `e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222`.
- **Reports:** [outputs/consolidation_retention_stage1_report/report.md](../../outputs/consolidation_retention_stage1_report/report.md), `..._v2/report.md`, and corrected accounting `..._v3/report.md`.
- **Audit / figures:** [outputs/consolidation_retention_stage1/postrun_audit.json](../../outputs/consolidation_retention_stage1/postrun_audit.json); report v3 `figure.png` and `figure.pdf`.
- **Historical command:** exact argv not recovered. Documented reproduction recipe (not historical command): `.venv/Scripts/python.exe scripts/run_consolidation_retention.py --output <new empty directory>`.
- **Execution SHA:** MISSING; source manifest has per-file hashes only.
- **Run provenance:** [run001](runs/run001/provenance.json).

This frozen transfer assay does not measure automatic routing, general retention, eviction recovery, or storage optimality. E101–E103 are separate natural-eviction experiments.


## 归属与分析版本导航

主生命周期：[H016](../../research/records/H016.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
- [analysis003](analyses/analysis003/README.md)
- [analysis004](analyses/analysis004/README.md)
