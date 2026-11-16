# E099 — Consolidation retention Stage 1 pilot

- **H:** H016 ([lifecycle](../../research/records/H016.md)).
- **Kind / status:** pipeline pilot / COMPLETED.
- **Import:** retrospective classification dated 2026-10-20; not a new run or preregistration.

Seed 1009; local_only, mean, weighted, and random arms. This pilot checked the frozen extraction/deletion pipeline, not retention performance. It completed 974 observer/evaluation interactions and 0 training steps; CPU 1.66 s, wall 1.94 s. Metric was signed episode return and pipeline completion (higher return is better). Historical approved budget: UNKNOWN.

- **Config / protocol:** [configs/consolidation_retention_stage1.json](../../configs/consolidation_retention_stage1.json); [docs/CONSOLIDATION_RETENTION_STAGE1_PROTOCOL.md](../../docs/CONSOLIDATION_RETENTION_STAGE1_PROTOCOL.md).
- **Raw output / source manifest:** [outputs/consolidation_retention_stage1_pilot](../../outputs/consolidation_retention_stage1_pilot); [outputs/consolidation_retention_stage1_pilot/source_snapshot/manifest.json](../../outputs/consolidation_retention_stage1_pilot/source_snapshot/manifest.json).
- **Input:** [data/foundation.pt](../../data/foundation.pt), SHA-256 `e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222`; source checkpoint [outputs/write_decision_causal_control/seed_1009/trial_rollback/checkpoints/checkpoint_step_00032768.pt](../../outputs/write_decision_causal_control/seed_1009/trial_rollback/checkpoints/checkpoint_step_00032768.pt).
- **Audit:** [outputs/consolidation_retention_stage1_pilot/seed_1009/source_audit.json](../../outputs/consolidation_retention_stage1_pilot/seed_1009/source_audit.json).
- **Report:** no pilot-only report; pilot cost is discussed in [docs/CONSOLIDATION_RETENTION_STAGE1_REVIEW.md](../../docs/CONSOLIDATION_RETENTION_STAGE1_REVIEW.md) and [outputs/consolidation_retention_stage1_report_v3/report.md](../../outputs/consolidation_retention_stage1_report_v3/report.md).
- **Figure:** NOT_APPLICABLE — no pilot-specific figure was emitted.
- **Historical command:** exact argv not recovered. Reproduction recipe (not historical command): `.venv/Scripts/python.exe scripts/run_consolidation_retention.py --output <new empty directory>`.
- **Execution SHA:** MISSING; source manifest has per-file hashes only. The archive context commit is not an execution SHA.
- **Run provenance:** [run001](runs/run001/provenance.json).

E100 is the separate 3-seed exploratory cohort; this pilot does not validate its performance claims.


## 归属与分析版本导航

主生命周期：[H016](../../research/records/H016.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
