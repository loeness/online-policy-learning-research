# E102 — Retention stream Stage 2 pilot v2

- **H:** H017 ([lifecycle](../../research/records/H017.md)).
- **Kind / status:** pilot v2 / COMPLETED.
- **Import:** retrospective classification dated 2026-10-20; not a new run or preregistration.

One low-budget pilot, seed 18997, five methods, 1,024 steps per bout. Cost: 61,440 online interactions. It exercised candidate write/reject paths; it remains a pipeline diagnostic and is not a formal seed. Full-stream reward/raw step and natural-eviction recovery are intended outcomes (higher is better). Historical approved budget: UNKNOWN; CPU/wall for this attempt were not recorded.

- **Config / protocol:** [configs/retention_stream_stage2.json](../../configs/retention_stream_stage2.json); [docs/RETENTION_STREAM_STAGE2_PROTOCOL.md](../../docs/RETENTION_STREAM_STAGE2_PROTOCOL.md); [docs/RETENTION_STREAM_STAGE2_METHOD.md](../../docs/RETENTION_STREAM_STAGE2_METHOD.md).
- **Raw output / source manifest:** [outputs/retention_stream_stage2_pilot_v2](../../outputs/retention_stream_stage2_pilot_v2); [outputs/retention_stream_stage2_pilot_v2/source_snapshot/manifest.json](../../outputs/retention_stream_stage2_pilot_v2/source_snapshot/manifest.json).
- **Report:** no standalone pilot report; pilot coverage/rationale is documented in [docs/RETENTION_STREAM_STAGE2_REVIEW.md](../../docs/RETENTION_STREAM_STAGE2_REVIEW.md); main report lists aggregate pilot costs.
- **Audit:** raw pilot summary and branch records; no separate pilot audit artifact identified.
- **Figure:** NOT_APPLICABLE — no pilot-specific figure.
- **Historical command:** exact argv not recovered. Reproduction recipe (not historical command): use [scripts/run_retention_stream.py](../../scripts/run_retention_stream.py) with preserved v2 pilot config and a new output directory.
- **Code/data gaps:** no full execution SHA; manifest has file hashes only. Frozen config and manifest do not bind raw dataset path/hash.
- **Run provenance:** [run001](runs/run001/provenance.json).

E101 is retained as the distinct failed-to-reach-write pilot; E103 is the main cohort.


## 归属与分析版本导航

主生命周期：[H017](../../research/records/H017.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
