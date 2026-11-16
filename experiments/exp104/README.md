# E104 — Frozen score calibration pilot

- **H:** H018 ([lifecycle](../../research/records/H018.md)).
- **Kind / status:** single-seed pilot / COMPLETED.
- **Import:** retrospective classification dated 2026-10-20; not a new run or preregistration.

Seed 19701, low budget. It checked the frozen score, delayed observer, and reporting pipeline; it is not main-cohort evidence. Three candidates were scored, with rank statistics not estimable.

- **Primary metric:** Spearman association between frozen original mean-lower score and conditional future 128-step frozen-policy reward/raw-step gain; higher is better.
- **Actual cost:** 69,632 interactions (12,288 learner + 57,344 observer); CPU 45.047 s; wall 51.889 s. Historical approved budget: UNKNOWN.
- **Raw output/config:** [outputs/score_calibration_pilot](../../outputs/score_calibration_pilot); [configs/score_calibration.json](../../configs/score_calibration.json).
- **Report / analysis / figures:** latest pilot report [outputs/score_calibration_pilot_report_v2/report.md](../../outputs/score_calibration_pilot_report_v2/report.md), `analysis.json`, `figure.png`, `figure.pdf`; prior report retained at [outputs/score_calibration_pilot_report/](../../outputs/score_calibration_pilot_report).
- **Audit:** [outputs/score_calibration_audit/artifact_audit.json](../../outputs/score_calibration_audit/artifact_audit.json) includes pilot cohort audit.
- **Protocol/review:** [docs/SCORE_CALIBRATION_PROTOCOL.md](../../docs/SCORE_CALIBRATION_PROTOCOL.md); [docs/SCORE_CALIBRATION_REVIEW.md](../../docs/SCORE_CALIBRATION_REVIEW.md).
- **Code manifest:** [outputs/score_calibration_pilot/source_snapshot/manifest.json](../../outputs/score_calibration_pilot/source_snapshot/manifest.json); it does not bind raw dataset/foundation path and checksum.
- **Historical command:** exact argv not recovered. Reproduction recipe (not historical command): run [scripts/run_score_calibration.py](../../scripts/run_score_calibration.py) with pilot config and a new output directory.
- **Execution SHA:** MISSING; manifest contains per-file hashes only.
- **Run provenance:** [run001](runs/run001/provenance.json).

E105 is the separate three-seed exploratory main cohort.


## 独立审计产物

- [outputs/score_calibration_pilot_audit.json](../../outputs/score_calibration_pilot_audit.json)

## 归属与分析版本导航

主生命周期：[H018](../../research/records/H018.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
- [analysis003](analyses/analysis003/README.md)
- [analysis004](analyses/analysis004/README.md)
