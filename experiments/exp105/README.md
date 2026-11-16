# E105 — Frozen score calibration main cohort

- **H:** H018 ([lifecycle](../../research/records/H018.md)).
- **Kind / status:** exploratory frozen-score main cohort / COMPLETED.
- **Import:** retrospective classification dated 2026-10-20; not a new run or preregistration.

Three seeds (19703, 19717, 19727) × low/wide budget: six branches. Scores were frozen before later natural recurrence tests; observer did not feed back to learners. The 47 candidates are nested in these six seed×budget cells, not independent seeds.

- **Primary metric:** within-seed Spearman correlation of `original_mean_lower` with conditional 128-step frozen-policy reward/raw-step gain; higher is better. Frozen exploratory criterion required mean rho ≥ 0.3 and at least two positive seed rhos for each budget.
- **Results:** 47 candidates, 35 conditionally beneficial, 12 ambiguous, 0 harmful. 27 conditionally beneficial candidates were not approved. Mean primary rho: −0.043 (low), −0.325 (wide); frozen criterion failed at both budgets. No harmful outcomes were observed, so gate risk-rejection accuracy is unknown. Codec lower did not change rankings; this does not establish a causal effect of uncertainty penalty.
- **Actual cost:** 1,040,384 interactions (147,456 learner + 892,928 observer); CPU 535.719 s; wall 283.846 s. Including E104: 1,110,016 interactions; CPU 580.77 s; sequential wall 335.74 s. Historical approved budget: UNKNOWN.
- **Raw output / config:** [outputs/score_calibration_main](../../outputs/score_calibration_main); [configs/score_calibration.json](../../configs/score_calibration.json).
- **Report / analysis / figure:** [outputs/score_calibration_report/report.md](../../outputs/score_calibration_report/report.md), `analysis.json`, `figure.png`, `figure.pdf`.
- **Audit:** [outputs/score_calibration_audit/artifact_audit.json](../../outputs/score_calibration_audit/artifact_audit.json); historical software validation in `software_validation.json` (not run during this import).
- **Protocol/review:** [docs/SCORE_CALIBRATION_PROTOCOL.md](../../docs/SCORE_CALIBRATION_PROTOCOL.md); [docs/SCORE_CALIBRATION_REVIEW.md](../../docs/SCORE_CALIBRATION_REVIEW.md).
- **Code manifest:** [outputs/score_calibration_main/source_snapshot/manifest.json](../../outputs/score_calibration_main/source_snapshot/manifest.json).
- **Historical command:** exact argv not recovered. Reproduction recipe (not historical command): run [scripts/run_score_calibration.py](../../scripts/run_score_calibration.py) with frozen main config and a new output directory.
- **Execution/data gaps:** no full execution SHA; raw dataset/foundation path and hash are not bound by the frozen config or source manifest.
- **Run provenance:** [run001](runs/run001/provenance.json).

This diagnostic is conditional on observed contexts/anchors; it does not estimate full online return, adaptation, or general harmful-write risk.


## 归属与分析版本导航

主生命周期：[H018](../../research/records/H018.md)。 引用及使用范围：[citation 总索引](../../research/CITATIONS.md)（提及或相关工作不代表本 E 已实现该方法）。

- [analysis001](analyses/analysis001/README.md)
- [analysis002](analyses/analysis002/README.md)
