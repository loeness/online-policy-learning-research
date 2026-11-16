# E161 analysis001

State: COMPLETED. Scientific outcome: INCONCLUSIVE. Frozen primary estimator/thresholds were not changed after results.

Input: `../../runs/run001/producer` (byte-identical mirror of `/data/robot/p001/e161/run001/producer`), identified by `../../runs/run001/artifact-manifest.json`. Summary SHA256 `8643ab86522a5edc79974f07d1569291131d5a250d5c7f3aa37d15775dc096cd`.

Source: full Git commit `81de4afd2d703d17c3953facb78fd04b19cb64a9` plus `../../preparation/s002.zip` and its manifest/diff. The auditor received a portable path correction; statistical reporter is unchanged from frozen s001. Failed first audit and passing second audit are both embedded in analysis.json. No new environment interactions.

Execution: `cd /data/robot/p001/s002` then `/data/robot/p001/env/bin/python scripts/report_retention_candidate.py --input /data/robot/p001/e161/run001/producer --output /data/robot/p001/e161/run001/analysis001`. Exact supervised argv and CPU cost: supervisor.json. Parent producer and sources: `../../runs/run001/provenance.json`.

Figures: figure.png, figure.pdf and figure.svg were produced by this command from the saved raw step/rollout ledgers and derived analysis.json. Their hashes are bound in the artifact manifest. The plot shows main paired Welch intervals, KL fit diagnostics and measured prefix/branch CPU; that plot CPU excludes audit/report/archive overhead, which is settled in `../../budget.json`.

Independent arithmetic review: review.json. Main report: report.md; full per-seed, future, recovery, fit, codec, eviction, byte and update evidence: analysis.json. The local source report keeps canonical remote paths in its metadata; use those allocated paths or explicitly restore their mapped bytes to reproduce, rather than silently editing raw provenance.
