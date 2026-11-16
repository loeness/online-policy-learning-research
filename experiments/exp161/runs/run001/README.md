# E161 run001

Producer COMPLETED: 16 shared prefixes, 288 branches, 2244608 interactions. Final audit PASSED; scientific outcome INCONCLUSIVE.

`producer/` is a complete verified mirror; remote canonical path is `/data/robot/p001/e161/run001/producer`. `raw.zip` and artifact-manifest.json preserve all 3691 returned artifacts. `mirror.json` records local placement. The archive SHA256 is `85774a7a9bb083638686a42636c210d2a89a215421728ebba7f705d1d08b78d3`.

`state.json` intentionally retains the first supervisor AUDIT_FAILED state: the producer succeeded, but the original auditor used Windows separators under Linux. `retention_candidate_audit/main.json` is that original failed audit. `main002.json` is the corrected zero-error audit. The corrected analysis supervisor state is `../../analyses/analysis001/control002/state.json`; no RL run was repeated. Exact source/config/data identities, commands and measured costs are in provenance.json; the authoritative budget is ../../budget.json.
