# C1 supplemental archive

- Date: 2026-10-15.
- Repository: https://github.com/loeness/online-policy-learning-research (private).
- Branch: `codex/c1-strong-baselines`.
- Original snapshot: `v0.1.0-ab-c1`, unchanged.
- Scope: C1 model controls, grouped equivalence, individual tuning,
  full actor/critic controls, EWC, bounded behavioral replay, compact
  representations, stronger full conditioning, raw candidates, results,
  checkpoints, plots, code, cost/accounting and negative findings.
- C2/C3: not executed.

Read [the three-part report](../outputs/c1_supplement/结果解读.md),
[the alternative report](../outputs/c1_supplement/alternative_results.md),
and [the research conclusions](../docs/C1_SUPPLEMENT_CONCLUSIONS.md).
The updated [manifest](manifest.json) covers current branch artifacts;
the original tag preserves the earlier manifest.

Historical source modules and the tensor launch document are retained in
`archive/source_versions/`. Their hashes match recorded core-run provenance.
Current code fixes a nested optimizer hook that could double-count calls;
the independent checkpoint audit corrects derived reports without changing
original JSON/checkpoints or policy updates. The frozen launch document and
its post-run wording clarification are distinguished explicitly.

All research artifacts are ordinary Git blobs, including simulated raw data
and checkpoints. Temporary private manuscript extractions, environments,
credentials and verification checkouts remain excluded. The research archive
is not extra memory available to deployed agents. Verify it with
`python scripts/verify_archive.py` after downloading.
