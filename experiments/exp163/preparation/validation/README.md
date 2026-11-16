# E163 validation preparation — 2026-10-22

Status: **PREPARATION_ONLY_NOT_EXECUTABLE**. This is validation-owned preparation,
not acceptance of an unfinished scientific delivery or permission to execute it.
The coordinator ledger remains `execution_authorized=false`, with no active job.
No real-prefix deserialization, fit, SSH operation or campaign metering precheck
was performed for this preparation.

## Prepared evidence

- `input_manifest_planned.json`: 48 exact E161 prefix file references across the
  16 fixed seeds, totaling 2,861,895 bytes. Local opaque-byte identities match
  the preserved E161 artifact manifest. Remote identities must be checked afresh;
  the references must also match the frozen H036 input contract before use.
- `gate.py`: provisional scalar-only report logic for the five encoded,
  fresh-local-zero holdout KL values. It does not compute KL or define fitting,
  installation, compensation, codec or teacher semantics.
- `gate_checks.json`: all nine synthetic arithmetic/batch checks passed. These
  checks used no real prefix or model, and are not an E163 scientific result.
- `execution_plan.json`: fixed resource limits, neutral remote paths and cache
  overrides; scientific argv, module hashes and adapter mapping intentionally
  remain unset until frozen handoff.
- `review_checklist.json`: independent acceptance and operational readiness
  requirements. Pending items must be resolved, not inferred by the runner.
- `metering_and_transfer_plan.md`: planned admission, transfer, accounting and
  stopping procedure. No zero-model metering precheck has run.

## Scalar gate

For each of the fixed 16 seeds, let `G_new,a = K_new_W0 - K_new_a`.
The seed passes exactly when `K_old_C > 1e-10`, `G_new_C > 1e-10`,
`K_old_D <= 0.75*K_old_C`, and `G_new_D >= 0.5*G_new_C`.
At least 12 passes yield `MECHANISM_SCREEN_PASSED_EXPLORATORY`; a complete,
valid batch otherwise yields `NO_ADVANCEMENT`. Invalid/incomplete input suppresses
whole-batch scoring. Legitimate fallbacks remain in the denominator and are
annotated. No reward benefit or optimization-gradient claim follows from this
arithmetic screen.

The adapter schema is provisional. Its final mapping must point to the frozen
scientific interface's actual encoded holdout outputs and preserve detailed
termination, returned-is-C0 and fallback information outside the scalar gate.

## Next required handoff

Wait for METHOD_V1, HANDOFF_V1, frozen delivery manifest and coordinator
acceptance. Independently review the frozen files against the checklist,
finalize exact commands and hashes, and wait for the coordinator's formal
execution-state update. Only then may the single metered campaign begin.
The zero-model accounting check consumes the same 120 CPU / 180 wall seconds
as the subsequent 32 fits and reporting. Failure stops the campaign with its
partial evidence; it does not permit an automatic retry.

This directory does not modify the coordinator contract, budget, shared H/index,
algorithm implementation, or historical E161/E162 evidence.
