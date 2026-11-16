# E164 a002 runtime and binding mapping review

This is an offline auditor correction only. The frozen verifier and campaign files remain unchanged. The previous audit.json was preserved byte-for-byte as audit_intermediate001.json with status INVALID; the coordinator will re-audit the same mirror after accepting this patch.

The a002 reference enumerator now includes the separately validated foundation descriptor. The accepted spec and worker receipt describe 308 references; the earlier a002 copy counted 307.

The wait4 contract is purpose-specific and exact. The supervisor's stop-precheck sends SIGTERM and treats -signal.SIGTERM as the successful expected result. record_wait4 retains both the OS wait status and its derived exit code. The mirrored rows are (0, 0) for cpu_descendant_accounting_precheck, (-15, 15) for owned_process_group_stop_precheck, and (0, 0) for scientific_worker. All three rows must still be present exactly once and retain the existing PID/process-group, start-tick, rusage, CPU sum, and state-summary checks.

The supervisor state.json schema has no cohort_lock_sha256 field. The runner's state_worker.json does, and the existing worker-record check compares that value to cohort.json. The auditor no longer compares the worker hash to a nonexistent supervisor field; it does not supply or infer a cohort identity.

The external launcher check remains fail-closed: it still requires separately supplied fixed-name descriptors with independently supplied byte counts and SHA-256 values, checks the terminal markers for parent PID, supervisor PID, supervisor return code, and entry return code, and binds the launcher PIDs to campaign_started.startup_proof. This patch also binds them to the campaign_started outer parent_pid and supervisor_pid fields, which the supervisor emits. Final state and meter return-code/status consistency remains enforced.

All CPU totals, component reconciliation tolerances, stop thresholds, 60 CPU-second / 120 wall-second hard caps, 55 / 110 stop limits, 5 CPU-second / 10 wall-second / 65,536-byte footer reserve, event ordering, and per-purpose CPU requirements are unchanged. No verifier execution, model loading, scientific import, fit, score, remote action, or raw output edit was performed here.

Current correction artifacts:
- Frozen verifier SHA-256: 0e174f3b57d2dbf894bc8e7f4dc36b55aeb336f730998439b64ccca2a014ac2f
- Prior a002 verifier SHA-256: 9bada3fea0768b670bdcdad7a41d9b139dbd58c5fb64ea36beb3953b05e7b011
- Corrected a002 verifier SHA-256: b1622e02e340350dcb9591ff56079b01bfb466be952feba0001852c58693891b
- Preserved intermediate INVALID audit SHA-256: 0c20996f061ec9eedad0f7e7732ade9e4cc3c7bc74a195d7ca0322ccb1dc52cb
