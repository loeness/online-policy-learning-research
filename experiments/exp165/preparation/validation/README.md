# E165 execution preparation

Current state: **STATIC PREPARATION; no remote preparation or runtime admission**.
The experiment and phase limits are owned by the coordinator's
[contract](../../../../research/appendix/H036/COORDINATOR_E165_CONTRACT_20261024.md)
and [ledger](../../budget.json). This directory does not grant resources.

The scientific owner supplies the frozen C32-to-U32 S0/S1 search implementation,
tests and precise endpoint references. The validation runner will bind that
actual API; it will not introduce search, fallback or scoring rules. The design
uses all sixteen prefixes, zero fits/recoveries, at most 336 complete real-data
training-oracle evaluations and 33 scoring packages including the fixed D
anchor. The independent encoded-C0 comparison is excluded.

All runtime validation occurs in the allocated robot project environment:
CHECKS in `c001`, DIAGNOSTIC in `run001`, and AUDIT in `a001`, each with a fresh
supervisor and an exact phase authorization. Phase limits and commands are in
[phase_plan.json](phase_plan.json). Per-stage finalization reserves are included
in the caps. Prior-stage evidence and accumulated cost remain bound; unused
sub-limits cannot silently move between phases.

Local work is limited to editing and static source/manifest inspection. Runtime
checks, synthetic checks, diagnostics, statistical recomputation and independent
audits have not run. Scientific and orchestration source, allowed inputs,
interpreter/dependencies, output schemas and transfer identities must be frozen
before preparation admission. Each phase has one attempt; failures are retained
and returned to the coordinator. No earlier experiment allowance is reused.
