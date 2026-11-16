# E163 operational plan — not executed

This plan leaves scientific calls unresolved until the frozen delivery arrives.
It is not an executable launch script. The coordinator budget is the source of
execution status; preparation does not activate its reservation.

## Before any transfer or campaign

1. Accept the scientific freeze, independently review it, and map every required
   source/configuration to a relative path, size and SHA-256 in an explicit
   transfer manifest. Bind dirty-source files and the Git parent separately.
   Include no instruction files, credentials, Git history, general outputs or
   unrelated inputs. Never infer a transfer set from a broad directory copy.
2. Resolve the allocated destination and every input reference. Reject symlinks,
   absolute archive members, `..`, duplicate members and any target outside the
   allocated project. Existing E161 inputs are read-only exact-manifest entries;
   do not enumerate their parent directories for discovery.
3. Check only aggregate project bytes and allocated-destination filesystem free
   bytes/inodes and applicable quota. Estimate peak source archive + extraction,
   outputs, logs, caches and temp, including retained evidence. Require both the
   100,000,000,000-byte project cap and 268,435,456-byte E163 new-write cap.
   Admission is fresh; historical disk measurements are not admission.
4. If transfer is ready and authorized, use byte-only archive extraction and
   hash verification. Do not apply archive permissions, chmod, copy2 or timestamps
   on the known mount. Preserve the exact archive/member manifests. No dependency
   installation or system operation is planned.
5. Source the project entry helper, then apply the one-thread/cache/temp overrides
   in `execution_plan.json`; leave HOME unchanged. Capture the actual interpreter,
   required dependency versions, module paths/hashes, argv/cwd and environment
   overrides. Check the 48 remote input hashes against their exact references.

These preparation operations must not deserialize real prefixes or start any
candidate fit. Record their operational cost separately; they are not silently
charged as model CPU or counted as environment interactions.

## One campaign after readiness

Freeze a Linux supervisor and its command before launch. It must account for the
actual compute interpreter and every descendant it starts, including imports,
startup, failure, validation, fitting, statistics and reporting. The scientific
command uses one single-thread CPU worker and no GPU. Do not assume the PID of a
launcher is the interpreter, or assume an orphan's usage is included.

Use kernel child-exit usage (`wait4`/equivalent) for completed, waited processes,
and record supervisor usage and any still-live owned descendants during limit
enforcement without double counting. Record owned handles/start identities;
inspect no unrelated process arguments or jobs. The final command-lifetime
accounting must include descendants through exit and preserve raw counters.
Final accounting/footer overhead needs an explicit reserve inside the limits.
If available counters cannot provide and enforce this coverage, fail closed
before the real fits; do not report a partial counter as complete CPU.

The first zero-model accounting precheck begins the campaign clock. In that
same campaign, a deliberately launched, waited child/grandchild must demonstrate
that the accounting covers descendant work and lifecycle rather than only a
launcher. Also verify the single-thread configuration and stopping path without
models. This check is **not yet run**. It must pass before any real-prefix fit.
Its cost and any failures consume the same reservation; clocks never reset.

Run only the frozen sequence: all input validity checks, 16 prefixes times the
two fixed lambda values, frozen encoded scoring, scalar report and required
provenance. The runner composes the scientific interfaces; it does not implement
teachers, losses, candidate selection, fallback or encoding itself.

Monitor aggregate CPU, campaign wall and E163 new bytes throughout. Use a
conservative stop margin sufficient for termination and evidence finalization
within 120 CPU seconds, 180 wall seconds and the artifact cap. Signal only
recorded project-owned processes. On accounting, identity, correctness,
nonfinite-value or capacity failure, stop before subsequent fits and preserve
partial outputs. On budget exhaustion, stop and mark the batch incomplete unless
the entire fixed batch/report is valid and complete. No automatic real-input
retry or replacement seed is allowed.

At completion retain raw fit/closure records, original versus encoded metrics,
fallback/termination details, actor and complete learner bytes, resident drift,
transient-memory peaks, source/input/candidate identities and full cost records.
The gate uses only the frozen encoded fresh-local-zero holdout fields. Return
evidence to the coordinator for interpretation; no later phase starts here.
