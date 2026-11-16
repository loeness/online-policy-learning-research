# E163 repair002 local operational repair

LOCAL_ONLY_NOT_EXECUTABLE

This is a local repair proposal. It is not a run package or execution authorization. Do not transfer or launch these files. No SSH, remote probe, supervisor, runner, fit, model test, or experiment was run for this repair. The coordinator must independently review the frozen files and create the run spec and authorization after the package is stable.

The launcher is LF-normalized for the exact remote sh entry and starts the Python supervisor as a background child, then waits for it while preserving the shell PID. It records the shell PID, child PID, and supervisor exit code. After spec and authorization validation succeeds, the supervisor checks the expected parent PID against its actual parent and captures both process start ticks plus resource usage at POST_AUTHORIZATION_PRE_FORK, before any owned fork. An unauthorized request returns before parent /proc or resource sampling and has no startup_proof field. The supervisor refuses startup if the post-authorization RUSAGE_CHILDREN user or system CPU baseline is nonzero. CPU usage remains cumulative from process birth, and the campaign wall origin remains the supervisor process start ticks. The final audit reports the zero post-authorization pre-fork boundary, final RUSAGE_CHILDREN, its per-component delta, and corresponding wait4 sums.

The operational roots are o002 and run002. The scientific source remains s001 with source identity 0685e20c095b4c49c12d04041e3857f19b584c062dbe3eea456af1561fa661ea. Runner edits are limited to the expected o002/run002 placements; E161 input references remain e161/run001. Gate code and mapping are copied byte-for-byte. The 32 fits, 12 of 16 gate threshold, zero environment interactions, one single-thread CPU worker, 120 CPU-second cap, 180-second wall cap, and 256 MiB artifact/cache/temp cap are unchanged and do not constitute new authorization.

Static verification was limited to Python AST parsing, exact gate-copy hashes, and diff/hash inspection. No module was imported or executed.
