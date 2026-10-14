# Reproducibility

The archive contains both the original experiment records and commands for fresh reruns. Fresh runs should write to a new output directory; several historical runners default to paths inside `outputs/` and would otherwise overwrite archive records. The expected quick software check is 82 tests. Full A/B+ and C1 runs execute many hundreds of thousands to over a million simulated transitions and are not quick checks.

## Environment

The recorded environment is Python 3.12, CPU PyTorch 2.8.0, Stable-Baselines3 2.8.0, Gymnasium 1.2.2, NumPy 2.5.4, Matplotlib 3.11.2, pytest 9.1.1, and psutil 7.2.2. `requirements-lock.txt` says it is resolved from a Windows x64 Python 3.12 environment; it is not a platform-neutral lock. Use it for the recorded Windows environment. On Linux, use the version constraints in `requirements.txt` and resolve a Linux CPU environment; exact dependency resolution can vary by platform.

### PowerShell (recorded Windows setup)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match -r requirements-lock.txt
& .\.venv\Scripts\python.exe -m pytest tests -q
```

### Linux (CPU, version constraints)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match -r requirements.txt
.venv/bin/python -m pytest tests -q
```

The test suite is expected to report `82 passed`. No tests generate experiment results. For provenance-sensitive reruns, verify the shared files before starting:

```powershell
Get-FileHash data\pretrain_source.npz -Algorithm SHA256
Get-FileHash data\foundation.pt -Algorithm SHA256
```

```bash
sha256sum data/pretrain_source.npz data/foundation.pt
```

Expected hashes are in `data/pretrain_source.json` and `data/foundation.json`. Do not regenerate these two inputs in place if the goal is to reproduce the archived cohorts.

## A/B runs

The first archived software check is the 82-test suite. A fresh A/B suite with the checked-in W0 checkpoint can be run as follows.

### PowerShell

```powershell
& .\.venv\Scripts\python.exe scripts/run_experiments.py suite --config configs/selected.json --output outputs/reproduction/ab --seeds 41,53,67 --dataset data/pretrain_source.npz --checkpoint data/foundation.pt
```

### Linux

```bash
.venv/bin/python scripts/run_experiments.py suite --config configs/selected.json --output outputs/reproduction/ab --seeds 41,53,67 --dataset data/pretrain_source.npz --checkpoint data/foundation.pt
```

The full A/B suite trains the A and B arms across three deployment seeds. `prepare` regenerates the dataset and checkpoint, so use it only with new `--dataset` and `--checkpoint` paths when reproducing pretraining. The original source transitions and W0 checkpoint are already archived.

## B+ and compression reruns

These commands preserve the checked-in results by using new roots wherever the runner exposes an output option.

```powershell
& .\.venv\Scripts\python.exe scripts/run_ab_controls.py --config configs/ab_plus.json --output outputs/reproduction/ab_plus
& .\.venv\Scripts\python.exe scripts/run_matched_online_references.py --config configs/ab_plus.json --output outputs/reproduction/ab_plus/matched_online
& .\.venv\Scripts\python.exe scripts/run_compression_attempt.py --config configs/compression.json --output outputs/reproduction/compression_attempt
& .\.venv\Scripts\python.exe scripts/run_compression_alternatives.py --config configs/compression_alternatives.json --output outputs/reproduction/compression_alternatives
```

```bash
.venv/bin/python scripts/run_ab_controls.py --config configs/ab_plus.json --output outputs/reproduction/ab_plus
.venv/bin/python scripts/run_matched_online_references.py --config configs/ab_plus.json --output outputs/reproduction/ab_plus/matched_online
.venv/bin/python scripts/run_compression_attempt.py --config configs/compression.json --output outputs/reproduction/compression_attempt
.venv/bin/python scripts/run_compression_alternatives.py --config configs/compression_alternatives.json --output outputs/reproduction/compression_alternatives
```

`run_compression_repair.py` has no CLI options and reads/writes `outputs/compression_attempt` directly. Run it from an isolated copy or worktree that contains the completed primary screen at that relative path; it writes `outputs/compression_attempt/repair`. It performs fixed 50% repair, not a new proportion search. `report_supplements.py` and `report_compression_alternatives.py --root <output>` generate derived reports from their corresponding saved runs.

## C1: quick gate and formal run

The checked-in C1 archive has one common foundation, a six-context selected precheck, a full 40-entry validation ledger, a frozen 18-run config, and an offline representation diagnostic. The formal cohort is 18 runs and adds 1,142,784 training plus 276,480 fixed-policy evaluation interactions; the selected precheck alone adds 55,296 interactions. The 304,641 validation-ledger total includes pre-formal development and the selected precheck, so do not add it to the selected-precheck count as if the counts were disjoint.

Run the selected six-context precheck in a fresh directory first:

```powershell
& .\.venv\Scripts\python.exe scripts/run_c1.py --mode precheck --config configs/c1.json --output outputs/c1_reproduction
```

```bash
.venv/bin/python scripts/run_c1.py --mode precheck --config configs/c1.json --output outputs/c1_reproduction
```

This creates the configured `precheck_selected/` results and is only the learnability gate; it does **not** create the historical validation ledger or the representation diagnostic. `--mode formal` reads the selected precheck, runs the 18 formal arms, and then reads `validation_ledger.json` to write the summary's validation interaction count. The reporter additionally requires a nonempty validation ledger and `representation_diagnostic.json`. The original full ledger and diagnostic are retained under `outputs/c1/`.

To run the formal cohort while retaining the original development record, copy those two archival static records into the fresh output before formal/reporting. This reuses the archive's validation accounting; it does not claim to regenerate the 40 historical validation entries.

### PowerShell

```powershell
Copy-Item outputs\c1\validation_ledger.json outputs\c1_reproduction\validation_ledger.json
Copy-Item outputs\c1\representation_diagnostic.json outputs\c1_reproduction\representation_diagnostic.json
& .\.venv\Scripts\python.exe scripts/run_c1.py --mode formal --config configs/c1.json --output outputs/c1_reproduction
& .\.venv\Scripts\python.exe scripts/report_c1.py --output outputs/c1_reproduction
```

### Linux

```bash
cp outputs/c1/validation_ledger.json outputs/c1_reproduction/validation_ledger.json
cp outputs/c1/representation_diagnostic.json outputs/c1_reproduction/representation_diagnostic.json
.venv/bin/python scripts/run_c1.py --mode formal --config configs/c1.json --output outputs/c1_reproduction
.venv/bin/python scripts/report_c1.py --output outputs/c1_reproduction
```

`run_formal` computes `budget_preflight.json` from the selected precheck logs if that file is absent. It is possible to rerun all 18 formal runs from the checked-in code and shared W0; a fresh output root is not self-contained for the full historical validation narrative because no checked-in command regenerates the 40-entry ledger or the diagnostic JSON. Keep the copied archival provenance explicit when presenting a fresh formal rerun.

## Archive evidence versus fresh execution

- `outputs/c1/summary.json` records `complete=true` and 18 runs; all original formal phase logs and agent checkpoints remain under `outputs/c1/formal/`.
- The original C1 report is generated from saved results by `scripts/report_c1.py --output outputs/c1`; report generation does not rerun training.
- Original compression alternative outputs are already complete (`complete=true`, no runner errors, 774,144 new interactions). The primary Top-K screen and its fixed 50% repair have separate summaries and reports.
- A/B+ terminal archives do not include the entire historical RNG state for the original training run. Reconstructed deployment bundles restore the stored policy, optimizer, and router state and use the explicitly recorded new RNG arrangement; they are not exact interrupted-run continuations.
- Resource fields called serialized state or bundle bytes exclude process runtime and are distinct from separately sampled RSS. RSS is a sampled process measure, not a continuously measured hard peak.

