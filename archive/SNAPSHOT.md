# A/B/C1 snapshot

- Repository: <https://github.com/loeness/online-policy-learning-research>
- Visibility at creation: **private**.
- Snapshot version: **0.1.0-ab-c1**.
- Snapshot date: **2026-10-14**.
- Scope: source code, tests, configurations, protocols, research design notes, data, weights, complete experimental logs, plots, interpretation, references and failure records through C1.
- C2/C3: not executed. Earlier design documents describe future plans only.

The initial Git commit archives the existing workspace. It does not reconstruct the historical commit sequence of earlier experiments; the original cohort folders, protocols, run configuration and provenance records retain that history.

All included research artifacts are ordinary Git blobs, not LFS pointers or links to unavailable local files. The first inventory contained 280,077,831 raw bytes, principally compressible step-level CSVs; the largest file was 3,188,498 bytes. [GitHub's file-size documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) permits these individual files without LFS. Current exact coverage is in [manifest.json](manifest.json).

The archive excludes `.git/`, `.venv/`, Python/test caches, `.archive-verification/`, temporary third-party source-document extractions in `tmp/`, and machine-local credential files. Experiment failures and checkpoints are included. The archive is an audit resource for the researcher; its raw logs are not additional memory available to the deployed agents.

To verify downloaded content:

```text
python scripts/verify_archive.py
```

The manifest intentionally omits itself to avoid recursive hashing. Files are retained without Git newline normalization so original provenance hashes remain meaningful across operating systems. Historical provenance hashes describe their experiment-time source versions; the snapshot manifest describes the files archived now.
