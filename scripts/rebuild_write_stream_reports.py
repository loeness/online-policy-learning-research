"""Build current audits in a new folder, retaining earlier derived reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_write_stream import analyze_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/write_current_reports")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Preserve existing audit batch: {output}")
    output.mkdir(parents=True)
    results = []
    for cohort in sorted(args.root.resolve().glob("write_*")):
        source = cohort / "summary.json"
        if not source.is_file() or not (cohort / "frozen_config.json").is_file():
            continue
        summary_bytes = source.read_bytes()
        summary = json.loads(summary_bytes)
        if summary.get("phase") not in ("known", "unknown") or not summary.get("records"):
            continue
        analysis = analyze_cohort(cohort, output / cohort.name)
        results.append({"cohort": cohort.name, "source_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
                        "records": len(summary["records"]), "audit": analysis["audit"]})
        print(cohort.name, analysis["audit"]["passed"], flush=True)
    (output / "index.json").write_text(json.dumps({
        "purpose": "Current derived audit; earlier reports and raw artifacts retained unchanged.",
        "results": results,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
