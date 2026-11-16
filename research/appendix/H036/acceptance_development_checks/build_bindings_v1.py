"""Bind existing opaque artifacts for E164; never deserialize a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "research/appendix/H036/ACCEPTANCE_BINDINGS_V1.json"
BASE = ROOT / "experiments/exp163/runs/run002"
RAW = BASE / "remote/run002"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reference(path: Path, *, key: str | None = None, remote_path: str | None = None) -> dict:
    data = path.read_bytes()
    result = {"key": key or path.relative_to(ROOT).as_posix(),
              "path": path.relative_to(ROOT).as_posix(), "bytes": len(data), "sha256": sha(data)}
    if remote_path:
        result["remote_path"] = remote_path
    return result


def main() -> None:
    if OUT.exists():
        raise RuntimeError("do not overwrite accepted bindings")
    v1 = load(ROOT / "research/appendix/H036/delivery_v1/manifest.json")
    config = load(ROOT / "configs/h036_acceptance_diagnostic_v1.json")
    provenance = load(BASE / "provenance.json")
    mirror = load(BASE / "mirror_manifest.json")
    mirrored = {item["path"]: item for item in mirror["files"]}
    inputs = {(row["seed"], row["file"]): row for row in v1["input_files"]}
    if v1["source_identity_sha256"] != config["v1_source_identity_sha256"] or provenance["scientific_source_identity_sha256"] != v1["source_identity_sha256"]:
        raise RuntimeError("scientific source identity differs")

    def raw_ref(relative: str) -> dict:
        result = reference(RAW / relative, key="e163/run002/" + relative,
                           remote_path=mirrored[relative]["remote_path"])
        if any(result[k] != mirrored[relative][k] for k in ("bytes", "sha256")):
            raise RuntimeError(f"opaque raw mirror differs: {relative}")
        return result

    scientific = []
    for row in v1["snapshot_files"]:
        if not (row["path"].startswith("src/") or row["path"] in {"configs/h036_protected_v1.json", "requirements.txt", "requirements-lock.txt"}):
            continue
        frozen = ROOT / "research/appendix/H036/delivery_v1/source_snapshot" / row["path"]
        for path in (frozen, ROOT / row["path"]):
            if sha(path.read_bytes()) != row["sha256"]:
                raise RuntimeError(f"old scientific file changed: {row['path']}")
        scientific.append({"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]})

    records = {}
    for seed in config["seeds"]:
        logical = f"s{seed}_dynamics_late"
        preparation = load(RAW / f"cases/s{seed}/preparation.json")
        prep_ref = raw_ref(f"cases/s{seed}/preparation.json")
        if preparation["input_identity"]["logical_case"] != logical:
            raise RuntimeError("case identity differs")
        input_refs = {}
        for name in ("checkpoint.pt", "actor.bin", "evidence.npz"):
            historical = inputs[(seed, name)]
            path = ROOT / historical["relative_path"]
            item = reference(path, key=f"e161/run001/producer/cases/{logical}/prefix/{name}",
                             remote_path=preparation["input_identity"]["prefix_path"] + "/" + name)
            if any(item[k] != historical[k] or item[k] != preparation["input_identity"]["files"][name][k] for k in ("bytes", "sha256")):
                raise RuntimeError("prefix identity differs")
            input_refs[name] = item
        foundation = preparation["foundation"]
        if any(foundation[k] != v1["foundation"][k] for k in ("bytes", "sha256")):
            raise RuntimeError("foundation identity differs")
        arms = {}
        for arm, lam in (("C", 0), ("D", 1)):
            prefix = f"cases/s{seed}/{arm.lower()}"
            metrics = load(RAW / prefix / "metrics.json")
            diagnostics = load(RAW / prefix / "diagnostics.json")
            candidate = load(RAW / prefix / "candidate.json")
            persistent = load(RAW / prefix / "persistent_references.json")
            artifacts = {key: raw_ref(prefix + "/" + filename) for key, filename in (
                ("candidate_f32", "candidate.f32"), ("candidate_json", "candidate.json"),
                ("metrics", "metrics.json"), ("diagnostics", "diagnostics.json"),
                ("actor", "actor.bin"), ("checkpoint", "checkpoint.pt"),
                ("persistent_references", "persistent_references.json"))}
            # Scientific hash is formed from metadata + opaque bytes; no float unpack.
            raw = (RAW / prefix / "candidate.f32").read_bytes()
            helper = sha(b"<f4\0(3, 3)\0" + raw)
            if len(raw) != 36 or (candidate["dtype_str"], candidate["shape"], candidate["order"], candidate["byteorder"]) != ("<f4", [3, 3], "C", "little"):
                raise RuntimeError("candidate layout differs")
            if candidate["raw_file_sha256"] != sha(raw) or any(value != helper for value in (candidate["scientific_candidate_sha256"], diagnostics["candidate_sha256"], metrics["candidate_sha256"])):
                raise RuntimeError("candidate helper/raw chain differs")
            if any(item["prepared_id"] != preparation["prepared_id"] for item in (candidate, metrics, diagnostics)):
                raise RuntimeError("prepared identity differs")
            if candidate["lambda_value"] != lam or metrics["lambda_value"] != lam or metrics["binding"] != "wildcard" or metrics["seed"] != seed:
                raise RuntimeError("arm identity differs")
            if metrics["artifacts"]["actor_sha256"] != artifacts["actor"]["sha256"] or metrics["artifacts"]["checkpoint_sha256"] != artifacts["checkpoint"]["sha256"] or metrics["artifacts"]["evidence_sha256"] != input_refs["evidence.npz"]["sha256"]:
                raise RuntimeError("deployment chain differs")
            if not persistent["deployment_checkpoint_saved_verbatim"] or any(persistent["foundation"][k] != foundation[k] or persistent["evidence"][k] != input_refs["evidence.npz"][k] for k in ("bytes", "sha256")):
                raise RuntimeError("persistent references differ")
            if artifacts["actor"]["bytes"] != 89 or metrics["capacity"]["full_persistent_bytes"] > 262144:
                raise RuntimeError("capacity differs")
            arms[arm] = {
                "lambda_value": lam, "prepared_id": preparation["prepared_id"], "binding": "wildcard",
                "helper_candidate_sha256": helper, "raw_candidate_sha256": sha(raw),
                "selected32_helper_sha256": diagnostics["selected_float32_candidate_sha256"],
                "best64_helper_sha256": diagnostics["best_float64_candidate_sha256"],
                "fallback_applied": diagnostics["fallback_applied"],
                "metrics_format": metrics["format"], "artifacts": artifacts,
                "persistent_capacity": metrics["capacity"],
            }
        records[str(seed)] = {"seed": seed, "logical_case": logical,
                              "prepared_id": preparation["prepared_id"], "binding": "wildcard",
                              "preparation_reference": prep_ref, "inputs": input_refs,
                              "foundation": foundation, "arms": arms}
    value = {
        "format": "h036_acceptance_bindings_v1", "date": "2026-10-23 Asia/Singapore",
        "scope": "Opaque identity binding only; no binary model decoding, fitting or new policy scoring",
        "v1_source_identity_sha256": v1["source_identity_sha256"],
        "run002_provenance": reference(BASE / "provenance.json"),
        "run002_mirror_manifest": reference(BASE / "mirror_manifest.json"),
        "v1_manifest": reference(ROOT / "research/appendix/H036/delivery_v1/manifest.json"),
        "foundation": {key: provenance["foundation_reference"][key] for key in ("bytes", "sha256", "remote_path")} if "foundation_reference" in provenance else records[str(config["seeds"][0])]["foundation"],
        "original_scientific_files": scientific, "records": records,
        "source_verification": "all original science dependency working/frozen bytes matched v1; all referenced mirror files matched opaque mirror SHA/size",
        "placement": "path/remote_path are provenance; exact key/SHA/bytes bind files and allow validator-controlled relocation",
    }
    OUT.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "arms": 2*len(records), "original_scientific_files": len(scientific),
                      "output_bytes": OUT.stat().st_size, "output_sha256": sha(OUT.read_bytes())}))


if __name__ == "__main__":
    main()
