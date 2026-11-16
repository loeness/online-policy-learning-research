"""Fail-closed, fixed-file E164 evidence verifier.

This module reads only explicitly named files beneath caller-supplied roots. It
does not import the scientific package, deserialize model artifacts, or infer
runtime completion from a worker receipt alone.
"""

import hashlib
import json
import math
import os
import pathlib
import re
import stat
import struct


INPUT_FORMAT = "e164_logical_scores_v1"
OUTPUT_FORMAT = "e164_exploratory_report_v1"
SCORE_FORMAT = "h036_score_metrics_v1"
EXPECTED_SEEDS = (
    26201, 26202, 26203, 26204, 26205, 26206, 26207, 26208,
    26209, 26210, 26211, 26212, 26213, 26215, 26216, 26218,
)
ARMS = ("V", "UG", "UR")
ORIGINS = {"REUSED_E163", "NEW_SCORE", "ALIAS"}
SPLITS = ("train", "holdout")
OLD_KL_VIEWS = ("pre_codec_fresh", "encoded_fresh")
G_NEW_VIEWS = ("pre_codec_fresh", "encoded_fresh")
ALL_FRESH_VIEWS = OLD_KL_VIEWS + ("W0_fresh",)
COMPARISONS = (("UG-V", "V", "UG"), ("UR-UG", "UG", "UR"))

# The accepted H036 method source and manifest were independently reviewed by
# the coordinator. Runtime completion still requires caller-supplied source
# binding and external launcher evidence.
FINAL_SOURCE_MANIFEST_ACCEPTED = True

METHOD_SOURCE_IDENTITY_SHA256 = "baae9a6a766518105a6d5b2e9bc64fef3488ff581870931cf83ff2d7b23a4c9a"
METHOD_MANIFEST_SHA256 = "76073211ec4ff37af2677e0d3cae33c87e46a703a25c306ccb0257c03a2ef379"
METHOD_BINDINGS_BYTES = 172892
METHOD_BINDINGS_SHA256 = "b13bfe7c9e3d51e0e82872e7d87eeb7e30cb89e21d55b68129f1fdf46147b860"
METHOD_CONFIG_SHA256 = "c937165e6dc25e1640d8f7b7b80108558b41499f77d42d8501c34caf052cf93d"
METHOD_CONFIG_CANONICAL_SHA256 = "f0d4e2b97dd0d7087d74885c8bab05a571e9ef646f249298547053dce4eba6a9"
METHOD_V1_SOURCE_IDENTITY_SHA256 = "0685e20c095b4c49c12d04041e3857f19b584c062dbe3eea456af1561fa661ea"
FOUNDATION_BYTES = 55477
FOUNDATION_SHA256 = "e6a22024a83aa5f1f8699d44702007f31e0d3132fd1cc6ed3a0f0fdc538c7222"
METHOD_SOURCE_BYTES = 762031
METHOD_FILE_COUNT = 27
ORCHESTRATION_FILE_COUNT = 8
REFERENCE_COUNT = 308
EXPECTED_REMOTE_PROJECT_PREFIX = "/data/robot/p001/"
CPU_LIMIT_SECONDS = 60.0
WALL_LIMIT_SECONDS = 120.0
ARTIFACT_LIMIT_BYTES = 67108864
CPU_STOP_SECONDS = 55.0
WALL_STOP_SECONDS = 110.0
FOOTER_CPU_RESERVE_SECONDS = 5.0
FOOTER_WALL_RESERVE_SECONDS = 10.0
FOOTER_BYTE_RESERVE = 65536
RECOVERY_SEEDS = (26204, 26206, 26209, 26212, 26213)
REUSE_SEEDS = tuple(seed for seed in EXPECTED_SEEDS if seed not in RECOVERY_SEEDS)


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(label + " must be an object")
    return value


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(label + " must be a finite JSON number")
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise ValueError(label + " must be a finite JSON number") from exc
    if not finite:
        raise ValueError(label + " must be a finite JSON number")
    return value


def _validated_values(metrics, seed, expected_lambda, label):
    metrics = _mapping(metrics, label)
    if metrics.get("format") != SCORE_FORMAT:
        raise ValueError(label + " has the wrong metrics format")
    lambda_value = metrics.get("lambda_value")
    if (metrics.get("seed") != seed or isinstance(lambda_value, bool)
            or not isinstance(lambda_value, (int, float)) or lambda_value != expected_lambda):
        raise ValueError(label + " seed/lambda identity mismatch")
    if metrics.get("binding") != "wildcard":
        raise ValueError(label + " binding identity mismatch")
    prepared_id = metrics.get("prepared_id")
    if not isinstance(prepared_id, str) or not prepared_id.strip():
        raise ValueError(label + " prepared_id is missing")

    views = {}
    for view in ALL_FRESH_VIEWS:
        node = _mapping(metrics.get(view), label + "." + view)
        views[view] = {}
        for split in SPLITS:
            for kind in ("old", "new"):
                key = kind + "_" + split
                views[view][key] = _finite(node.get(key), label + "." + view + "." + key)

    aliases = (
        ("K_old", views["encoded_fresh"]["old_holdout"]),
        ("K_new", views["encoded_fresh"]["new_holdout"]),
        ("K_new_W0", views["W0_fresh"]["new_holdout"]),
    )
    for key, nested in aliases:
        if _finite(metrics.get(key), label + "." + key) != nested:
            raise ValueError(label + "." + key + " does not match the nested score")
    holdout_gain = _finite(metrics.get("G_new"), label + ".G_new")
    if holdout_gain != views["W0_fresh"]["new_holdout"] - views["encoded_fresh"]["new_holdout"]:
        raise ValueError(label + ".G_new does not match its legacy definition")
    gains = {
        view: {
            split: views["W0_fresh"]["new_" + split] - views[view]["new_" + split]
            for split in SPLITS
        }
        for view in G_NEW_VIEWS
    }
    for view, split_values in gains.items():
        for split, value in split_values.items():
            _finite(value, label + ".G_new." + view + "." + split)
    if gains["encoded_fresh"]["holdout"] != holdout_gain:
        raise ValueError(label + ".G_new does not match encoded holdout gain")
    old_kl = {
        view: {split: views[view]["old_" + split] for split in SPLITS}
        for view in OLD_KL_VIEWS
    }
    return prepared_id, old_kl, gains, views["W0_fresh"]


def _classify(delta, metric):
    if delta == 0:
        return "unchanged"
    if metric == "old_kl":
        return "improved" if delta < 0 else "degraded"
    return "improved" if delta > 0 else "degraded"


def _delta(left, right, metric):
    change = right - left
    _finite(change, metric + " paired delta")
    return {
        "from_value": left,
        "to_value": right,
        "delta_to_minus_from": change,
        "absolute_delta": abs(change),
        "classification": _classify(change, metric),
    }


def _zero_counts():
    return {"improved": 0, "degraded": 0, "unchanged": 0}


def _independent_recalculation(logical_scores):
    scores = _mapping(logical_scores, "logical_scores")
    if scores.get("format") != INPUT_FORMAT:
        raise ValueError("logical score format is not frozen to the proposed schema")
    if scores.get("source_freeze_state") != "FROZEN":
        raise ValueError("logical score source binding is not frozen")
    freeze_id = scores.get("source_freeze_id")
    if not isinstance(freeze_id, str) or not freeze_id.strip():
        raise ValueError("logical score source_freeze_id is missing")
    prefixes = scores.get("prefixes")
    if not isinstance(prefixes, list) or len(prefixes) != 16:
        raise ValueError("logical score input must contain 16 prefixes")
    seeds = tuple(item.get("seed") if isinstance(item, dict) else None for item in prefixes)
    if seeds != EXPECTED_SEEDS or len(set(seeds)) != 16:
        raise ValueError("logical prefix identities do not match the fixed 16-prefix set")

    parsed = []
    for prefix in prefixes:
        seed = prefix["seed"]
        c = _mapping(prefix.get("C"), "seed " + str(seed) + ".C")
        c_id, c_old, c_gains, c_w0 = _validated_values(c, seed, 0, "seed " + str(seed) + ".C")
        c_ref = _mapping(prefix.get("C_reference"), "seed " + str(seed) + ".C_reference")
        if not c_ref:
            raise ValueError("original C provenance reference is missing")
        arms = {}
        for arm in ARMS:
            wrapper = _mapping(prefix.get(arm), "seed " + str(seed) + "." + arm)
            origin = wrapper.get("origin")
            if not isinstance(origin, str) or origin not in ORIGINS:
                raise ValueError("seed " + str(seed) + "." + arm + " origin is invalid")
            reference = _mapping(wrapper.get("reference"), "seed " + str(seed) + "." + arm + ".reference")
            if not reference:
                raise ValueError("arm provenance reference is missing")
            prepared_id, old_kl, gains, w0 = _validated_values(
                wrapper.get("metrics"), seed, 1, "seed " + str(seed) + "." + arm
            )
            arms[arm] = {
                "origin": origin,
                "reference": reference,
                "prepared_id": prepared_id,
                "old_kl": old_kl,
                "gains": gains,
                "W0_fresh": w0,
            }
        all_values = {"C": {"prepared_id": c_id, "W0_fresh": c_w0}}
        all_values.update({arm: value for arm, value in arms.items()})
        if len({value["prepared_id"] for value in all_values.values()}) != 1:
            raise ValueError("seed " + str(seed) + " C/V/UG/UR prepared_id values differ")
        for arm, value in all_values.items():
            for key in ("old_train", "new_train", "old_holdout", "new_holdout"):
                if value["W0_fresh"][key] != c_w0[key]:
                    raise ValueError("seed " + str(seed) + " W0_fresh baseline differs for " + arm + "." + key)
        parsed.append({
            "seed": seed,
            "prepared_id": c_id,
            "C_reference": c_ref,
            "arms": arms,
        })

    recalculated = []
    for comparison, left_arm, right_arm in COMPARISONS:
        old_counts = {view: {split: _zero_counts() for split in SPLITS} for view in OLD_KL_VIEWS}
        gain_counts = {view: {split: _zero_counts() for split in SPLITS} for view in G_NEW_VIEWS}
        rows = []
        for prefix in parsed:
            left = prefix["arms"][left_arm]
            right = prefix["arms"][right_arm]
            old_changes = {}
            for view in OLD_KL_VIEWS:
                old_changes[view] = {}
                for split in SPLITS:
                    record = _delta(
                        left["old_kl"][view][split],
                        right["old_kl"][view][split],
                        "old_kl",
                    )
                    old_changes[view][split] = record
                    old_counts[view][split][record["classification"]] += 1
            gain_changes = {view: {} for view in G_NEW_VIEWS}
            for view in G_NEW_VIEWS:
                for split in SPLITS:
                    record = _delta(left["gains"][view][split], right["gains"][view][split], "G_new")
                    gain_changes[view][split] = record
                    gain_counts[view][split][record["classification"]] += 1
            rows.append({
                "seed": prefix["seed"],
                "prepared_id": prefix["prepared_id"],
                "sources": {
                    left_arm: {"origin": left["origin"], "reference": left["reference"]},
                    right_arm: {"origin": right["origin"], "reference": right["reference"]},
                    "C_reference": prefix["C_reference"],
                },
                "old_kl_changes": old_changes,
                "G_new_changes": gain_changes,
            })
        recalculated.append({
            "comparison": comparison,
            "from_arm": left_arm,
            "to_arm": right_arm,
            "logical_prefix_count": len(rows),
            "pairing_unit": "logical_prefix",
            "paired_rows": rows,
            "summary": {
                "old_kl": {
                    view: {
                        split: {
                            "direction": "lower_is_better",
                            "logical_prefix_count": len(rows),
                            "counts": old_counts[view][split],
                        }
                        for split in SPLITS
                    }
                    for view in OLD_KL_VIEWS
                },
                "G_new": {
                    view: {
                        split: {
                            "direction": "higher_is_better",
                            "logical_prefix_count": len(rows),
                            "counts": gain_counts[view][split],
                        }
                        for split in SPLITS
                    }
                    for view in G_NEW_VIEWS
                },
            },
        })
    return freeze_id, recalculated


def verify_report_recalculation(logical_scores, report):
    """Recompute pairwise values without importing report.py."""
    freeze_id, expected_comparisons = _independent_recalculation(logical_scores)
    report = _mapping(report, "report")
    if report.get("format") != OUTPUT_FORMAT:
        return False
    if report.get("status") != "EXPLORATORY_ONLY":
        return False
    if report.get("exploratory_only") is not True or report.get("no_advancement_decision") is not True:
        return False
    if report.get("source_contract") != INPUT_FORMAT or report.get("source_freeze_id") != freeze_id:
        return False
    if report.get("logical_prefix_count") != 16 or report.get("pairing_unit") != "logical_prefix":
        return False
    if report.get("classification_rule") != "exact delta zero only; no tolerance":
        return False
    return report.get("comparisons") == expected_comparisons


def _resolve_regular_file(allowed_root, supplied_path, label):
    try:
        root = pathlib.Path(os.fspath(allowed_root)).resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as exc:
        raise ValueError("allowlisted root cannot be resolved: " + str(exc)) from exc
    if not root.is_dir():
        raise ValueError("allowlisted root must resolve to a directory")
    try:
        path = pathlib.Path(os.fspath(supplied_path))
        if not path.is_absolute():
            path = root / path
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as exc:
        raise ValueError(label + " path cannot be resolved: " + str(exc)) from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(label + " resolved path escapes the caller-supplied allowlisted root") from exc
    try:
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise ValueError(label + " path cannot be inspected: " + str(exc)) from exc
    if not stat.S_ISREG(mode):
        raise ValueError(label + " resolved path is not a regular file")
    return root, resolved


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_mirror_sha256(allowed_root, source_path, mirror_path, expected_sha256):
    """Hash only two explicitly named regular files inside an explicit root.

    Resolved source or mirror paths outside the caller-supplied root are
    rejected, including symlinks that point outside it. No directory is scanned.
    """
    result = {
        "source_path": str(source_path),
        "mirror_path": str(mirror_path),
        "expected_sha256": expected_sha256,
        "ok": False,
    }
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_sha256)):
        result["error"] = "expected SHA256 must be 64 lowercase hexadecimal characters"
        return result
    try:
        root, source = _resolve_regular_file(allowed_root, source_path, "source")
        mirror_root, mirror = _resolve_regular_file(allowed_root, mirror_path, "mirror")
        if mirror_root != root:
            result["error"] = "source and mirror did not resolve under the same allowlisted root"
            return result
        result["allowlisted_root"] = str(root)
        result["source_path"] = str(source)
        result["mirror_path"] = str(mirror)
        hashes = [_file_sha256(source), _file_sha256(mirror)]
    except (OSError, ValueError) as exc:
        result["error"] = str(exc)
        return result
    result["source_sha256"] = hashes[0]
    result["mirror_sha256"] = hashes[1]
    result["ok"] = hashes[0] == expected_sha256 and hashes[1] == expected_sha256
    if not result["ok"]:
        result["error"] = "source, mirror, and expected SHA256 do not all match"
    return result


def _canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _strict_json_loads(raw, label):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(label + " contains duplicate JSON key " + repr(key))
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(label + " contains non-finite JSON token " + value)

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise ValueError(label + " is not strict UTF-8 JSON: " + str(exc)) from exc
    return value


def _safe_relative_path(value, label):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(label + " must be a non-empty POSIX relative path")
    path = pathlib.PurePosixPath(value)
    raw_parts = value.split("/")
    if (path.is_absolute() or any(part in ("", ".", "..") for part in raw_parts)
            or ":" in raw_parts[0]):
        raise ValueError(label + " is absolute or contains traversal components")
    if any(char in value for char in "*?[]"):
        raise ValueError(label + " must name one file, not a glob")
    return value


def _read_file_under_root(root, relative_path, label):
    relative_path = _safe_relative_path(relative_path, label)
    resolved_root, resolved = _resolve_regular_file(root, relative_path, label)
    if resolved_root != pathlib.Path(root).resolve(strict=True):
        raise ValueError(label + " did not resolve under the caller's exact root")
    raw = resolved.read_bytes()
    return raw, resolved


def _read_json_under_root(root, relative_path, label):
    raw, resolved = _read_file_under_root(root, relative_path, label)
    value = _strict_json_loads(raw, label)
    return _mapping(value, label), raw, resolved


def _read_json_lines_under_root(root, relative_path, label):
    raw, resolved = _read_file_under_root(root, relative_path, label)
    rows = []
    for index, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise ValueError(label + " contains an empty line at " + str(index))
        value = _strict_json_loads(line, label + " line " + str(index))
        rows.append(_mapping(value, label + " line " + str(index)))
    if not rows:
        raise ValueError(label + " is empty")
    return rows, raw, resolved


def _required(mapping, key, label):
    if key not in mapping:
        raise ValueError(label + " is missing " + key)
    return mapping[key]


def _exact_int(value, label, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(label + " must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(label + " is below its minimum")
    return value


def _exact_number(value, label):
    return _finite(value, label)


def _valid_sha(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(label + " must be a lowercase 64-character SHA-256")
    return value


def _hash_without_field(value, field, label):
    body = dict(_mapping(value, label))
    expected = _valid_sha(_required(body, field, label), label + "." + field)
    body.pop(field)
    actual = _sha256_bytes(_canonical_bytes(body))
    if actual != expected:
        raise ValueError(label + "." + field + " does not match canonical content")
    return expected


def _bound_reference(reference, label):
    reference = _mapping(reference, label)
    # Match runner._reference_key exactly: key, then remote_path, then path.
    path = reference.get("remote_path") or reference.get("path")
    key = reference.get("key") or path
    if not isinstance(key, str) or not key or not isinstance(path, str) or not path:
        raise ValueError(label + " must bind a key and source path")
    size = _exact_int(_required(reference, "bytes", label), label + ".bytes", minimum=0)
    sha = _valid_sha(_required(reference, "sha256", label), label + ".sha256")
    actual_path = reference.get("remote_path", path)
    if not isinstance(actual_path, str) or not actual_path:
        raise ValueError(label + " has no exact source path")
    return {"key": key, "actual_path": actual_path, "bytes": size, "sha256": sha}


def _binding_references(bindings):
    bindings = _mapping(bindings, "bindings")
    if bindings.get("format") != "h036_acceptance_bindings_v1":
        raise ValueError("bindings format differs from the frozen H036 binding")
    if bindings.get("v1_source_identity_sha256") != METHOD_V1_SOURCE_IDENTITY_SHA256:
        raise ValueError("bindings old source identity differs from the accepted H036 package")
    records = _mapping(_required(bindings, "records", "bindings"), "bindings.records")
    if set(records) != {str(seed) for seed in EXPECTED_SEEDS}:
        raise ValueError("bindings do not contain exactly the 16 fixed seeds")

    references = {}

    def add(reference, label):
        item = _bound_reference(reference, label)
        previous = references.get(item["key"])
        if previous is not None and previous != item:
            raise ValueError("binding key is reused with conflicting identity: " + item["key"])
        references[item["key"]] = item

    for field in ("run002_provenance", "run002_mirror_manifest", "v1_manifest"):
        add(_required(bindings, field, "bindings"), "bindings." + field)
    original_sources = _required(bindings, "original_scientific_files", "bindings")
    if not isinstance(original_sources, list) or not original_sources:
        raise ValueError("bindings.original_scientific_files must be a non-empty list")
    for index, reference in enumerate(original_sources):
        add(reference, "bindings.original_scientific_files[" + str(index) + "]")

    add(_required(bindings, "foundation", "bindings"), "bindings.foundation")

    foundation = _bound_reference(
        {**_mapping(_required(bindings, "foundation", "bindings"), "bindings.foundation"),
         "key": "foundation"},
        "bindings.foundation",
    )
    if foundation["bytes"] != FOUNDATION_BYTES or foundation["sha256"] != FOUNDATION_SHA256:
        raise ValueError("foundation identity differs from the frozen binding")

    for seed in EXPECTED_SEEDS:
        record = _mapping(records[str(seed)], "bindings.records[" + str(seed) + "]")
        if record.get("seed") != seed or record.get("binding") != "wildcard":
            raise ValueError("binding record seed or wildcard identity mismatch")
        if record.get("logical_case") != "s" + str(seed) + "_dynamics_late":
            raise ValueError("binding logical case mismatch for seed " + str(seed))
        prepared_id = record.get("prepared_id")
        if not isinstance(prepared_id, str) or not prepared_id:
            raise ValueError("binding prepared_id is missing for seed " + str(seed))
        if record.get("foundation", {}).get("bytes") != FOUNDATION_BYTES:
            raise ValueError("binding foundation bytes mismatch for seed " + str(seed))
        if record.get("foundation", {}).get("sha256") != FOUNDATION_SHA256:
            raise ValueError("binding foundation SHA mismatch for seed " + str(seed))
        add(_required(record, "preparation_reference", "binding record"),
            "binding preparation reference " + str(seed))
        inputs = _mapping(_required(record, "inputs", "binding record"), "binding inputs")
        if set(inputs) != {"checkpoint.pt", "actor.bin", "evidence.npz"}:
            raise ValueError("binding inputs differ from the exact three-file prefix")
        for name, reference in inputs.items():
            add(reference, "binding input " + str(seed) + "/" + name)
        arms = _mapping(_required(record, "arms", "binding record"), "binding arms")
        if set(arms) != {"C", "D"}:
            raise ValueError("binding arms must contain C and D")
        for arm_name, expected_lambda in (("C", 0), ("D", 1)):
            arm = _mapping(arms[arm_name], "binding arm")
            if (arm.get("prepared_id") != prepared_id or arm.get("binding") != "wildcard"
                    or arm.get("lambda_value") != expected_lambda):
                raise ValueError("binding arm identity mismatch for seed " + str(seed) + "/" + arm_name)
            artifacts = _mapping(_required(arm, "artifacts", "binding arm"), "binding artifacts")
            expected_artifacts = {
                "candidate_f32", "candidate_json", "metrics", "diagnostics", "actor",
                "checkpoint", "persistent_references",
            }
            if set(artifacts) != expected_artifacts:
                raise ValueError("binding artifacts are incomplete for seed " + str(seed) + "/" + arm_name)
            for artifact_name, reference in artifacts.items():
                add(reference, "binding artifact " + str(seed) + "/" + arm_name + "/" + artifact_name)
            if arm.get("raw_candidate_sha256") != artifacts["candidate_f32"].get("sha256"):
                raise ValueError("binding raw candidate identity disagrees with candidate.f32")

    return bindings, references, foundation


def _validate_reference_receipt(root, receipt, expected_references, reference_map=None):
    if receipt.get("format") != "e164_reference_receipt_v1":
        raise ValueError("reference receipt format is invalid")
    rows = _required(receipt, "files", "reference_receipt")
    if not isinstance(rows, list):
        raise ValueError("reference_receipt.files must be a list")
    actual = {}
    json_copies = {}
    json_copy_raw = {}
    has_reference_map = reference_map is not None
    reference_map = {} if reference_map is None else reference_map
    for index, row_value in enumerate(rows):
        row = _mapping(row_value, "reference_receipt.files[" + str(index) + "]")
        key = row.get("key")
        if not isinstance(key, str) or not key or key in actual:
            raise ValueError("reference receipt keys must be unique non-empty strings")
        expected = expected_references.get(key)
        if expected is None:
            raise ValueError("reference receipt contains an unbound key: " + key)
        mapped = reference_map.get(key)
        if has_reference_map and not isinstance(mapped, dict):
            raise ValueError("reference receipt key is absent from the frozen reference_map: " + key)
        expected_path = mapped["actual_path"] if isinstance(mapped, dict) else row.get("actual_path")
        _project_path_string(expected_path, "reference receipt actual_path")
        expected_fields = {"key", "actual_path", "bytes", "sha256"}
        expected_suffix = pathlib.PurePosixPath(expected_path).suffix.lower()
        if expected_suffix == ".json":
            expected_fields.add("local_json_copy")
        if set(row) != expected_fields:
            raise ValueError("reference receipt fields differ from the runner schema for key " + key)
        observed = {
            "actual_path": row.get("actual_path"),
            "bytes": _exact_int(_required(row, "bytes", "reference row"), "reference row.bytes", minimum=0),
            "sha256": _valid_sha(_required(row, "sha256", "reference row"), "reference row.sha256"),
        }
        if ((has_reference_map and observed["actual_path"] != expected_path)
                or observed["bytes"] != expected["bytes"]
                or observed["sha256"] != expected["sha256"]):
            raise ValueError("reference receipt differs from binding for key " + key)
        actual[key] = row
        if expected_suffix == ".json":
            local_copy = row.get("local_json_copy")
            local_copy = _safe_relative_path(local_copy, "reference local_json_copy")
            expected_copy = "reference_json/" + expected["sha256"] + ".json"
            if local_copy != expected_copy:
                raise ValueError("reference JSON copy path differs from the runner's fixed SHA name")
            raw, resolved = _read_file_under_root(root, local_copy, "reference JSON copy " + key)
            if len(raw) != expected["bytes"] or _sha256_bytes(raw) != expected["sha256"]:
                raise ValueError("local reference JSON copy bytes/SHA differ for key " + key)
            json_copies[key] = _strict_json_loads(raw, "reference JSON copy " + key)
            json_copy_raw[key] = raw
    if set(actual) != set(expected_references):
        missing = sorted(set(expected_references) - set(actual))
        raise ValueError("reference receipt omits exact bound keys: " + ", ".join(missing[:8]))

    required_json_keys = {
        key for key, reference in expected_references.items()
        if pathlib.PurePosixPath(
            (reference_map.get(key) or actual[key]).get("actual_path", reference["actual_path"])
        ).suffix.lower() == ".json"
    }
    missing_json = sorted(required_json_keys - set(json_copies))
    if missing_json:
        raise ValueError("reference receipt lacks required local JSON evidence copies: "
                         + ", ".join(missing_json[:8]))
    return actual, json_copies, json_copy_raw


def _diagnostics_view(value, label):
    value = _mapping(value, label)
    nested = value.get("diagnostics")
    if nested is not None:
        value = _mapping(nested, label + ".diagnostics")
    return value


def _normalize_training_path(diag, label):
    diag = _diagnostics_view(diag, label)
    initial_metrics = _mapping(_required(diag, "initial_metrics", label), label + ".initial_metrics")
    initial = {
        "index": 0,
        "fullL": _exact_number(_required(diag, "initial_fullL", label), label + ".initial_fullL"),
        "newKL": _exact_number(_required(initial_metrics, "new_train_kl", label), label + ".initial_metrics.new_train_kl"),
        "oldKL": _exact_number(_required(initial_metrics, "old_train_kl", label), label + ".initial_metrics.old_train_kl"),
        "ridge": _exact_number(_required(initial_metrics, "ridge", label), label + ".initial_metrics.ridge"),
        "candidate_hash": None,
    }
    raw_closures = _required(diag, "closure_records", label)
    if not isinstance(raw_closures, list):
        raise ValueError(label + ".closure_records must be a list")
    closures = []
    for expected_index, raw_value in enumerate(raw_closures, start=1):
        raw = _mapping(raw_value, label + ".closure_records")
        index = _exact_int(_required(raw, "index", label), label + ".closure.index")
        if index != expected_index:
            raise ValueError(label + " closure indices are not continuous and ordered")
        item = {"index": index}
        for field in ("fullL", "newKL", "oldKL", "ridge"):
            item[field] = _exact_number(_required(raw, field, label), label + ".closure." + field)
        item["candidate_hash"] = _valid_sha(_required(raw, "candidate_hash", label), label + ".closure.candidate_hash")
        if not isinstance(_required(raw, "accepted", label), bool):
            raise ValueError(label + " closure accepted must be boolean")
        item["accepted"] = raw["accepted"]
        reject_reason = _required(raw, "reject_reason", label)
        if reject_reason is not None and not isinstance(reject_reason, str):
            raise ValueError(label + " closure reject_reason must be a string or null")
        item["reject_reason"] = reject_reason
        closures.append(item)
    if "closure_count" in diag and _exact_int(diag["closure_count"], label + ".closure_count") != len(closures):
        raise ValueError(label + " closure_count differs from closure_records")
    return diag, initial, closures


def _replay_record(diag_value, seed, label):
    diag, initial, closures = _normalize_training_path(diag_value, label)
    f64_hash = _valid_sha(_required(diag, "best_float64_candidate_sha256", label), label + ".best_float64_candidate_sha256")
    f64_full = _exact_number(_required(diag, "best_float64_fullL", label), label + ".best_float64_fullL")
    initial_for_selection = dict(initial)
    explicit_c0 = diag.get("initial_candidate_sha256")
    if explicit_c0 is not None:
        initial_for_selection["candidate_hash"] = _valid_sha(explicit_c0, label + ".initial_candidate_sha256")
    elif closures:
        if any(closures[0][field] != initial[field] for field in ("fullL", "newKL", "oldKL", "ridge")):
            raise ValueError(label + " old diagnostics cannot establish C0 from first closure")
        initial_for_selection["candidate_hash"] = closures[0]["candidate_hash"]
    else:
        initial_for_selection["candidate_hash"] = f64_hash
    if closures:
        if any(closures[0][field] != initial[field] for field in ("fullL", "newKL", "oldKL", "ridge")):
            raise ValueError(label + " first closure is not the unmodified C0 point")
        if closures[0]["candidate_hash"] != initial_for_selection["candidate_hash"]:
            raise ValueError(label + " first closure candidate hash differs from C0")

    f64 = dict(initial_for_selection)
    u64 = dict(initial_for_selection)
    for item in closures:
        if item["newKL"] <= initial["newKL"] + 1e-12 and item["fullL"] <= f64["fullL"]:
            f64 = {key: item[key] for key in ("index", "fullL", "newKL", "oldKL", "ridge", "candidate_hash")}
        if item["fullL"] <= u64["fullL"]:
            u64 = {key: item[key] for key in ("index", "fullL", "newKL", "oldKL", "ridge", "candidate_hash")}
    if f64["fullL"] != f64_full or f64["candidate_hash"] != f64_hash:
        raise ValueError(label + " recorded F64 does not match exact scalar replay")
    cast = {
        "fallback_applied": _required(diag, "fallback_applied", label),
        "selected_float32_candidate_sha256": _valid_sha(_required(diag, "selected_float32_candidate_sha256", label), label + ".selected_float32_candidate_sha256"),
        "candidate_sha256": _valid_sha(_required(diag, "candidate_sha256", label), label + ".candidate_sha256"),
        "prepared_id": _required(diag, "prepared_id", label),
        "seed": _exact_int(_required(diag, "seed", label), label + ".seed"),
    }
    if not isinstance(cast["fallback_applied"], bool) or cast["seed"] != seed:
        raise ValueError(label + " cast identity seed/fallback is invalid")
    if not isinstance(cast["prepared_id"], str) or not cast["prepared_id"]:
        raise ValueError(label + " cast identity prepared_id is invalid")
    if not cast["fallback_applied"] and cast["candidate_sha256"] != cast["selected_float32_candidate_sha256"]:
        raise ValueError(label + " no-fallback candidate identity differs from selected float32")
    path_record = {"initial_train": initial, "closure_records": closures}
    path_sha = _sha256_bytes(_canonical_bytes(path_record))
    same = f64["candidate_hash"] is not None and f64["candidate_hash"] == u64["candidate_hash"]
    replay = {
        "format": "h036_acceptance_replay_v1",
        "F64": f64,
        "U64": u64,
        "same_float64_candidate": same,
        "reuse_eligible": same and not cast["fallback_applied"],
        "recovery_required": not (same and not cast["fallback_applied"]),
        "original_float32_identity": cast,
        "path_sha256": path_sha,
        "closure_count": len(closures),
    }
    replay["seed"] = seed
    return diag, initial, closures, replay


def _compare_replay_diagnostics(expected_value, actual_trace, seed):
    expected_diag, expected_initial, expected_closures, expected_replay = _replay_record(
        expected_value, seed, "original D diagnostics " + str(seed)
    )
    actual_trace = _mapping(actual_trace, "trace " + str(seed))
    if "candidates" in actual_trace:
        raise ValueError("trace.json must omit coefficient arrays")
    actual_diag, actual_initial, actual_closures = _normalize_training_path(
        _required(actual_trace, "diagnostics", "trace"), "trace diagnostics " + str(seed)
    )
    actual_replay = _replay_record(actual_diag, seed, "trace diagnostics " + str(seed))[3]
    if actual_trace.get("format") != "h036_selection_trace_v1" or actual_trace.get("seed") != seed:
        raise ValueError("trace format/seed mismatch for " + str(seed))
    if actual_trace.get("lambda_value") != 1 or actual_trace.get("prepared_id") != expected_replay["original_float32_identity"]["prepared_id"]:
        raise ValueError("trace lambda/prepared identity mismatch for " + str(seed))
    _valid_sha(actual_trace.get("training_hash"), "trace.training_hash")
    if actual_initial != expected_initial or actual_closures != expected_closures:
        raise ValueError("trace C0 or ordered closure records differ from original D for " + str(seed))
    if actual_replay["F64"] != expected_replay["F64"]:
        raise ValueError("trace F64 result differs from original D for " + str(seed))
    if actual_replay["original_float32_identity"] != expected_replay["original_float32_identity"]:
        raise ValueError("trace float32 identity differs from original D for " + str(seed))
    for key in (
        "iterations", "termination_reason", "optimizer_max_evaluations",
        "optimizer_max_iterations", "optimizer_history_size", "optimizer_learning_rate",
        "optimizer_tolerance_grad", "optimizer_tolerance_change",
    ):
        if _required(actual_diag, key, "trace diagnostics") != _required(expected_diag, key, "original D diagnostics"):
            raise ValueError("trace termination/configuration field differs from original D: " + key)
    for key in ("returned_is_C0", "fallback_reason", "selected_float32_new_train_kl", "final_float32_fullL", "final_float32_ridge"):
        if key in expected_diag or key in actual_diag:
            if expected_diag.get(key) != actual_diag.get(key):
                raise ValueError("trace optional cast field differs from original D: " + key)
    for key in ("closure_objective_evaluations", "closure_kl_metric_evaluations"):
        expected_counts = expected_diag.get("computation_counts")
        actual_counts = actual_diag.get("computation_counts")
        if expected_counts is not None or actual_counts is not None:
            if not isinstance(expected_counts, dict) or not isinstance(actual_counts, dict):
                raise ValueError("trace computation counts are not comparable")
            if expected_counts.get(key) != actual_counts.get(key):
                raise ValueError("trace computation count differs from original D: " + key)
    if actual_trace.get("path_sha256") != actual_replay["path_sha256"]:
        raise ValueError("trace path SHA does not match its ordered training path")
    return actual_diag, actual_replay


def _project_path_string(value, label):
    if (not isinstance(value, str) or "\\" in value or not value.startswith("/")):
        raise ValueError(label + " must be an absolute POSIX project path")
    path = pathlib.PurePosixPath(value)
    if (any(part in (".", "..") for part in path.parts)
            or not value.startswith(EXPECTED_REMOTE_PROJECT_PREFIX)):
        raise ValueError(label + " escapes the robot project allocation")
    return str(path)


def _source_file_identity(rows, label):
    if not isinstance(rows, list) or len(rows) != METHOD_FILE_COUNT:
        raise ValueError(label + " must contain the exact 27 frozen method files")
    identities = []
    seen = set()
    total_bytes = 0
    for index, row_value in enumerate(rows):
        row = _mapping(row_value, label + "[" + str(index) + "]")
        path = _safe_relative_path(_required(row, "path", label), label + ".path")
        size = _exact_int(_required(row, "bytes", label), label + ".bytes", minimum=0)
        sha = _valid_sha(_required(row, "sha256", label), label + ".sha256")
        if path in seen:
            raise ValueError(label + " contains a duplicate path")
        seen.add(path)
        total_bytes += size
        identities.append({"path": path, "bytes": size, "sha256": sha})
    identities.sort(key=lambda item: item["path"])
    if total_bytes != METHOD_SOURCE_BYTES:
        raise ValueError(label + " byte total differs from the accepted 762031-byte source")
    return _sha256_bytes(_canonical_bytes(identities)), identities


def _validate_final_source_spec(spec, expected_references):
    spec = _mapping(spec, "final_source_spec")
    if spec.get("format") != "e164_execution_spec_v1" or spec.get("experiment_id") != "E164":
        raise ValueError("final_source_spec is not the frozen E164 execution spec")
    if spec.get("source_identity_sha256") != METHOD_SOURCE_IDENTITY_SHA256:
        raise ValueError("final_source_spec method source identity differs from the accepted snapshot")
    if spec.get("scientific_manifest_sha256") != METHOD_MANIFEST_SHA256:
        raise ValueError("final_source_spec manifest identity differs from the accepted snapshot")
    source_identity, source_rows = _source_file_identity(
        _required(spec, "source_files", "final_source_spec"), "final_source_spec.source_files"
    )
    if source_identity != METHOD_SOURCE_IDENTITY_SHA256:
        raise ValueError("final_source_spec source file list does not reproduce the accepted identity")
    if sum(item["bytes"] for item in source_rows) != METHOD_SOURCE_BYTES:
        raise ValueError("final_source_spec source byte total is invalid")
    orchestration = _required(spec, "orchestration_files", "final_source_spec")
    if not isinstance(orchestration, list) or len(orchestration) != ORCHESTRATION_FILE_COUNT:
        raise ValueError("final_source_spec must bind the exact eight orchestration files")
    orchestration_paths = set()
    for index, value in enumerate(orchestration):
        row = _mapping(value, "final_source_spec.orchestration_files[" + str(index) + "]")
        path = _safe_relative_path(_required(row, "path", "orchestration file"), "orchestration path")
        _exact_int(_required(row, "bytes", "orchestration file"), "orchestration bytes", minimum=0)
        _valid_sha(_required(row, "sha256", "orchestration file"), "orchestration sha256")
        if path in orchestration_paths:
            raise ValueError("final_source_spec has duplicate orchestration paths")
        orchestration_paths.add(path)

    bindings = _mapping(_required(spec, "bindings", "final_source_spec"), "final_source_spec.bindings")
    if (_exact_int(_required(bindings, "bytes", "final_source_spec.bindings"), "bindings.bytes")
            != METHOD_BINDINGS_BYTES
            or _valid_sha(_required(bindings, "sha256", "final_source_spec.bindings"), "bindings.sha256")
            != METHOD_BINDINGS_SHA256):
        raise ValueError("final_source_spec binding identity differs from the accepted 172892-byte binding")
    _project_path_string(_required(bindings, "path", "final_source_spec.bindings"), "bindings.path")
    manifest = _mapping(_required(spec, "scientific_manifest", "final_source_spec"),
                        "final_source_spec.scientific_manifest")
    if (_exact_int(_required(manifest, "bytes", "scientific_manifest"), "scientific_manifest.bytes")
            != 13363
            or _valid_sha(_required(manifest, "sha256", "scientific_manifest"), "scientific_manifest.sha256")
            != METHOD_MANIFEST_SHA256):
        raise ValueError("final_source_spec scientific manifest identity differs from the accepted source")
    _project_path_string(_required(manifest, "path", "scientific_manifest"), "scientific_manifest.path")

    expected_limits = {
        "cpu_seconds": 60, "campaign_wall_seconds": 120,
        "new_artifact_cache_temp_bytes": 67108864,
        "workers": 1, "numeric_threads_per_worker": 1, "gpu_devices": 0,
        "environment_interactions": 0, "candidate_fit_calls": 5,
        "new_complete_scoring_packages": 15, "logical_paired_prefixes": 16,
        "aggregate_remote_project_bytes": 100000000000,
    }
    if spec.get("limits") != expected_limits:
        raise ValueError("final_source_spec limits differ from the frozen E164 single-attempt cap")
    if (spec.get("project_root") != "/data/robot/p001"
            or spec.get("e_root") != "/data/robot/p001/e164"
            or spec.get("run_root") != "/data/robot/p001/e164/run001"
            or spec.get("source_cwd") != "/data/robot/p001/e164/s001"
            or spec.get("orchestration_root") != "/data/robot/p001/e164/o001"
            or tuple(spec.get("seeds", ())) != EXPECTED_SEEDS
            or tuple(spec.get("recovery_seed_ids", ())) != RECOVERY_SEEDS):
        raise ValueError("final_source_spec paths or fixed cohort differ from the accepted E164 scope")
    cost_scope = _mapping(_required(spec, "campaign_cost_scope", "final_source_spec"),
                         "final_source_spec.campaign_cost_scope")
    if (cost_scope.get("origin") != "fresh supervisor process birth ticks"
            or cost_scope.get("cpu") != "lifetime supervisor RUSAGE_SELF plus direct-child wait4 including waited descendants; live owned counters during execution"
            or cost_scope.get("included") != "Python startup; in-campaign prechecks; worker preparation/fits/scoring/logging; stop/reap and reserved footer"
            or cost_scope.get("outside_cost_reporting") != "separate operational elapsed; unmeasured CPU UNKNOWN, never zero or included in measured campaign subtotal"
            or set(cost_scope.get("outside_origin", [])) != {
                "parent launch shell", "SSH transport", "source/input transfer", "capacity/admission operations"}
            or cost_scope.get("safety_stops") != {"cpu_seconds": 55, "wall_seconds": 110}
            or cost_scope.get("finalization_reserve") != {
                "cpu_seconds": 5, "wall_seconds": 10, "byte_reserve": 65536}):
        raise ValueError("final_source_spec campaign metering scope differs from the accepted supervisor scope")

    raw_map = _mapping(_required(spec, "reference_map", "final_source_spec"),
                       "final_source_spec.reference_map")
    if set(raw_map) != set(expected_references):
        raise ValueError("final_source_spec reference_map keys differ from frozen bindings")
    if len(raw_map) != REFERENCE_COUNT:
        raise ValueError("final_source_spec must bind the exact 308 frozen reference keys")
    normalized_map = {}
    for key, expected in expected_references.items():
        row = _mapping(raw_map[key], "final_source_spec.reference_map[" + key + "]")
        if set(row) != {"actual_path", "bytes", "sha256"}:
            raise ValueError("reference_map entry fields differ from runner schema for " + key)
        actual_path = _project_path_string(row["actual_path"], "reference_map.actual_path")
        if (_exact_int(row["bytes"], "reference_map.bytes", minimum=0) != expected["bytes"]
                or _valid_sha(row["sha256"], "reference_map.sha256") != expected["sha256"]):
            raise ValueError("reference_map attempts to replace a frozen reference identity for " + key)
        normalized_map[key] = {"actual_path": actual_path, "bytes": expected["bytes"],
                               "sha256": expected["sha256"]}
    return {
        "source_identity_sha256": source_identity,
        "manifest_sha256": METHOD_MANIFEST_SHA256,
        "source_file_count": len(source_rows),
        "source_bytes": METHOD_SOURCE_BYTES,
        "reference_count": len(normalized_map),
        "reference_map": normalized_map,
        "environment": _mapping(_required(spec, "environment", "final_source_spec"),
                                "final_source_spec.environment"),
        "expected_packages": _mapping(_required(spec, "expected_packages", "final_source_spec"),
                                      "final_source_spec.expected_packages"),
        "expected_python_version": _required(spec, "expected_python_version", "final_source_spec"),
        "python": _project_path_string(_required(spec, "python", "final_source_spec"),
                                       "final_source_spec.python"),
        "campaign_cost_scope": cost_scope,
        "project_bytes_before": spec.get("project_bytes_before"),
    }


def _array_helper_sha256(dtype, shape, raw):
    return _sha256_bytes(dtype.encode("ascii") + b"\0" + repr(tuple(shape)).encode("ascii") + b"\0" + raw)


def _finite_array(raw, fmt, count, label):
    try:
        values = struct.unpack("<" + str(count) + fmt, raw)
    except struct.error as exc:
        raise ValueError(label + " byte length differs from its fixed array shape") from exc
    if any(not math.isfinite(value) for value in values):
        raise ValueError(label + " contains a non-finite coefficient")
    return values


def _trace_digest(trace, array_bytes):
    diagnostics = _mapping(_required(trace, "diagnostics", "trace"), "trace.diagnostics")
    identity = {
        "format": trace.get("format"), "seed": trace.get("seed"),
        "lambda_value": trace.get("lambda_value"), "training_hash": trace.get("training_hash"),
        "path_sha256": trace.get("path_sha256"), "origin_indices": trace.get("origin_indices"),
        "closure_records": diagnostics.get("closure_records"),
        "initial_fullL": diagnostics.get("initial_fullL"),
        "initial_candidate_sha256": diagnostics.get("initial_candidate_sha256"),
        "best_float64_candidate_sha256": diagnostics.get("best_float64_candidate_sha256"),
        "u_best_float64_candidate_sha256": diagnostics.get("u_best_float64_candidate_sha256"),
        "best_float64_fullL": diagnostics.get("best_float64_fullL"),
        "u_best_float64_fullL": diagnostics.get("u_best_float64_fullL"),
        "selected_float32_candidate_sha256": diagnostics.get("selected_float32_candidate_sha256"),
        "candidate_sha256": diagnostics.get("candidate_sha256"),
        "fallback_applied": diagnostics.get("fallback_applied"),
        "fallback_reason": diagnostics.get("fallback_reason"),
        "termination_reason": diagnostics.get("termination_reason"),
        "iterations": diagnostics.get("iterations"),
    }
    digest = hashlib.sha256(b"h036_selection_trace_v1\0")
    digest.update(_canonical_bytes(identity))
    for label in ("C0", "F64", "U64"):
        digest.update(label.encode("ascii") + b"\0")
        digest.update(array_bytes[label])
    return digest.hexdigest()


def _lock_digest(lock, probe_arrays):
    metadata = {key: value for key, value in lock.items() if key not in {"probes", "lock_sha256"}}
    digest = hashlib.sha256(b"h036_acceptance_locked_v1\0")
    digest.update(_canonical_bytes(metadata))
    probes = _mapping(_required(lock, "probes", "lock"), "lock.probes")
    for label in ("V", "UG", "UR"):
        entry = _mapping(_required(probes, label, "lock.probes"), "lock probe")
        entry_metadata = {key: value for key, value in entry.items()
                          if key not in {"candidate", "raw_cast_candidate"}}
        digest.update(label.encode("ascii") + b"\0")
        digest.update(_canonical_bytes(entry_metadata))
        for key, array_name in (("raw_cast_candidate", "raw_cast.f32"),
                                ("candidate", "candidate.f32")):
            digest.update(key.encode("ascii") + b"\0")
            digest.update(probe_arrays[(label, array_name)])
    return digest.hexdigest()


def _read_output_file(root, relative_path, label):
    raw, resolved = _read_file_under_root(root, relative_path, label)
    return raw, {"path": relative_path, "bytes": len(raw), "sha256": _sha256_bytes(raw)}


def _validate_output_descriptor(root, descriptor, expected_path, expected_bytes, label,
                                expected_sha=None):
    descriptor = _mapping(descriptor, label)
    if set(descriptor) != {"path", "bytes", "sha256"}:
        raise ValueError(label + " descriptor fields differ from the runner's fixed file reference")
    path = _safe_relative_path(_required(descriptor, "path", label), label + ".path")
    if path != expected_path:
        raise ValueError(label + " path differs from its fixed package location")
    size = _exact_int(_required(descriptor, "bytes", label), label + ".bytes", minimum=0)
    sha = _valid_sha(_required(descriptor, "sha256", label), label + ".sha256")
    if expected_bytes is not None and size != expected_bytes:
        raise ValueError(label + " descriptor byte count differs from expected bytes")
    if expected_sha is not None and sha != expected_sha:
        raise ValueError(label + " descriptor SHA differs from its frozen source")
    raw, observed = _read_output_file(root, path, label)
    if len(raw) != size or observed["sha256"] != sha:
        raise ValueError(label + " file bytes/SHA differ from its descriptor")
    return raw, observed


def _expect_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(label + " differs from the fixed E164 contract")


def _read_fixed_json(root, relative_path, label):
    value, raw, _resolved = _read_json_under_root(root, relative_path, label)
    return value, raw, {"path": relative_path, "bytes": len(raw), "sha256": _sha256_bytes(raw)}


def _validate_input_path_mapping(bindings, reference_map, expected_references):
    """Bind each file prepare_inputs opens to its exact basename and frozen path."""
    records = _mapping(_required(bindings, "records", "bindings"), "bindings.records")
    required_names = {"actor.bin", "checkpoint.pt", "evidence.npz"}
    validated = {}
    for seed in EXPECTED_SEEDS:
        record = _mapping(records[str(seed)], "bindings.records[" + str(seed) + "]")
        inputs = _mapping(_required(record, "inputs", "binding record"), "binding inputs")
        if set(inputs) != required_names:
            raise ValueError("input descriptors must be exactly actor.bin, checkpoint.pt, evidence.npz")
        paths = {}
        for name, value in inputs.items():
            ref = _bound_reference(value, "input " + str(seed) + "/" + name)
            expected = expected_references.get(ref["key"])
            mapped = reference_map.get(ref["key"])
            if expected is None or not isinstance(mapped, dict):
                raise ValueError("input key is not explicitly bound by final_source_spec: " + ref["key"])
            path = _project_path_string(mapped["actual_path"], "input actual_path")
            if (pathlib.PurePosixPath(path).name != name
                    or ref["bytes"] != expected["bytes"]
                    or ref["sha256"] != expected["sha256"]):
                raise ValueError("input reference does not bind the exact expected file identity: " + name)
            paths[name] = path
        parents = {str(pathlib.PurePosixPath(value).parent) for value in paths.values()}
        if len(parents) != 1:
            raise ValueError("the three input references do not share one exact prefix directory")
        prefix = next(iter(parents))
        for name, path in paths.items():
            if path != prefix + "/" + name:
                raise ValueError("input mapping differs from the filename actually opened by prepare_inputs")
        validated[seed] = {"prefix": prefix, "paths": paths}
    return validated


def _validate_environment(environment, source):
    environment = _mapping(environment, "environment")
    _expect_equal(environment.get("source_identity_sha256"), source["source_identity_sha256"],
                  "environment source identity")
    _expect_equal(environment.get("scientific_manifest_sha256"), source["manifest_sha256"],
                  "environment manifest identity")
    _expect_equal(environment.get("numeric_environment"), source["environment"],
                  "environment numeric environment")
    _expect_equal(environment.get("dependencies"), source["expected_packages"],
                  "environment locked dependencies")
    if not isinstance(environment.get("python"), str) or not environment["python"].startswith(
            source["expected_python_version"] + " "):
        raise ValueError("environment Python version differs from the frozen execution spec")
    _expect_equal(environment.get("executable"), source["python"], "environment executable")
    modules = _mapping(_required(environment, "scientific_modules", "environment"),
                       "environment.scientific_modules")
    expected_modules = {
        "online_rl.retention_acceptance_replay": "retention_acceptance_replay.py",
        "online_rl.retention_acceptance_diagnostic": "retention_acceptance_diagnostic.py",
        "online_rl.retention_protected_inputs": "retention_protected_inputs.py",
        "online_rl.retention_protected_candidate": "retention_protected_candidate.py",
    }
    if set(modules) != set(expected_modules):
        raise ValueError("environment scientific module list differs from the runner's import record")
    source_dir = "/data/robot/p001/e164/s001/src/"
    for name, basename in expected_modules.items():
        path = _project_path_string(modules[name], "environment scientific module path")
        if not path.startswith(source_dir) or pathlib.PurePosixPath(path).name != basename:
            raise ValueError("environment scientific module escaped the frozen E164 source tree")
    if _exact_int(_required(environment, "torch_threads", "environment"),
                  "environment.torch_threads") != 1:
        raise ValueError("environment torch_threads must be exactly one")
    if _exact_int(_required(environment, "torch_interop_threads", "environment"),
                  "environment.torch_interop_threads") != 1:
        raise ValueError("environment torch_interop_threads must be exactly one")
    _expect_equal(environment.get("torch_build"), source["expected_packages"].get("torch"),
                  "environment torch build")
    _expect_equal(environment.get("torch_cuda_build"), None, "environment torch CUDA build")
    _expect_equal(environment.get("requested_device"), "cpu", "environment requested device")
    _expect_equal(_exact_int(_required(environment, "gpu_devices_used", "environment"),
                             "environment.gpu_devices_used"), 0,
                  "environment GPU device count")
    _expect_equal(_exact_int(_required(environment, "environment_interactions", "environment"),
                             "environment.environment_interactions"), 0,
                  "environment interaction count")
    if source["environment"].get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("final source spec does not hide CUDA devices")
    return environment


def _validate_array_artifact(root, descriptor, expected_path, dtype, label):
    descriptor = _mapping(descriptor, label)
    required = {"path", "bytes", "sha256", "dtype", "shape", "order", "scientific_sha256"}
    if set(descriptor) != required:
        raise ValueError(label + " descriptor fields differ from runner array schema")
    if descriptor.get("path") != expected_path:
        raise ValueError(label + " path differs from its fixed array location")
    _expect_equal(descriptor.get("dtype"), dtype, label + " dtype")
    _expect_equal(descriptor.get("shape"), [3, 3], label + " shape")
    _expect_equal(descriptor.get("order"), "C", label + " order")
    size = 72 if dtype == "<f8" else 36
    raw, observed = _read_output_file(root, expected_path, label)
    if len(raw) != size or descriptor.get("bytes") != size or descriptor.get("sha256") != observed["sha256"]:
        raise ValueError(label + " raw file identity differs from its fixed shape")
    values = _finite_array(raw, "d" if dtype == "<f8" else "f", 9, label)
    scientific = _array_helper_sha256(dtype, (3, 3), raw)
    _expect_equal(descriptor.get("scientific_sha256"), scientific, label + " scientific array hash")
    return raw, scientific, values


def _validate_trace_and_lock(root, seed, binding_record, expected_diagnostics,
                             expected_references, input_paths):
    case = "cases/s" + str(seed)
    preparation, _raw, _desc = _read_fixed_json(root, case + "/preparation.json",
                                                 "preparation " + str(seed))
    _expect_equal(_exact_int(_required(preparation, "seed", "preparation"), "preparation.seed"),
                  seed, "preparation seed")
    _expect_equal(preparation.get("prepared_id"), binding_record["prepared_id"],
                  "preparation prepared_id")
    expected_input_identity = {"kind": "three_file_prefix", "prefix_path": input_paths["prefix"],
                               "logical_case": binding_record["logical_case"], "files": {}}
    raw_inputs = binding_record["inputs"]
    for name, value in raw_inputs.items():
        ref = _bound_reference(value, "preparation input " + name)
        expected = expected_references[ref["key"]]
        expected_input_identity["files"][name] = {
            "sha256": expected["sha256"], "bytes": expected["bytes"]}
    _expect_equal(preparation.get("input_identity"), expected_input_identity,
                  "preparation input identity")
    memory = _mapping(_required(preparation, "memory_cpu_wall", "preparation"),
                      "preparation.memory_cpu_wall")
    for field in ("cpu_s", "wall_s"):
        if _exact_number(_required(memory, field, "preparation.memory_cpu_wall"),
                         "preparation.memory_cpu_wall." + field) < 0:
            raise ValueError("preparation timing must be non-negative")
    for field in ("rss_start_bytes", "rss_end_bytes"):
        value = memory.get(field)
        if value is not None:
            _exact_int(value, "preparation.memory_cpu_wall." + field, minimum=0)
    if _exact_int(_required(memory, "torch_threads", "preparation.memory_cpu_wall"),
                  "preparation.memory_cpu_wall.torch_threads") != 1:
        raise ValueError("preparation numeric thread count differs from one")
    for field in ("materialized_array_bytes", "actor_bytes", "checkpoint_serialized_bytes", "evidence_npz_bytes"):
        _exact_int(_required(memory, field, "preparation.memory_cpu_wall"),
                   "preparation.memory_cpu_wall." + field, minimum=0)
    for name in raw_inputs:
        ref = _bound_reference(raw_inputs[name], "preparation input " + name)
        expected = expected_references[ref["key"]]
        if name == "actor.bin" and memory["actor_bytes"] != expected["bytes"]:
            raise ValueError("preparation actor byte count differs from its bound input")
        if name == "checkpoint.pt" and memory["checkpoint_serialized_bytes"] != expected["bytes"]:
            raise ValueError("preparation checkpoint byte count differs from its bound input")
        if name == "evidence.npz" and memory["evidence_npz_bytes"] != expected["bytes"]:
            raise ValueError("preparation evidence byte count differs from its bound input")

    trace, _trace_raw, _trace_file = _read_fixed_json(root, case + "/trace.json", "trace " + str(seed))
    if "candidates" in trace:
        raise ValueError("trace.json must omit candidate arrays")
    _expect_equal(trace.get("format"), "h036_selection_trace_v1", "trace format")
    _expect_equal(_exact_int(_required(trace, "seed", "trace"), "trace.seed"), seed, "trace seed")
    _expect_equal(_required(trace, "lambda_value", "trace"), 1, "trace lambda")
    _expect_equal(trace.get("prepared_id"), binding_record["prepared_id"], "trace prepared_id")
    _valid_sha(_required(trace, "training_hash", "trace"), "trace.training_hash")
    _valid_sha(_required(trace, "path_sha256", "trace"), "trace.path_sha256")
    _valid_sha(_required(trace, "trace_sha256", "trace"), "trace.trace_sha256")
    actual_diag, actual_replay = _compare_replay_diagnostics(expected_diagnostics, trace, seed)

    artifacts, _artifact_raw, _artifact_desc = _read_fixed_json(
        root, case + "/trace_artifacts.json", "trace_artifacts " + str(seed))
    if set(artifacts) != {"C0", "F64", "U64"}:
        raise ValueError("trace_artifacts must contain exactly C0, F64, U64")
    array_bytes = {}
    array_hashes = {}
    for label in ("C0", "F64", "U64"):
        raw, helper_hash, _values = _validate_array_artifact(
            root, artifacts[label], case + "/arrays/" + label + ".f64", "<f8",
            "trace array " + str(seed) + "/" + label)
        array_bytes[label] = raw
        array_hashes[label] = helper_hash
    _expect_equal(array_hashes["F64"], actual_replay["F64"]["candidate_hash"], "F64 array candidate identity")
    _expect_equal(array_hashes["U64"], actual_replay["U64"]["candidate_hash"], "U64 array candidate identity")
    initial_hash = actual_diag.get("initial_candidate_sha256")
    if initial_hash is None:
        closures = actual_diag.get("closure_records", [])
        initial_hash = closures[0].get("candidate_hash") if closures else actual_replay["F64"]["candidate_hash"]
    _expect_equal(array_hashes["C0"], initial_hash, "C0 array candidate identity")
    _expect_equal(_trace_digest(trace, array_bytes), trace["trace_sha256"], "trace digest")

    lock, _lock_raw, _lock_file = _read_fixed_json(root, case + "/lock.json", "lock " + str(seed))
    if "candidates" in lock:
        raise ValueError("lock.json must omit candidate and raw-cast arrays")
    _expect_equal(lock.get("format"), "h036_acceptance_locked_v1", "lock format")
    _expect_equal(lock.get("diagnostic_only"), True, "lock diagnostic-only flag")
    _expect_equal(_exact_int(_required(lock, "seed", "lock"), "lock.seed"), seed, "lock seed")
    _expect_equal(lock.get("prepared_id"), binding_record["prepared_id"], "lock prepared_id")
    _expect_equal(lock.get("path_sha256"), trace.get("path_sha256"), "lock path hash")
    _valid_sha(_required(lock, "lock_sha256", "lock"), "lock.lock_sha256")
    raw_probes = _mapping(_required(lock, "probes", "lock"), "lock.probes")
    if set(raw_probes) != set(ARMS):
        raise ValueError("lock probes must contain exactly V, UG, UR")
    probe_arrays = {}
    for label in ARMS:
        probe = _mapping(raw_probes[label], "lock probe " + label)
        selector = "F64" if label == "V" else "U64"
        _expect_equal(probe.get("selector"), selector, "probe selector " + label)
        for filename, key in (("candidate.f32", "candidate"), ("raw_cast.f32", "raw_cast_candidate")):
            relative = case + "/probes/" + label + "/" + filename
            raw, _descriptor = _read_output_file(
                root, relative, "probe array " + str(seed) + "/" + label + "/" + filename)
            if len(raw) != 36:
                raise ValueError("probe array byte length differs from its fixed 3 by 3 float32 shape")
            _finite_array(raw, "f", 9, "probe array " + str(seed) + "/" + label + "/" + filename)
            helper_hash = _array_helper_sha256("<f4", (3, 3), raw)
            probe_arrays[(label, filename)] = raw
            if key == "candidate":
                _expect_equal(helper_hash, probe.get("candidate_sha256"), "probe candidate helper hash")
                _expect_equal(_sha256_bytes(raw), probe.get("raw_sha256"), "probe candidate raw hash")
            else:
                _expect_equal(helper_hash, probe.get("raw_cast_candidate_sha256"),
                              "probe raw-cast helper hash")
                _expect_equal(_sha256_bytes(raw), probe.get("raw_cast_sha256"),
                              "probe raw-cast raw hash")
    _expect_equal(_lock_digest(lock, probe_arrays), lock["lock_sha256"], "lock digest")
    lock_identity, _identity_raw, _identity_desc = _read_fixed_json(
        root, case + "/lock_identity.json", "lock_identity " + str(seed))
    expected_identity = {
        "seed": lock["seed"], "prepared_id": lock["prepared_id"],
        "path_sha256": lock["path_sha256"], "lock_sha256": lock["lock_sha256"],
        "logical_case": lock.get("logical_case"), "binding": lock.get("binding"),
        "probes": {label: {key: raw_probes[label].get(key) for key in
                            ("source", "candidate_sha256", "raw_sha256")} for label in ARMS},
    }
    _expect_equal(lock_identity, expected_identity, "locked probe identity projection")
    _expect_equal(lock_identity.get("logical_case"), binding_record["logical_case"], "lock logical case")
    _expect_equal(lock_identity.get("binding"), "wildcard", "lock binding")
    return {"trace": trace, "trace_arrays": array_bytes, "trace_array_hashes": array_hashes,
            "trace_replay": actual_replay, "diagnostics": actual_diag, "lock": lock,
            "lock_identity": lock_identity, "probe_arrays": probe_arrays,
            "preparation": preparation}


def _validate_recovery_and_cohort(root, bindings, binding_raw, expected_references,
                                  json_copies, recovered):
    original_records = _mapping(_required(bindings, "records", "bindings"), "bindings.records")
    replay_rows = {}
    recovery_manifest, _manifest_raw, _manifest_desc = _read_fixed_json(
        root, "recovery_manifest.json", "recovery_manifest")
    _expect_equal(recovery_manifest.get("format"), "h036_acceptance_replay_v1",
                  "recovery manifest format")
    _hash_without_field(recovery_manifest, "manifest_sha256", "recovery_manifest")
    _expect_equal(recovery_manifest.get("config_sha256"), METHOD_CONFIG_CANONICAL_SHA256,
                  "recovery manifest config hash")
    _expect_equal(recovery_manifest.get("v1_source_identity_sha256"),
                  METHOD_V1_SOURCE_IDENTITY_SHA256, "recovery manifest v1 source")
    _expect_equal(recovery_manifest.get("cohort_seeds"), list(EXPECTED_SEEDS),
                  "recovery manifest cohort")
    _expect_equal(recovery_manifest.get("recovery_seeds"), list(RECOVERY_SEEDS),
                  "recovery manifest recovery seeds")
    _expect_equal(recovery_manifest.get("reuse_seeds"), list(REUSE_SEEDS),
                  "recovery manifest reuse seeds")
    manifest_replays = _mapping(_required(recovery_manifest, "replays", "recovery_manifest"),
                                "recovery_manifest.replays")
    if set(manifest_replays) != {str(seed) for seed in EXPECTED_SEEDS}:
        raise ValueError("recovery manifest does not contain the exact fixed cohort")
    for seed in EXPECTED_SEEDS:
        original = _mapping(original_records[str(seed)], "binding record " + str(seed))
        d_artifacts = _mapping(original["arms"]["D"]["artifacts"], "original D artifacts")
        diagnostics_ref = _bound_reference(d_artifacts["diagnostics"], "original D diagnostics")
        diagnostics = json_copies.get(diagnostics_ref["key"])
        if diagnostics is None:
            raise ValueError("original D diagnostics JSON copy is missing for seed " + str(seed))
        _diag, _initial, _closures, replay = _replay_record(
            diagnostics, seed, "original D diagnostics " + str(seed))
        replay_rows[str(seed)] = replay
        _expect_equal(manifest_replays[str(seed)], replay,
                      "recovery manifest replay " + str(seed))
        if seed in RECOVERY_SEEDS:
            recovered[seed]["original_diagnostics"] = diagnostics
    cohort, _cohort_raw, cohort_desc = _read_fixed_json(root, "cohort.json", "cohort")
    _expect_equal(cohort.get("format"), "h036_acceptance_cohort_v1", "cohort format")
    _hash_without_field(cohort, "cohort_lock_sha256", "cohort")
    _expect_equal(cohort.get("seeds"), list(EXPECTED_SEEDS), "cohort seeds")
    _expect_equal(cohort.get("recovery_seeds"), list(RECOVERY_SEEDS), "cohort recovery seeds")
    _expect_equal(cohort.get("manifest_sha256"), recovery_manifest["manifest_sha256"],
                  "cohort recovery manifest link")
    _expect_equal(cohort.get("bindings_sha256"), _sha256_bytes(_canonical_bytes(bindings)),
                  "cohort canonical binding hash")
    _expect_equal(cohort.get("v1_source_identity_sha256"), METHOD_V1_SOURCE_IDENTITY_SHA256,
                  "cohort v1 source identity")
    cases = _mapping(_required(cohort, "cases", "cohort"), "cohort.cases")
    if set(cases) != {str(seed) for seed in EXPECTED_SEEDS}:
        raise ValueError("cohort cases do not cover exactly the fixed 16 seeds")
    for seed in EXPECTED_SEEDS:
        binding_record = original_records[str(seed)]
        case = _mapping(cases[str(seed)], "cohort case " + str(seed))
        _expect_equal(_exact_int(_required(case, "seed", "cohort case"), "cohort case seed"),
                      seed, "cohort case seed")
        _expect_equal(case.get("reuse"), seed in REUSE_SEEDS, "cohort reuse classification")
        _expect_equal(case.get("prepared_id"), binding_record.get("prepared_id"),
                      "cohort prepared identity")
        _expect_equal(case.get("logical_case"), binding_record.get("logical_case"),
                      "cohort logical case")
        _expect_equal(case.get("binding"), "wildcard", "cohort binding")
        _expect_equal(case.get("v1_source_identity_sha256"), METHOD_V1_SOURCE_IDENTITY_SHA256,
                      "cohort case v1 identity")
        _expect_equal(case.get("path_sha256"), replay_rows[str(seed)]["path_sha256"],
                      "cohort case replay path")
        _expect_equal(case.get("inputs"), binding_record.get("inputs"), "cohort case input bindings")
        if seed in RECOVERY_SEEDS:
            lock = recovered[seed]["lock"]
            _expect_equal(case.get("lock_sha256"), lock.get("lock_sha256"), "cohort recovered lock")
            _expect_equal(case.get("probes"), {
                label: {
                    **lock["probes"][label].get("source", {}),
                    "candidate_sha256": lock["probes"][label].get("candidate_sha256"),
                    "raw_sha256": lock["probes"][label].get("raw_sha256"),
                } for label in ARMS
            }, "cohort recovered probe projection")
        else:
            _hash_without_field(case, "lock_sha256", "reused cohort case")
            _expect_equal(case.get("arms"), binding_record.get("arms"), "cohort reused arms")
            probes = _mapping(_required(case, "probes", "reused cohort case"), "reused cohort probes")
            if set(probes) != set(ARMS):
                raise ValueError("reused cohort case does not bind all three labels")
            for label in ARMS:
                probe = _mapping(probes[label], "reused cohort probe")
                expected_d = binding_record["arms"]["D"]
                _valid_sha(_required(probe, "candidate_sha256", "reused cohort probe"),
                           "reused cohort candidate hash")
                _valid_sha(_required(probe, "raw_sha256", "reused cohort probe"),
                           "reused cohort raw hash")
                _expect_equal(probe.get("score_ref"), expected_d["artifacts"]["metrics"],
                              "reused cohort metric reference")
        if seed in RECOVERY_SEEDS:
            case_arms = _mapping(_required(case, "arms", "recovered cohort case"),
                                 "recovered cohort arms")
            if set(case_arms) != {"C", "D"}:
                raise ValueError("recovered cohort case does not retain original C/D bindings")
    for field in ("foundation", "run002_provenance", "run002_mirror_manifest", "v1_manifest"):
        _mapping(_required(cohort, field, "cohort"), "cohort." + field)
    if not isinstance(cohort.get("original_scientific_files"), list):
        raise ValueError("cohort original_scientific_files must be an explicit list")
    anchor, _anchor_raw, anchor_desc = _read_fixed_json(root, "anchor_permit.json", "anchor permit")
    _expect_equal(anchor.get("format"), "h036_acceptance_stage_permit_v1", "anchor permit format")
    _expect_equal(anchor.get("stage"), "anchor", "anchor permit stage")
    _expect_equal(anchor.get("cohort"), cohort, "anchor permit cohort")
    _hash_without_field(anchor, "permit_sha256", "anchor permit")
    counter, _counter_raw, _counter_desc = _read_fixed_json(
        root, "counterfactual_permit.json", "counterfactual permit")
    _expect_equal(counter.get("format"), "h036_acceptance_stage_permit_v1",
                  "counterfactual permit format")
    _expect_equal(counter.get("stage"), "counterfactual", "counterfactual permit stage")
    _expect_equal(counter.get("cohort"), cohort, "counterfactual permit cohort")
    _hash_without_field(counter, "permit_sha256", "counterfactual permit")
    token = _mapping(_required(counter, "anchors_pass_token", "counterfactual permit"),
                     "counterfactual anchors pass token")
    _hash_without_field(token, "token_sha256", "anchors pass token")
    _expect_equal(token.get("cohort_lock_sha256"), cohort["cohort_lock_sha256"],
                  "anchors pass token cohort")
    token_records = _mapping(_required(token, "records", "anchors pass token"),
                             "anchors pass token records")
    if set(token_records) != {str(seed) for seed in RECOVERY_SEEDS}:
        raise ValueError("anchor pass token does not cover exactly the five recovery seeds")
    for seed in RECOVERY_SEEDS:
        token_record = _mapping(token_records[str(seed)], "anchor pass token record")
        for field in ("expected_metrics_sha", "actual_metrics_canonical_sha", "actor_sha"):
            _valid_sha(_required(token_record, field, "anchor pass token record"),
                       "anchor pass token " + field)
    return recovery_manifest, cohort, anchor, counter, {
        "replay_rows": replay_rows, "cohort_descriptor": cohort_desc,
        "anchor_descriptor": anchor_desc,
    }


def _validate_new_score_package(root, seed, label, reference, lock, cohort,
                                expected_evidence, expected_prepared_id):
    reference = _mapping(reference, "score reference")
    required = {"package_path", "metrics", "canonical_label", "candidate_sha256",
                "candidate_raw_sha256"}
    if set(reference) != required:
        raise ValueError("new score reference fields differ from runner schema")
    canonical = reference["canonical_label"]
    if canonical not in ARMS:
        raise ValueError("score reference canonical label is invalid")
    directory = "scores/s" + str(seed) + "/" + canonical
    expected_package_path = directory + "/package.json"
    if reference["package_path"] != expected_package_path:
        raise ValueError("score package path differs from its fixed seed/label directory")
    package, _package_raw, package_desc = _read_fixed_json(
        root, expected_package_path, "score package " + str(seed) + "/" + canonical)
    metrics_path = directory + "/metrics.json"
    metrics_raw, metrics_desc = _validate_output_descriptor(
        root, reference["metrics"], metrics_path, None,
        "score metrics " + str(seed) + "/" + canonical)
    metrics = _strict_json_loads(metrics_raw, "score metrics " + str(seed) + "/" + canonical)
    metrics = _mapping(metrics, "score metrics")
    _expect_equal(package.get("format"), "e164_score_package_v1", "score package format")
    _expect_equal(_exact_int(_required(package, "seed", "score package"), "score package seed"),
                  seed, "score package seed")
    _expect_equal(package.get("label"), canonical, "score package label")
    _expect_equal(package.get("prepared_id"), expected_prepared_id, "score package prepared identity")
    _expect_equal(package.get("candidate_sha256"), reference["candidate_sha256"],
                  "score package candidate identity")
    _expect_equal(package.get("candidate_raw_sha256"), reference["candidate_raw_sha256"],
                  "score package raw candidate identity")
    if package.get("diagnostic_only") is not True:
        raise ValueError("score package must be diagnostic-only")
    probe = lock["probes"][canonical]
    _expect_equal(reference["candidate_sha256"], probe.get("candidate_sha256"),
                  "score package locked candidate")
    _expect_equal(reference["candidate_raw_sha256"], probe.get("raw_sha256"),
                  "score package locked raw candidate")
    _expect_equal(metrics.get("candidate_sha256"), probe.get("candidate_sha256"),
                  "score metrics locked candidate")
    _expect_equal(metrics.get("seed"), seed, "score metrics seed")
    _expect_equal(metrics.get("lambda_value"), 1, "score metrics lambda")
    _expect_equal(metrics.get("prepared_id"), expected_prepared_id, "score metrics prepared identity")
    _expect_equal(metrics.get("binding"), "wildcard", "score metrics binding")
    _expect_equal(metrics.get("format"), SCORE_FORMAT, "score metrics format")
    artifacts = _mapping(_required(package, "artifacts", "score package"), "score package artifacts")
    if set(artifacts) != {"actor", "checkpoint", "evidence", "foundation"}:
        raise ValueError("score package does not bind the exact deployment layout")
    local_artifacts = {}
    for key, filename in (("actor", "actor.bin"), ("checkpoint", "checkpoint.pt"),
                          ("evidence", "evidence.npz")):
        descriptor = _mapping(artifacts[key], "score package " + key)
        if set(descriptor) != {"path", "bytes", "sha256"}:
            raise ValueError("score package deployment artifact descriptor has an unexpected schema")
        expected_path = directory + "/" + filename
        if descriptor["path"] != expected_path:
            raise ValueError("score package deployment artifact path is not fixed")
        size = _exact_int(descriptor["bytes"], "score package " + key + ".bytes", minimum=0)
        sha = _valid_sha(descriptor["sha256"], "score package " + key + ".sha256")
        raw, observed = _read_output_file(root, expected_path, "score package " + key)
        if len(raw) != size or observed["sha256"] != sha:
            raise ValueError("score package deployment artifact differs from its descriptor")
        local_artifacts[key] = {"bytes": size, "sha256": sha}
    if local_artifacts["actor"]["bytes"] != 89:
        raise ValueError("score package actor is not exactly 89 bytes")
    _expect_equal(local_artifacts["evidence"], expected_evidence, "score evidence source identity")
    foundation = _mapping(artifacts["foundation"], "score package foundation")
    if set(foundation) != {"bytes", "sha256", "reference_key", "remote_path", "reference_only"}:
        raise ValueError("score package foundation reference schema is invalid")
    _expect_equal(foundation.get("bytes"), FOUNDATION_BYTES, "score package foundation bytes")
    _expect_equal(foundation.get("sha256"), FOUNDATION_SHA256, "score package foundation hash")
    if foundation.get("reference_only") is not True:
        raise ValueError("foundation must remain an opaque reference, not a copied model")
    remote_path = _project_path_string(foundation.get("remote_path"), "score foundation remote path")
    _expect_equal(remote_path, foundation["remote_path"], "score foundation path normalization")
    full_bytes = sum(item["bytes"] for item in local_artifacts.values()) + FOUNDATION_BYTES
    _expect_equal(_exact_int(_required(package, "full_persistent_bytes", "score package"),
                             "score package full_persistent_bytes"), full_bytes,
                  "score package full persistent byte total")
    if full_bytes > 262144:
        raise ValueError("score package exceeds the frozen 256 KiB full persistent cap")
    metric_artifacts = _mapping(_required(metrics, "artifacts", "score metrics"), "score metrics artifacts")
    for key, metric_key in (("actor", "actor_sha256"), ("checkpoint", "checkpoint_sha256"),
                            ("evidence", "evidence_sha256")):
        _expect_equal(metric_artifacts.get(metric_key), local_artifacts[key]["sha256"],
                      "score metrics " + metric_key)
    capacity = _mapping(_required(metrics, "capacity", "score metrics"), "score metrics capacity")
    _expect_equal(capacity.get("actor_bytes"), 89, "score capacity actor bytes")
    _expect_equal(capacity.get("full_persistent_bytes"), full_bytes, "score capacity total bytes")
    _expect_equal(capacity.get("checkpoint_bytes"), local_artifacts["checkpoint"]["bytes"],
                  "score capacity checkpoint bytes")
    _expect_equal(capacity.get("evidence_npz_bytes"), local_artifacts["evidence"]["bytes"],
                  "score capacity evidence bytes")
    _expect_equal(capacity.get("foundation_bytes"), FOUNDATION_BYTES, "score capacity foundation bytes")

    selection, _selection_raw, _selection_desc = _read_fixed_json(
        root, directory + "/selection_source.json", "score selection source")
    expected_selection = {
        "label": canonical, "selector": probe.get("selector"),
        "origin_index": probe.get("origin_index"),
        "candidate_origin_index": probe.get("candidate_origin_index"),
        "post_cast_gate_enabled": probe.get("post_cast_gate_enabled"),
        "post_cast_gate_pass": probe.get("post_cast_gate_pass"),
        "fallback_applied": probe.get("fallback_applied"),
        "candidate_sha256": probe.get("candidate_sha256"),
        "raw_sha256": probe.get("raw_sha256"),
        "trace_sha256": lock.get("trace_sha256"),
        "lock_sha256": lock.get("lock_sha256"),
        "cohort_lock_sha256": cohort.get("cohort_lock_sha256"),
        "stage": "anchor" if canonical == "V" else "counterfactual",
    }
    _expect_equal(selection, expected_selection, "score selection source")
    _expect_equal(package.get("candidate_sha256"), probe.get("candidate_sha256"),
                  "package locked candidate")
    return {"package": package, "package_descriptor": package_desc,
            "metrics": metrics, "metrics_descriptor": metrics_desc,
            "selection_source": selection, "canonical_label": canonical}


def _validate_launcher_evidence(evidence, state, meter):
    if evidence is None:
        return {"status": "PENDING_RUNTIME_EVIDENCE", "parent_pid": None,
                "supervisor_pid": None, "external_operational_elapsed_seconds": None,
                "external_operational_cpu_seconds": "UNKNOWN"}
    evidence = _mapping(evidence, "launcher_evidence")
    allowed = {"root", "launcher", "launch_attempt"}
    if set(evidence) != allowed:
        raise ValueError("launcher_evidence must contain only root, launcher, launch_attempt")
    root_value = evidence["root"]
    if not isinstance(root_value, str) or not pathlib.Path(root_value).is_absolute():
        raise ValueError("launcher_evidence.root must be an explicit absolute directory")
    try:
        external_root = pathlib.Path(root_value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("launcher evidence root cannot be resolved: " + str(exc)) from exc
    if not external_root.is_dir():
        raise ValueError("launcher evidence root must be a directory")

    parsed = {}
    for field, filename in (("launcher", "launcher.json"), ("launch_attempt", "launch_attempt.json")):
        descriptor = _mapping(evidence[field], "launcher_evidence." + field)
        if set(descriptor) != {"path", "bytes", "sha256"}:
            raise ValueError("external launcher evidence descriptor must bind path, bytes, and SHA-256")
        if descriptor["path"] != filename:
            raise ValueError("external launcher evidence path differs from the fixed filename")
        expected_bytes = _exact_int(descriptor["bytes"], "launcher evidence bytes", minimum=0)
        expected_sha = _valid_sha(descriptor["sha256"], "launcher evidence SHA-256")
        external_file_root, path = _resolve_regular_file(external_root, filename,
                                                         "external " + field)
        if external_file_root != external_root:
            raise ValueError("external launcher file escaped its separately supplied root")
        raw = path.read_bytes()
        if len(raw) != expected_bytes or _sha256_bytes(raw) != expected_sha:
            raise ValueError("external launcher evidence differs from its independently supplied identity")
        parsed[field] = _strict_json_loads(raw, "external " + field)
        parsed[field] = _mapping(parsed[field], "external " + field)

    launcher = parsed["launcher"]
    attempt = parsed["launch_attempt"]
    expected_argv = ["sh", "/data/robot/p001/e164/o001/launch_command.sh"]
    _expect_equal(launcher.get("entry_argv"), expected_argv, "launcher entry argv")
    _expect_equal(attempt.get("entry_argv"), expected_argv, "launch attempt entry argv")
    if not isinstance(launcher.get("source"), str) or not launcher["source"].strip():
        raise ValueError("launcher record must state where PID and return evidence came from")
    if not isinstance(attempt.get("directive"), str) or not attempt["directive"].strip():
        raise ValueError("launch attempt must bind the explicit execution directive")
    _exact_int(_required(attempt, "ssh_session_handle", "launch_attempt"),
               "launch_attempt.ssh_session_handle", minimum=1)
    parent_pid = _exact_int(_required(launcher, "parent_pid", "launcher"), "launcher.parent_pid", minimum=2)
    child_pid = _exact_int(_required(launcher, "supervisor_pid", "launcher"),
                           "launcher.supervisor_pid", minimum=2)
    if parent_pid == child_pid:
        raise ValueError("launch parent and supervisor child PIDs must differ")
    _expect_equal(meter[0].get("parent_pid"), parent_pid,
                  "campaign meter launch parent PID")
    _expect_equal(meter[0].get("supervisor_pid"), child_pid,
                  "campaign meter supervisor PID")
    for mapping, field, label in ((launcher, "supervisor_exit_code", "launcher"),
                                  (launcher, "entry_return_code", "launcher"),
                                  (attempt, "supervisor_exit_code", "launch_attempt"),
                                  (attempt, "entry_return_code", "launch_attempt")):
        if _exact_int(_required(mapping, field, label), label + "." + field) != 0:
            raise ValueError(label + " records a nonzero launcher return code")
    raw_terminal = _required(attempt, "raw_terminal_output", "launch_attempt")
    if not isinstance(raw_terminal, str):
        raise ValueError("launch attempt raw_terminal_output must preserve terminal text")

    def marker(pattern, label):
        found = re.findall(pattern, raw_terminal, flags=re.MULTILINE)
        if len(found) != 1:
            raise ValueError("raw terminal output must contain one " + label + " marker")
        return _exact_int(int(found[0]), "raw terminal " + label)

    raw_parent = marker(r"^launch_parent_pid=(\d+)\s*$", "launch_parent_pid")
    raw_child = marker(r"^launch_child_pid=(\d+)\s*$", "launch_child_pid")
    raw_supervisor_rc = marker(r"^supervisor_exit_code=(\d+)\s*$", "supervisor_exit_code")
    raw_entry_rc = marker(r"^(?:[A-Z0-9]+_)?ENTRY_RETURN_CODE=(\d+)\s*$", "ENTRY_RETURN_CODE")
    _expect_equal(raw_parent, parent_pid, "raw terminal launch parent PID")
    _expect_equal(raw_child, child_pid, "raw terminal launch child PID")
    _expect_equal(raw_supervisor_rc, launcher["supervisor_exit_code"], "raw terminal supervisor exit code")
    _expect_equal(raw_entry_rc, launcher["entry_return_code"], "raw terminal entry return code")

    proof = _mapping(_required(meter[0], "startup_proof", "campaign_started"),
                     "campaign_started.startup_proof")
    _expect_equal(proof.get("parent_pid"), parent_pid, "supervisor startup parent PID")
    _expect_equal(proof.get("supervisor_pid"), child_pid, "supervisor startup child PID")
    _expect_equal(proof.get("expected_parent_pid"), parent_pid, "supervisor expected parent PID")
    _expect_equal(proof.get("launch_parent_pid_env"), str(parent_pid), "supervisor launch parent environment")
    _expect_equal(state.get("return_code"), launcher["supervisor_exit_code"], "supervisor return code")
    _expect_equal(state.get("worker_return_code"), launcher["supervisor_exit_code"], "worker return code")
    return {"status": "VERIFIED", "parent_pid": parent_pid, "supervisor_pid": child_pid,
            "external_operational_elapsed_seconds": None,
            "external_operational_cpu_seconds": "UNKNOWN"}


def _validate_runtime_accounting(state, meter, wait4_rows, worker, source):
    state = _mapping(state, "state")
    if state.get("experiment_id") != "E164" or state.get("status") != "COMPLETE":
        raise ValueError("supervisor state does not record a completed E164 campaign")
    if _exact_int(_required(state, "return_code", "state"), "state.return_code") != 0:
        raise ValueError("supervisor return code is nonzero")
    if _exact_int(_required(state, "worker_return_code", "state"), "state.worker_return_code") != 0:
        raise ValueError("supervisor worker return code is nonzero")
    _expect_equal(state.get("campaign_cost_scope"), source["campaign_cost_scope"],
                  "state campaign cost scope")
    _expect_equal(state.get("environment_interactions"), 0, "state environment interaction count")
    _expect_equal(state.get("gpu_devices"), 0, "state GPU device count")
    _expect_equal(state.get("limits"), {
        "cpu_seconds": 60, "campaign_wall_seconds": 120,
        "new_artifact_cache_temp_bytes": 67108864, "workers": 1,
        "numeric_threads_per_worker": 1, "gpu_devices": 0,
        "environment_interactions": 0, "candidate_fit_calls": 5,
        "new_complete_scoring_packages": 15, "logical_paired_prefixes": 16,
        "aggregate_remote_project_bytes": 100000000000,
    }, "state hard limits")
    _expect_equal(state.get("safety_stops"), {"cpu_seconds": CPU_STOP_SECONDS,
                  "wall_seconds": WALL_STOP_SECONDS}, "state safety stops")
    _expect_equal(state.get("finalization_reserve"), {"cpu_seconds": FOOTER_CPU_RESERVE_SECONDS,
                  "wall_seconds": FOOTER_WALL_RESERVE_SECONDS,
                  "byte_reserve": FOOTER_BYTE_RESERVE}, "state finalization reserve")
    if state.get("owned_child_handles_final") != []:
        raise ValueError("supervisor final state retains live owned child handles")
    if _exact_int(_required(state, "completed_direct_children", "state"),
                  "state.completed_direct_children") != 3:
        raise ValueError("supervisor must account for the two prechecks and one worker child")
    if not isinstance(wait4_rows, list) or len(wait4_rows) != 3:
        raise ValueError("wait4 evidence must contain exactly the three owned direct children")
    required_purposes = {"cpu_descendant_accounting_precheck", "owned_process_group_stop_precheck",
                         "scientific_worker"}
    if {row.get("purpose") for row in wait4_rows} != required_purposes:
        raise ValueError("wait4 purposes differ from both prechecks and the scientific worker")
    pids = set()
    row_by_pid = {}
    for index, row_value in enumerate(wait4_rows):
        row = _mapping(row_value, "wait4 row " + str(index))
        pid = _exact_int(_required(row, "pid", "wait4 row"), "wait4.pid", minimum=2)
        if pid in pids:
            raise ValueError("wait4 contains a duplicate owned child PID")
        pids.add(pid)
        row_by_pid[pid] = row
        purpose = row.get("purpose")
        expected_wait = {
            "cpu_descendant_accounting_precheck": (0, 0),
            "owned_process_group_stop_precheck": (-15, 15),
            "scientific_worker": (0, 0),
        }[purpose]
        _expect_equal(_exact_int(_required(row, "exit_code", "wait4 row"), "wait4.exit_code"),
                      expected_wait[0], "wait4 exit code for " + purpose)
        _expect_equal(_exact_int(_required(row, "wait_status", "wait4 row"), "wait4.wait_status"),
                      expected_wait[1], "wait4 raw wait status for " + purpose)
        _expect_equal(_exact_int(_required(row, "process_group", "wait4 row"),
                                 "wait4.process_group", minimum=2), pid, "wait4 process group")
        _exact_int(_required(row, "start_ticks", "wait4 row"), "wait4.start_ticks", minimum=1)
        rusage = _mapping(_required(row, "rusage", "wait4 row"), "wait4.rusage")
        user = _exact_number(_required(rusage, "user_seconds", "wait4.rusage"), "wait4 user CPU")
        system_cpu = _exact_number(_required(rusage, "system_seconds", "wait4.rusage"), "wait4 system CPU")
        if user < 0 or system_cpu < 0:
            raise ValueError("wait4 CPU usage cannot be negative")
        measured = _exact_number(_required(row, "wait4_cpu_seconds_including_waited_descendants",
                                           "wait4 row"), "wait4 CPU total")
        if abs(measured - (user + system_cpu)) > 1e-9:
            raise ValueError("wait4 row total differs from recorded user/system CPU")
        if row["purpose"] == "cpu_descendant_accounting_precheck" and measured < 0.08:
            raise ValueError("CPU descendant accounting precheck did not record its bounded burn")
    state_children = _required(state, "child_wait4_records", "state")
    if not isinstance(state_children, list) or len(state_children) != len(wait4_rows):
        raise ValueError("state wait4 summary differs in row count")
    for summary in state_children:
        summary = _mapping(summary, "state child wait4 summary")
        row = row_by_pid.get(summary.get("pid"))
        if row is None:
            raise ValueError("state wait4 summary refers to an unknown PID")
        for field in ("purpose", "exit_code", "start_ticks",
                      "wait4_cpu_seconds_including_waited_descendants"):
            _expect_equal(summary.get(field), row.get(field), "state wait4 summary " + field)

    snapshot = _mapping(_required(state, "cpu_snapshot_before_final_footer", "state"),
                        "state cpu snapshot")
    own = _exact_number(_required(snapshot, "supervisor_rusage_self_cpu_seconds", "cpu snapshot"),
                        "cpu snapshot supervisor CPU")
    children = _exact_number(_required(snapshot, "reaped_wait4_cpu_seconds", "cpu snapshot"),
                             "cpu snapshot wait4 CPU")
    live = _exact_number(_required(snapshot, "live_owned_process_cpu_seconds_from_proc_stat", "cpu snapshot"),
                         "cpu snapshot live CPU")
    total = _exact_number(_required(snapshot, "total_cpu_seconds", "cpu snapshot"),
                          "cpu snapshot total CPU")
    _expect_equal(snapshot.get("live_owned_processes"), [], "final live owned processes")
    wait4_total = sum(float(row["wait4_cpu_seconds_including_waited_descendants"]) for row in wait4_rows)
    if min(own, children, live, total) < 0 or abs(children - wait4_total) > 0.03:
        raise ValueError("final CPU snapshot differs from owned wait4 accounting")
    if abs(total - (own + children + live)) > 0.02:
        raise ValueError("final supervisor CPU total does not add from its recorded components")
    observed_wall = _exact_number(_required(state, "campaign_wall_seconds_before_final_footer", "state"),
                                  "observed supervisor wall time")
    cpu_upper = _exact_number(_required(state, "bounded_campaign_cpu_seconds_upper", "state"),
                              "bounded supervisor CPU")
    wall_upper = _exact_number(_required(state, "bounded_campaign_wall_seconds_upper", "state"),
                               "bounded supervisor wall")
    _expect_equal(cpu_upper, total + FOOTER_CPU_RESERVE_SECONDS, "bounded CPU with footer reserve")
    _expect_equal(wall_upper, observed_wall + FOOTER_WALL_RESERVE_SECONDS,
                  "bounded wall time with footer reserve")
    if total >= CPU_STOP_SECONDS or observed_wall >= WALL_STOP_SECONDS:
        raise ValueError("observed supervisor cost reached or exceeded the safety stop")
    if cpu_upper > CPU_LIMIT_SECONDS or wall_upper > WALL_LIMIT_SECONDS:
        raise ValueError("supervisor plus finalization reserve exceeds the hard budget")
    _expect_equal(state.get("campaign_start_source"),
                  "new supervisor process start ticks from /proc/self/stat; includes Python interpreter startup; launch shell and SSH parent are recorded separately and excluded from the campaign clock",
                  "campaign start source")
    _expect_equal(state.get("final_footer_tail_measurement"),
                  "not observed after supervisor exit; explicitly bounded by the reported CPU and wall reserves",
                  "final footer measurement disclosure")
    _expect_equal(state.get("finalization_reserve", {}).get("byte_reserve"), 65536,
                  "finalization byte reserve")
    if source["project_bytes_before"] is not None:
        before = _exact_int(source["project_bytes_before"], "run_spec.project_bytes_before", minimum=0)
        _expect_equal(state.get("project_bytes_before_non_e164_baseline"), before,
                      "project size baseline")
        aggregate = _exact_int(_required(state, "aggregate_project_apparent_bytes", "state"),
                               "aggregate project bytes", minimum=0)
        e_bytes = _exact_int(_required(state, "e_root_apparent_bytes", "state"),
                             "E164 apparent bytes", minimum=0)
        _expect_equal(aggregate, before + e_bytes, "aggregate project size arithmetic")
        if aggregate > 100000000000:
            raise ValueError("aggregate project apparent bytes exceed the fixed project cap")
    else:
        raise ValueError("runtime evidence lacks the required pre-run aggregate project byte measurement")
    e_bytes = _exact_int(_required(state, "e_root_apparent_bytes", "state"), "E164 apparent bytes", minimum=0)
    if e_bytes > ARTIFACT_LIMIT_BYTES:
        raise ValueError("E164 outputs exceed the fixed 64 MiB artifact/cache/temp cap")
    if state.get("wait4_audit_error") is not None:
        raise ValueError("supervisor wait4 audit records an error")
    audit = _mapping(_required(state, "wait4_children_audit", "state"), "wait4 children audit")
    for field in ("initial_rusage_children_cpu_seconds", "final_rusage_children_cpu_seconds",
                  "rusage_children_delta_cpu_seconds", "wait4_sum_cpu_seconds", "difference_seconds"):
        _mapping(_required(audit, field, "wait4 children audit"), "wait4 audit." + field)
    if audit["initial_rusage_children_cpu_seconds"].get("user") != 0.0 or audit["initial_rusage_children_cpu_seconds"].get("system") != 0.0:
        raise ValueError("initial RUSAGE_CHILDREN baseline was not zero")
    for component in ("user", "system"):
        delta = _exact_number(audit["rusage_children_delta_cpu_seconds"][component],
                              "RUSAGE_CHILDREN delta")
        wait_sum = _exact_number(audit["wait4_sum_cpu_seconds"][component], "wait4 component sum")
        difference = _exact_number(audit["difference_seconds"][component], "wait4 audit difference")
        if abs(delta - wait_sum) > 0.03 or abs(delta - wait_sum) - difference > 1e-9 or difference > 0.03:
            raise ValueError("wait4 component sum does not reconcile with RUSAGE_CHILDREN")

    if worker.get("format") != "e164_worker_state_v1" or worker.get("status") != "COMPLETE":
        raise ValueError("worker state does not record a complete E164 result")
    expected_counts = {"fit_attempts": 5, "fit_returns": 5, "completed_fits": 5,
                       "logical_prefixes": 16, "anchors_confirmed_count": 5,
                       "environment_interactions": 0}
    for field, expected in expected_counts.items():
        _expect_equal(_exact_int(_required(worker, field, "worker state"), "worker." + field),
                      expected, "worker count " + field)
    for field in ("score_calls", "new_scoring_packages"):
        _exact_int(_required(worker, field, "worker state"), "worker." + field, minimum=5)
        if worker[field] > 15:
            raise ValueError("worker scoring count exceeds the fixed cap")
    for field in ("worker_body_cpu_seconds", "worker_body_wall_seconds"):
        value = _exact_number(_required(worker, field, "worker state"), "worker." + field)
        if value < 0:
            raise ValueError("worker subtotal cannot be negative")
    _expect_equal(worker.get("cost_scope"),
                  "worker body subtotal only; complete lifecycle is supervisor wait4",
                  "worker cost scope")

    if not meter or meter[0].get("kind") != "campaign_started" or meter[-1].get("kind") != "campaign_finalizing":
        raise ValueError("meter lifecycle must begin with campaign_started and end with campaign_finalizing")
    last_monotonic = -math.inf
    last_elapsed = -math.inf
    for index, row_value in enumerate(meter):
        row = _mapping(row_value, "meter row " + str(index))
        mono = _exact_number(_required(row, "monotonic", "meter row"), "meter monotonic")
        elapsed = _exact_number(_required(row, "campaign_elapsed_wall_seconds", "meter row"),
                                "meter elapsed wall")
        if mono < last_monotonic or elapsed < last_elapsed or elapsed < 0:
            raise ValueError("meter monotonic/wall measurements are not ordered")
        last_monotonic, last_elapsed = mono, elapsed
    kinds = [row.get("kind") for row in meter]
    required_kinds = ("precheck_started", "cpu_descendant_precheck_passed",
                      "owned_process_group_stop_precheck_passed", "all_prechecks_passed",
                      "worker_command_started", "worker_lifecycle_closed", "campaign_finalizing")
    for kind in required_kinds:
        if kind not in kinds:
            raise ValueError("meter is missing required lifecycle event " + kind)
    if kinds.count("precheck_started") != 2 or kinds.count("all_prechecks_passed") != 1:
        raise ValueError("meter precheck event count differs from the frozen lifecycle")
    if kinds.index("all_prechecks_passed") > kinds.index("worker_command_started"):
        raise ValueError("worker started before every precheck passed")
    started = meter[0]
    _expect_equal(started.get("campaign_start_source"), "new supervisor /proc/self/stat start ticks",
                  "meter campaign start source")
    proof = _mapping(_required(started, "startup_proof", "campaign_started"), "startup proof")
    _expect_equal(proof.get("captured_before_supervisor_fork"), True, "startup proof fork ordering")
    child_baseline = _mapping(_required(proof, "rusage_children_post_auth_pre_fork", "startup proof"),
                              "startup child baseline")
    if child_baseline.get("user_seconds") != 0.0 or child_baseline.get("system_seconds") != 0.0:
        raise ValueError("startup RUSAGE_CHILDREN baseline must be exactly zero")
    finalizing = meter[-1]
    _expect_equal(finalizing.get("status"), state.get("status"), "meter final status")
    _expect_equal(finalizing.get("return_code"), state.get("return_code"), "meter final return code")
    _expect_equal(finalizing.get("cpu_snapshot"), snapshot, "meter final CPU snapshot")
    _expect_equal(finalizing.get("campaign_wall_seconds"), observed_wall, "meter final wall measurement")
    _expect_equal(finalizing.get("bounded_cpu_seconds_upper"), cpu_upper, "meter bounded CPU")
    _expect_equal(finalizing.get("bounded_wall_seconds_upper"), wall_upper, "meter bounded wall")
    _expect_equal(finalizing.get("finalization_reserve"), state.get("finalization_reserve"),
                  "meter finalization reserve")
    return {"observed_supervisor_cpu_seconds": total,
            "observed_supervisor_wall_seconds": observed_wall,
            "bounded_supervisor_cpu_seconds": cpu_upper,
            "bounded_supervisor_wall_seconds": wall_upper,
            "external_operational_elapsed_seconds": None,
            "external_operational_cpu_seconds": "UNKNOWN"}


def _validate_aliases_and_report(root, bindings, expected_references, json_copies,
                                 source, recovered, cohort, anchor, counter):
    aliases, _aliases_raw, aliases_desc = _read_fixed_json(root, "aliases.json", "aliases")
    _expect_equal(aliases.get("format"), "e164_score_aliases_v1", "aliases format")
    records = _required(aliases, "records", "aliases")
    if not isinstance(records, list) or len(records) != 48:
        raise ValueError("aliases must contain exactly 48 logical prefix records")
    if set(aliases) != {"format", "records"}:
        raise ValueError("aliases top-level fields differ from runner output")
    logical, _logical_raw, logical_desc = _read_fixed_json(root, "logical_scores.json", "logical scores")
    if set(logical) != {"format", "source_freeze_state", "source_freeze_id", "prefixes"}:
        raise ValueError("logical_scores top-level fields differ from runner output")
    _expect_equal(logical.get("format"), INPUT_FORMAT, "logical score format")
    _expect_equal(logical.get("source_freeze_state"), "FROZEN", "logical source freeze state")
    _expect_equal(logical.get("source_freeze_id"), source["source_identity_sha256"],
                  "logical source identity")
    prefixes = _required(logical, "prefixes", "logical scores")
    if not isinstance(prefixes, list) or len(prefixes) != len(EXPECTED_SEEDS):
        raise ValueError("logical scores must contain the exact 16 ordered prefixes")
    original_records = _mapping(_required(bindings, "records", "bindings"), "bindings.records")
    aliases_expected = []
    package_results = {}
    reference_origins = {}
    new_score_order = []
    for index, seed in enumerate(EXPECTED_SEEDS):
        prefix = _mapping(prefixes[index], "logical prefix " + str(seed))
        if set(prefix) != {"seed", "C", "C_reference", "V", "UG", "UR"}:
            raise ValueError("logical prefix fields differ from runner output")
        _expect_equal(_exact_int(_required(prefix, "seed", "logical prefix"), "logical prefix seed"),
                      seed, "logical prefix order")
        raw_record = original_records[str(seed)]
        c_ref_raw = raw_record["arms"]["C"]["artifacts"]["metrics"]
        c_ref = _bound_reference(c_ref_raw, "original C metrics " + str(seed))
        c_metrics = json_copies.get(c_ref["key"])
        if c_metrics is None:
            raise ValueError("original C metrics JSON copy is missing for seed " + str(seed))
        _expect_equal(prefix.get("C"), c_metrics, "logical C metrics " + str(seed))
        _expect_equal(prefix.get("C_reference"), c_ref_raw, "logical C source reference " + str(seed))
        if set(_mapping(prefix["C"], "logical C metrics")) != set(prefix["C"]):
            raise ValueError("logical C metrics must be a JSON object")
        for label in ARMS:
            wrapper = _mapping(prefix[label], "logical " + label + " " + str(seed))
            if set(wrapper) != {"metrics", "origin", "reference"}:
                raise ValueError("logical label wrapper fields differ from runner schema")
            origin = wrapper.get("origin")
            if origin not in ORIGINS:
                raise ValueError("logical label origin is invalid")
            reference = _mapping(wrapper.get("reference"), "logical score reference")
            if seed in REUSE_SEEDS:
                _expect_equal(origin, "REUSED_E163", "reused logical origin")
                d_arm = raw_record["arms"]["D"]
                d_metrics_ref = d_arm["artifacts"]["metrics"]
                d_ref = _bound_reference(d_metrics_ref, "original D metrics " + str(seed))
                d_metrics = json_copies.get(d_ref["key"])
                if d_metrics is None:
                    raise ValueError("original D metrics JSON copy is missing for seed " + str(seed))
                expected_reference = {
                    "cohort_ref": {"path": "cohort.json", "cohort_lock_sha256": cohort["cohort_lock_sha256"]},
                    "metrics_key": d_ref["key"], "artifacts": d_arm["artifacts"],
                    "candidate_sha256": d_arm["helper_candidate_sha256"],
                    "candidate_raw_sha256": d_arm["raw_candidate_sha256"],
                }
                _expect_equal(reference, expected_reference, "reused score reference")
                _expect_equal(wrapper.get("metrics"), d_metrics, "reused logical metrics")
            else:
                if origin not in {"NEW_SCORE", "ALIAS"}:
                    raise ValueError("recovery label has an invalid score origin")
                if set(reference) != {"package_path", "metrics", "canonical_label",
                                     "candidate_sha256", "candidate_raw_sha256"}:
                    raise ValueError("recovery score reference fields differ from runner schema")
                lock = recovered[seed]["lock"]
                probe = lock["probes"][label]
                _expect_equal(reference.get("candidate_sha256"), probe.get("candidate_sha256"),
                              "logical locked candidate " + str(seed) + "/" + label)
                _expect_equal(reference.get("candidate_raw_sha256"), probe.get("raw_sha256"),
                              "logical locked raw candidate " + str(seed) + "/" + label)
                canonical = reference.get("canonical_label")
                if origin == "NEW_SCORE":
                    _expect_equal(canonical, label, "new score canonical label")
                    new_score_order.append((seed, label))
                result_key = (seed, canonical)
                if result_key not in package_results:
                    if origin != "NEW_SCORE":
                        raise ValueError("alias points to a package that was never recorded as NEW_SCORE")
                    evidence_ref = _bound_reference(raw_record["inputs"]["evidence.npz"],
                                                     "score evidence input")
                    expected_evidence = {"bytes": evidence_ref["bytes"], "sha256": evidence_ref["sha256"]}
                    package_results[result_key] = _validate_new_score_package(
                        root, seed, label, reference, lock, cohort, expected_evidence,
                        raw_record["prepared_id"])
                package_result = package_results[result_key]
                _expect_equal(wrapper.get("metrics"), package_result["metrics"],
                              "logical recovery metrics " + str(seed) + "/" + label)
                if origin == "ALIAS":
                    _expect_equal(package_result["canonical_label"], canonical,
                                  "alias canonical package")
            alias_record = {"seed": seed, "label": label, "origin": origin, "reference": reference}
            aliases_expected.append(alias_record)
            reference_origins[(seed, label)] = {"origin": origin, "reference": reference}
    _expect_equal(records, aliases_expected, "aliases rows versus logical score references")
    unique_new = [item for item in aliases_expected if item["origin"] == "NEW_SCORE"]
    unique_paths = {item["reference"]["package_path"] for item in unique_new}
    if len(unique_new) != len(unique_paths) or not (5 <= len(unique_paths) <= 15):
        raise ValueError("new score package count is outside the fixed 5-to-15 range")
    report, _report_raw, report_desc = _read_fixed_json(root, "scientific_report.json", "scientific report")
    if not verify_report_recalculation(logical, report):
        raise ValueError("scientific report differs from independent raw-metric recalculation")
    report_md, _md_desc = _read_output_file(root, "scientific_report.md", "scientific report markdown")
    expected_md = ("# E164 exploratory paired diagnosis\n\n"
                   "16 logical prefixes; UG-V and UR-UG. No advancement decision.\n"
                   "Exact per-prefix values, aliases and count summaries are in scientific_report.json.\n").encode("utf-8")
    _expect_equal(report_md, expected_md, "scientific report markdown")
    return {"logical_scores": logical, "report": report,
            "aliases": aliases, "aliases_descriptor": aliases_desc,
            "logical_descriptor": logical_desc, "report_descriptor": report_desc,
            "package_results": package_results, "reference_origins": reference_origins,
            "new_score_order": new_score_order, "new_package_count": len(unique_paths)}


def _validate_worker_records(root, worker, events, recovered, cohort, anchor, counter,
                             score_validation):
    worker = _mapping(worker, "state_worker")
    if worker.get("format") != "e164_worker_state_v1" or worker.get("status") != "COMPLETE":
        raise ValueError("state_worker does not have the runner's successful terminal state")
    completed_scores = _required(worker, "completed_scores", "state_worker")
    if not isinstance(completed_scores, list):
        raise ValueError("worker completed_scores must be an array")
    order = [(seed, "V") for seed in RECOVERY_SEEDS]
    order.extend((seed, label) for seed in RECOVERY_SEEDS for label in ("UG", "UR")
                 if score_validation["reference_origins"][(seed, label)]["origin"] == "NEW_SCORE")
    expected_completed = []
    for seed, label in order:
        reference = score_validation["reference_origins"][(seed, label)]["reference"]
        canonical = reference["canonical_label"]
        package = score_validation["package_results"][(seed, canonical)]["package"]
        metrics_desc = score_validation["package_results"][(seed, canonical)]["metrics_descriptor"]
        expected_completed.append({
            "seed": seed, "label": label,
            "relative_dir": "scores/s" + str(seed) + "/" + canonical,
            "candidate_raw_sha256": reference["candidate_raw_sha256"],
            "candidate_sha256": reference["candidate_sha256"],
        })
    _expect_equal(completed_scores, expected_completed, "worker completed score packages")
    _expect_equal(worker.get("new_scoring_packages"), score_validation["new_package_count"],
                  "worker new package count")
    _expect_equal(worker.get("score_calls"), score_validation["new_package_count"],
                  "worker actual scoring call count")
    _expect_equal(worker.get("cohort_lock_sha256"), cohort.get("cohort_lock_sha256"),
                  "worker cohort hash")

    expected_events = []
    for call, seed in enumerate(RECOVERY_SEEDS, start=1):
        trace = recovered[seed]["trace"]
        expected_events.extend([
            ("fit_started", {"seed": seed, "call": call}),
            ("fit_returned", {"seed": seed, "trace_sha256": trace["trace_sha256"]}),
            ("trace_matched", {"seed": seed, "path_sha256": trace["path_sha256"]}),
        ])
    cohort_raw, _ = _read_output_file(root, "cohort.json", "cohort event identity")
    expected_events.append(("cohort_locked", {
        "cohort_lock_sha256": cohort["cohort_lock_sha256"], "file_sha256": _sha256_bytes(cohort_raw)}))
    for seed in RECOVERY_SEEDS:
        reference = score_validation["reference_origins"][(seed, "V")]["reference"]
        package_result = score_validation["package_results"][(seed, reference["canonical_label"])]
        expected_events.extend([
            ("anchor_score_started", {"seed": seed, "label": "V", "permit_sha256": anchor["permit_sha256"]}),
            ("score_completed", {
                "seed": seed, "label": "V", "relative_dir": "scores/s" + str(seed) + "/V",
                "candidate_raw_sha256": reference["candidate_raw_sha256"],
                "candidate_sha256": reference["candidate_sha256"],
                "metrics_sha256": package_result["metrics_descriptor"]["sha256"],
                "package_sha256": package_result["package_descriptor"]["sha256"],
            }),
        ])
    expected_events.append(("anchors_confirmed", {
        "count": 5, "permit_sha256": counter["permit_sha256"],
        "cohort_lock_sha256": cohort["cohort_lock_sha256"]}))
    for seed in RECOVERY_SEEDS:
        for label in ("UG", "UR"):
            entry = score_validation["reference_origins"][(seed, label)]
            reference = entry["reference"]
            if entry["origin"] == "ALIAS":
                expected_events.append(("score_alias", {"seed": seed, "label": label, **reference}))
            else:
                package_result = score_validation["package_results"][(seed, reference["canonical_label"])]
                expected_events.extend([
                    ("counterfactual_score_started", {
                        "seed": seed, "label": label, "permit_sha256": counter["permit_sha256"]}),
                    ("score_completed", {
                        "seed": seed, "label": label,
                        "relative_dir": "scores/s" + str(seed) + "/" + label,
                        "candidate_raw_sha256": reference["candidate_raw_sha256"],
                        "candidate_sha256": reference["candidate_sha256"],
                        "metrics_sha256": package_result["metrics_descriptor"]["sha256"],
                        "package_sha256": package_result["package_descriptor"]["sha256"],
                    }),
                ])
    expected_events.append(("report_completed", {
        "logical_prefixes": 16, "new_scoring_packages": score_validation["new_package_count"]}))
    if len(events) != len(expected_events):
        raise ValueError("worker event count differs from the runner's expected successful sequence")
    for index, ((expected_kind, expected_fields), row_value) in enumerate(zip(expected_events, events)):
        row = _mapping(row_value, "worker event " + str(index))
        _expect_equal(row.get("event"), expected_kind, "worker event order")
        _exact_number(_required(row, "worker_body_wall_seconds", "worker event"),
                      "worker event wall time")
        for key, expected in expected_fields.items():
            _expect_equal(row.get(key), expected, "worker event " + expected_kind + "." + key)
    return worker


def verify_results(allowed_root, *, final_source_spec=None, launcher_evidence=None):
    """Verify one fixed E164 output tree without importing scientific code.

    ``allowed_root`` is the sole run-output root. ``final_source_spec`` is the
    separately reviewed, frozen execution-spec object. ``launcher_evidence``
    independently binds launcher.json and launch_attempt.json in an explicit
    external evidence root; the run tree cannot self-assert its parent PID.
    Missing runtime/launcher evidence stays pending and can never become valid.
    """
    result = {
        "format": "e164_verification_result_v1",
        "status": "INCOMPLETE",
        "final_source_manifest_accepted": FINAL_SOURCE_MANIFEST_ACCEPTED,
        "runtime_evidence_status": "PENDING_RUNTIME_EVIDENCE",
        "external_launcher_status": "PENDING_RUNTIME_EVIDENCE",
        "metering_scope": {
            "measured_origin": "fresh supervisor process birth ticks",
            "measured_includes": "Python startup; in-campaign prechecks; worker preparation/fits/scoring/logging; stop/reap and reserved footer",
            "measured_cpu": "RUSAGE_SELF plus direct-child wait4 including waited descendants, with live owned counters during execution",
            "safety_stops": {"cpu_seconds": CPU_STOP_SECONDS, "wall_seconds": WALL_STOP_SECONDS},
            "footer_reserve": {"cpu_seconds": FOOTER_CPU_RESERVE_SECONDS,
                                "wall_seconds": FOOTER_WALL_RESERVE_SECONDS,
                                "byte_reserve": FOOTER_BYTE_RESERVE},
            "outside_scope": ["parent launch shell", "SSH transport", "source/input transfer", "capacity/admission operations"],
            "outside_elapsed": "reported separately when independently measured; this verifier has no elapsed field and does not infer one",
            "outside_cpu_seconds": "UNKNOWN, never reported as zero",
        },
        "checks": {},
        "errors": [],
    }
    if final_source_spec is None:
        result["errors"].append("final_source_spec is required for the frozen E164 path/hash binding")
        result["status"] = "INCOMPLETE"
        return result
    try:
        root = pathlib.Path(os.fspath(allowed_root)).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("allowed_root must resolve to a directory")
        result["allowed_root"] = str(root)
        binding_raw, _binding_resolved = _read_file_under_root(root, "bindings.json", "bindings")
        if len(binding_raw) != METHOD_BINDINGS_BYTES or _sha256_bytes(binding_raw) != METHOD_BINDINGS_SHA256:
            raise ValueError("run bindings.json does not match the accepted 172892-byte source binding")
        bindings = _strict_json_loads(binding_raw, "bindings.json")
        bindings, expected_references, foundation = _binding_references(bindings)
        if foundation["bytes"] != FOUNDATION_BYTES or foundation["sha256"] != FOUNDATION_SHA256:
            raise ValueError("run binding foundation identity differs from the accepted opaque model reference")
        source = _validate_final_source_spec(final_source_spec, expected_references)
        input_paths = _validate_input_path_mapping(bindings, source["reference_map"], expected_references)
        receipt, _receipt_raw, _receipt_desc = _read_fixed_json(root, "reference_receipt.json", "reference receipt")
        reference_rows, json_copies, json_copy_raw = _validate_reference_receipt(
            root, receipt, expected_references, source["reference_map"])
        result["checks"]["source_binding"] = {
            "source_identity_sha256": source["source_identity_sha256"],
            "scientific_manifest_sha256": source["manifest_sha256"],
            "binding_bytes": len(binding_raw), "binding_sha256": _sha256_bytes(binding_raw),
            "reference_count": len(reference_rows),
            "local_json_reference_count": len(json_copies),
        }
        result["checks"]["environment"] = _validate_environment(
            _read_fixed_json(root, "environment.json", "environment")[0], source)

        raw_records = _mapping(_required(bindings, "records", "bindings"), "bindings.records")
        recovered = {}
        for seed in RECOVERY_SEEDS:
            record = dict(_mapping(raw_records[str(seed)], "binding record " + str(seed)))
            record["arms"] = dict(record["arms"])
            record["arms"]["D"] = dict(record["arms"]["D"])
            d_ref = _bound_reference(record["arms"]["D"]["artifacts"]["diagnostics"],
                                     "original D diagnostics " + str(seed))
            diagnostics = json_copies.get(d_ref["key"])
            if diagnostics is None:
                raise ValueError("local original D diagnostics copy is missing for recovery seed " + str(seed))
            recovered[seed] = _validate_trace_and_lock(
                root, seed, record, diagnostics, expected_references, input_paths[seed])
        recovery_manifest, cohort, anchor, counter, _contract_meta = _validate_recovery_and_cohort(
            root, bindings, binding_raw, expected_references, json_copies, recovered)
        result["checks"]["recovery_and_cohort"] = {
            "recovery_manifest_sha256": recovery_manifest["manifest_sha256"],
            "cohort_lock_sha256": cohort["cohort_lock_sha256"],
            "recovery_seeds": list(RECOVERY_SEEDS), "reuse_seeds": list(REUSE_SEEDS),
        }
        aliases, _aliases_raw, _aliases_desc = _read_fixed_json(root, "aliases.json", "aliases")
        logical, _logical_raw, _logical_desc = _read_fixed_json(root, "logical_scores.json", "logical scores")
        report, _report_raw, _report_desc = _read_fixed_json(root, "scientific_report.json", "scientific report")
        # Validate all score references and report values through the fixed runner schema.
        score_validation = _validate_aliases_and_report(
            root, bindings, expected_references, json_copies, source, recovered,
            cohort, anchor, counter)
        worker, _worker_raw, _worker_desc = _read_fixed_json(root, "state_worker.json", "worker state")
        events, _events_raw, _events_resolved = _read_json_lines_under_root(
            root, "events_worker.jsonl", "worker events")
        worker = _validate_worker_records(root, worker, events, recovered, cohort,
                                          anchor, counter, score_validation)
        state, _state_raw, _state_desc = _read_fixed_json(root, "state.json", "supervisor state")
        meter, _meter_raw, _meter_resolved = _read_json_lines_under_root(root, "meter.jsonl", "supervisor meter")
        wait4, _wait4_raw, _wait4_resolved = _read_json_lines_under_root(root, "wait4.jsonl", "supervisor wait4")
        accounting = _validate_runtime_accounting(state, meter, wait4, worker, source)
        launcher = _validate_launcher_evidence(launcher_evidence, state, meter)
        result["checks"]["scientific_report"] = {
            "format": report.get("format"), "logical_prefix_count": 16,
            "pairing_unit": "logical_prefix", "comparison_count": 2,
            "independent_recalculation": "PASS",
        }
        result["checks"]["score_packages"] = {
            "new_package_count": score_validation["new_package_count"],
            "alias_record_count": 48,
        }
        result["checks"]["runtime_accounting"] = accounting
        result["checks"]["launcher"] = launcher
        if launcher["status"] != "VERIFIED":
            result["status"] = "PENDING_RUNTIME_EVIDENCE"
            result["runtime_evidence_status"] = "VERIFIED_SUPERVISOR_PENDING_EXTERNAL_LAUNCHER"
            result["external_launcher_status"] = "PENDING_RUNTIME_EVIDENCE"
        else:
            result["runtime_evidence_status"] = "VERIFIED"
            result["external_launcher_status"] = "VERIFIED"
            result["status"] = "COMPLETE_VALID"
        result["final_source_manifest_accepted"] = True
        return result
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, OverflowError) as exc:
        message = str(exc)
        result["errors"].append(message)
        lowered = message.lower()
        if any(marker in lowered for marker in ("cannot be resolved", "is missing", "must contain",
                                                 "missing ", "cannot be inspected")):
            result["status"] = "INCOMPLETE"
        else:
            result["status"] = "INVALID"
        return result
