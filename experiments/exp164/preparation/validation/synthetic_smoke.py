"""Small artificial-JSON engineering checks; no scientific modules or inputs."""
from pathlib import Path
import copy
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("e164_engineering_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    sys.dont_write_bytecode = True
    report_module, verifier = load("report"), load("verify_results")
    seeds = tuple(range(26201, 26214)) + (26215, 26216, 26218)
    recovery = {26204, 26206, 26209, 26212, 26213}
    baseline = {"old_train": 2.0, "old_holdout": 2.5, "new_train": 4.0, "new_holdout": 5.0}

    def metric(seed, label):
        change = 0.0 if seed not in recovery or label in {"C", "V"} else (-0.2 if label == "UG" else -0.1)
        gain_change = 0.0 if seed not in recovery or label in {"C", "V"} else -0.2
        pre = {"old_train": 1.2 + change, "old_holdout": 1.5 + change,
               "new_train": 2.0 + gain_change, "new_holdout": 2.5 + gain_change}
        encoded = {"old_train": 1.25 + change, "old_holdout": 1.55 + change,
                   "new_train": 2.1 + gain_change, "new_holdout": 2.6 + gain_change}
        return {"format": "h036_score_metrics_v1", "seed": seed, "lambda_value": 0 if label == "C" else 1,
                "prepared_id": f"artificial-json-{seed}", "binding": "wildcard",
                "pre_codec_fresh": pre, "encoded_fresh": encoded, "W0_fresh": dict(baseline),
                "K_old": encoded["old_holdout"], "K_new": encoded["new_holdout"],
                "K_new_W0": baseline["new_holdout"], "G_new": baseline["new_holdout"] - encoded["new_holdout"]}

    payload = {"format": "e164_logical_scores_v1", "source_freeze_state": "FROZEN",
               "source_freeze_id": "ARTIFICIAL_JSON_SCHEMA_FIXTURE_ONLY", "prefixes": []}
    for seed in seeds:
        row = {"seed": seed, "C": metric(seed, "C"), "C_reference": {"synthetic_engineering": True}}
        for label in ("V", "UG", "UR"):
            row[label] = {"metrics": metric(seed, label), "origin": "NEW_SCORE" if seed in recovery else "REUSED_E163",
                          "reference": {"synthetic_engineering": True, "label": label}}
        payload["prefixes"].append(row)
    checks = []
    result = report_module.build_report(payload)
    counts = result["comparisons"][0]["summary"]["old_kl"]["encoded_fresh"]["holdout"]["counts"]
    assert counts == {"improved": 5, "degraded": 0, "unchanged": 11}
    assert result["comparisons"][1]["summary"]["G_new"]["pre_codec_fresh"]["holdout"]["counts"] == {
        "improved": 0, "degraded": 0, "unchanged": 16}
    assert result["logical_prefix_count"] == 16 and result["no_advancement_decision"] is True
    assert verifier.verify_report_recalculation(payload, result) is True
    checks.append("known paired deltas, 16-prefix unit and independent recomputation")
    changed_report = copy.deepcopy(result)
    changed_report["comparisons"][0]["summary"]["old_kl"]["encoded_fresh"]["holdout"]["counts"]["improved"] = 6
    assert verifier.verify_report_recalculation(payload, changed_report) is False
    checks.append("tampered summary rejected by independent recomputation")
    conflict = copy.deepcopy(payload)
    changed = conflict["prefixes"][0]["UG"]["metrics"]
    changed["W0_fresh"]["new_holdout"] += 1
    changed["K_new_W0"] += 1
    changed["G_new"] += 1
    missing = copy.deepcopy(payload)
    del missing["prefixes"][0]["UR"]["metrics"]["G_new"]
    for name, bad in (("same-prefix W0 conflict", conflict), ("missing legacy metric", missing)):
        try:
            report_module.build_report(bad)
        except (ValueError, KeyError):
            checks.append(name + " rejected")
        else:
            raise AssertionError(name + " was accepted")
    empty = HERE / "c001" / "empty_mirror"
    empty.mkdir(parents=True, exist_ok=True)
    rejected = verifier.verify_results(empty, final_source_spec=json.loads((HERE / "run_spec.json").read_bytes()))
    assert rejected.get("status") != "COMPLETE_VALID" and rejected.get("runtime_evidence_status") != "VERIFIED"
    checks.append("empty mirror cannot become valid scientific/runtime evidence")
    record = {"format": "e164_artificial_json_engineering_smoke_v1", "status": "PASS",
              "checks": checks, "data": "artificial JSON only; no real old/new scores read",
              "scientific_modules_imported": False, "torch_imported": "torch" in sys.modules,
              "fit_or_score_calls": 0, "supervisor_or_probe_calls": 0, "remote_operations": False,
              "scientific_campaign": "NOT_STARTED"}
    assert record["torch_imported"] is False
    (HERE / "engineering_smoke.json").write_bytes((json.dumps(record, indent=2) + "\n").encode())
    print(json.dumps({"status": "PASS_SYNTHETIC_ENGINEERING_ONLY", "check_groups": len(checks), "scientific_calls": 0}))


if __name__ == "__main__":
    main()
