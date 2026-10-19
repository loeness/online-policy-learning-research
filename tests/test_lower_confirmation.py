from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_lower_confirmation as runner  # noqa: E402


def _config() -> dict:
    return json.loads(runner.CONFIG_PATH.read_text(encoding="utf-8"))


def _write_steps(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "task"))
        writer.writerows((index, "A0+") for index in range(rows))


def test_frozen_main_and_pilot_configs() -> None:
    raw = _config()
    main, phase = runner._phase_config(raw, pilot=False)
    assert phase == "main"
    assert len(main["seeds"]) == 16
    assert main["pair_kinds"] == ["dynamics"]
    assert main["history_regimes"] == ["old_major", "new_major"]
    assert main["future_regimes"] == ["old_major", "new_major"]
    assert main["bootstrap_steps"] == 2048
    assert main["bout_steps"] == 1024
    assert main["max_workers"] == 3

    pilot, phase = runner._phase_config(raw, pilot=True)
    assert phase == "pilot"
    assert pilot["seeds"] == [22089]
    assert pilot["pair_kinds"] == ["dynamics"]
    assert pilot["history_regimes"] == ["old_major"]
    assert pilot["future_regimes"] == ["old_major", "new_major"]
    assert pilot["bootstrap_steps"] == 512
    assert pilot["bout_steps"] == 256


def test_case_payloads_run_only_the_matching_future(tmp_path: Path) -> None:
    config, _ = runner._phase_config(_config(), pilot=False)
    jobs = runner._case_payloads(config, tmp_path, [61001])

    assert len(jobs) == 2
    assert {job["history_regime"] for job in jobs} == {"old_major", "new_major"}
    for job in jobs:
        assert job["pair_kind"] == "dynamics"
        assert job["config"]["future_regimes"] == [job["history_regime"]]
        assert job["config"]["actions"] == ["keep", "replace", "mean", "none"]


def test_case_costs_count_persisted_prefix_and_branch_rows(tmp_path: Path) -> None:
    case_id = "s61001_dynamics_old_major"
    case_folder = tmp_path / "cases" / case_id
    _write_steps(case_folder / "prefix" / "steps.csv", 3)
    branches = []
    for action in ("keep", "replace", "mean", "none"):
        _write_steps(case_folder / "branches" / f"{action}__old_major" / "steps.csv", 7)
        branches.append({"action": action, "future_regime": "old_major"})

    record = {
        "case_id": case_id,
        "seed": 61001,
        "pair_kind": "dynamics",
        "history_regime": "old_major",
        "prefix": {},
        "selectors": {"surrogate_lower": "replace"},
        "branches": branches,
        "costs": {"case_cpu_s": 5.0},
        "errors": [],
    }
    result = runner._augment_case(record, tmp_path)

    assert result["confirmation_selector"] == "surrogate_lower"
    assert result["confirmation_action"] == "replace"
    assert result["prefix"]["raw_environment_steps"] == 3
    assert [branch["raw_environment_steps"] for branch in result["branches"]] == [7] * 4
    assert result["costs"]["prefix_environment_steps"] == 3
    assert result["costs"]["branch_environment_steps"] == 28
    assert result["costs"]["environment_steps_total"] == 31


def test_source_snapshot_contains_runner_config_protocol_and_excludes_local_rules(tmp_path: Path) -> None:
    manifest = runner._snapshot_sources(tmp_path, runner.CONFIG_PATH, runner.PROTOCOL_PATH)
    sources = manifest["sources"]

    runner_rel = runner.RUNNER_PATH.relative_to(ROOT).as_posix()
    config_rel = runner.CONFIG_PATH.relative_to(ROOT).as_posix()
    protocol_rel = runner.PROTOCOL_PATH.relative_to(ROOT).as_posix()
    assert {runner_rel, config_rel, protocol_rel} <= set(sources)
    assert manifest["excluded_local_instructions"] == ["Agent.md", "AGENTS.md"]
    assert not any(Path(name).name.casefold() in {"agent.md", "agents.md"} for name in sources)
    for relative in (runner_rel, config_rel, protocol_rel):
        assert (tmp_path / "source_snapshot" / relative).is_file()


def test_cli_help_is_available_without_starting_experiment(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        runner.main(["--help"])
    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "--pilot" in help_text
    assert "--protocol" in help_text
