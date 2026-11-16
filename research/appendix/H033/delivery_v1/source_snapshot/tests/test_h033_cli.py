"""Synthetic end-to-end packet interoperability for the three-stage CLI."""
import json
from pathlib import Path
import subprocess
import sys

from test_h033_diagnostic import config, feature_packet, outcome_packet


def test_crossfit_cli_accepts_packets_and_never_overwrites(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name, value in (("config", config()), ("features", feature_packet()), ("outcomes", outcome_packet())):
        (tmp_path / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "crossfit.json"
    command = [
        sys.executable, str(root / "scripts/h033_diagnostic.py"),
        "--config", str(tmp_path / "config.json"), "crossfit",
        "--features", str(tmp_path / "features.json"),
        "--outcomes", str(tmp_path / "outcomes.json"), "--output", str(output),
    ]
    first = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["format"] == "h033_crossfit_v1"
    assert len(result["rows"]) == 16
    assert result["execution"]["environment_interactions"] == 0
    assert len(result["input_packets"]["features"]["sha256"]) == 64
    previous = output.read_bytes()
    second = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
    assert output.read_bytes() == previous
