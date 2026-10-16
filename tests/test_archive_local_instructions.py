"""Private root instructions must not enter the research export manifest."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_archive_manifest as archive


@pytest.mark.parametrize("name", ["Agent.md", "AGENTS.md", "aGeNtS.Md"])
def test_root_local_instructions_are_never_hashed_but_research_data_remains(tmp_path, monkeypatch, name):
    (tmp_path / name).write_text("private local instruction", encoding="utf-8")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "evidence.csv").write_text("reward\n1\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("outputs/\n", encoding="utf-8")
    monkeypatch.setattr(archive, "ROOT", tmp_path)
    original_hash = archive.sha256_and_size

    def guarded_hash(path):
        assert path.name != name, "private root file contents must never be read"
        return original_hash(path)

    monkeypatch.setattr(archive, "sha256_and_size", guarded_hash)
    manifest = archive.build_manifest()
    paths = {row["path"] for row in manifest["files"]}
    assert name not in paths
    assert "outputs/evidence.csv" in paths
