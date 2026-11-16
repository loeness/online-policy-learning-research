"""Extract the exact returned run002 inventory, preserving every raw byte."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import stat
import zipfile

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    observation = json.loads((HERE / "remote_result_observation.json").read_text())
    manifest_bytes = (HERE / "return002_manifest.json").read_bytes()
    assert len(manifest_bytes) == observation["manifest_bytes"]
    assert sha(manifest_bytes) == observation["manifest_sha256"]
    manifest = json.loads(manifest_bytes)
    archive_bytes = (HERE / "return002.zip").read_bytes()
    assert len(archive_bytes) == observation["archive_bytes"]
    assert sha(archive_bytes) == observation["archive_sha256"] == manifest["archive_sha256"]
    expected = {row["path"]: row for row in manifest["files"]}
    assert len(expected) == manifest["file_count"] == 250
    assert sum(r["bytes"] for r in expected.values()) == observation["raw_bytes"]
    contents = {}
    with zipfile.ZipFile(HERE / "return002.zip") as archive:
        assert len(archive.infolist()) == len(expected)
        assert {info.filename for info in archive.infolist()} == set(expected)
        for info in archive.infolist():
            name = info.filename
            path = PurePosixPath(name)
            assert not path.is_absolute() and ".." not in path.parts and "\\" not in name
            assert not info.is_dir() and stat.S_IFMT(info.external_attr >> 16) != stat.S_IFLNK
            row = expected[name]
            assert row["remote_path"] == "/data/robot/p001/e163/run002/" + name
            data = archive.read(info)
            assert len(data) == row["bytes"] and sha(data) == row["sha256"]
            contents[name] = data
    destination = HERE / "remote" / "run002"
    assert not destination.exists()
    destination.mkdir(parents=True)
    for name, data in contents.items():
        target = destination / Path(name)
        assert target.resolve().is_relative_to(destination.resolve())
        assert not any(p.is_symlink() for p in (target, *target.parents))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    for name, row in expected.items():
        data = (destination / name).read_bytes()
        assert len(data) == row["bytes"] and sha(data) == row["sha256"]
    receipt = {"state": "REMOTE_BYTE_IDENTITIES_VERIFIED", "files": manifest["files"],
               "file_count": len(expected), "raw_total_bytes": sum(r["bytes"] for r in expected.values()),
               "archive_sha256": sha(archive_bytes), "manifest_sha256": sha(manifest_bytes),
               "scope": "all run002 raw evidence; no model deserialization or reserialization"}
    (HERE / "mirror_manifest.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"mirror_files": len(expected), "bytes": receipt["raw_total_bytes"], "hashes": "PASS"}))


if __name__ == "__main__":
    main()
