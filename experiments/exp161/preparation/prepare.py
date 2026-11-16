"""Build the explicit E161 source and input allowlist; no environment calls."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[3]
PREP = Path(__file__).resolve().parent


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    runner = ast.parse((ROOT / 'scripts/run_retention_candidate.py').read_text(encoding='utf-8'))
    selected: set[Path] = set()
    for node in runner.body:
        if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id in
                                               {'SOURCE_FILES', 'E161_SOURCE_FILES'} for n in node.targets):
            for item in node.value.elts:
                selected.add(Path(ast.literal_eval(item.args[0])))
    selected.update(Path(value) for value in (
        'scripts/launch_e161.py', 'scripts/audit_retention_candidate.py',
        'scripts/report_retention_candidate.py', 'data/foundation.pt',
        'data/pretrain_source.npz', 'requirements.txt', 'requirements-lock.txt',
        'tests/test_e161_plan.py', 'tests/test_e161_analysis.py',
        'tests/test_report_retention_candidate.py', 'experiments/exp161/README.md',
        'experiments/exp161/preparation/seed-map.csv',
        'experiments/exp161/preparation/seed-audit.json',
        'experiments/exp161/preparation/schedules.json',
        'experiments/exp161/preparation/checks.json',
        'experiments/exp161/preparation/power-plan.json',
        'experiments/exp161/preparation/prepare.py',
    ))

    def add_module(name: str) -> None:
        parts = name.split('.')
        for base in (Path('.'), Path('src')):
            for size in range(1, len(parts)+1):
                module = base.joinpath(*parts[:size])
                for path in (module.with_suffix('.py'), module / '__init__.py'):
                    if (ROOT / path).is_file():
                        selected.add(path)

    scanned = set()
    while True:
        pending = [path for path in selected if path.suffix == '.py' and path not in scanned]
        if not pending:
            break
        for path in pending:
            scanned.add(path)
            module_parts = list(path.with_suffix('').parts)
            if module_parts[0] == 'src':
                module_parts = module_parts[1:]
            for node in ast.walk(ast.parse((ROOT / path).read_text(encoding='utf-8-sig'))):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        add_module(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        prefix = '.'.join(module_parts[:-node.level])
                        module = '.'.join(part for part in (prefix, node.module) if part)
                    else:
                        module = node.module or ''
                    add_module(module)
                    for alias in node.names:
                        add_module('.'.join(part for part in (module, alias.name) if part))

    payload: dict[str, bytes] = {}
    for relative in sorted(selected):
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT.resolve()) or path.is_symlink():
            raise ValueError(f'unsafe source: {relative}')
        if any(part.casefold() in {'agent.md', 'agents.md', '.git'} for part in relative.parts):
            raise ValueError(f'private source: {relative}')
        payload[relative.as_posix()] = path.read_bytes()
    names = sorted(payload)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(['git', 'diff', '--binary', 'HEAD', '--', *names], cwd=ROOT)
    status = subprocess.check_output(['git', 'status', '--porcelain', '--', *names], cwd=ROOT, text=True)
    payload['provenance/changes.patch'] = diff
    payload['provenance/git.json'] = json.dumps({
        'git_commit': commit, 'source_status': status,
        'identity': 'Commit plus all manifested executed bytes, including dirty/untracked sources.',
    }, indent=2).encode() + b'\n'
    manifest = {'format': 'e161_source_bundle_v1', 'git_commit': commit,
                'remote_source': '/data/robot/p001/s001',
                'files': [{'path': name, 'bytes': len(data), 'sha256': sha(data)}
                          for name, data in sorted(payload.items())]}
    raw_manifest = json.dumps(manifest, indent=2).encode() + b'\n'
    (PREP / 'source-manifest.json').write_bytes(raw_manifest)
    (PREP / 'changes.patch').write_bytes(diff)
    payload['bundle-manifest.json'] = raw_manifest
    target = PREP / 's001.zip'
    if target.exists():
        raise FileExistsError(target)
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            item = zipfile.ZipInfo(name, date_time=(2026, 10, 21, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, data)
    metadata = {'file': target.name, 'sha256': sha(target.read_bytes()),
                'bytes': target.stat().st_size, 'unpacked_bytes': sum(map(len, payload.values())),
                'file_count': len(payload), 'manifest_sha256': sha(raw_manifest)}
    (PREP / 'bundle.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
