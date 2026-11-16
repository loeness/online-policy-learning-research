# E165 interactive preparation helper

Status: implementation prepared for review only. The helper has not been run,
and no SSH connection or remote operation was attempted while implementing it.

Run it manually from the project PowerShell window:

```powershell
python .\experiments\exp165\preparation\validation\interactive_prepare.py
```

The helper requires Windows and native OpenSSH. It accepts no arguments. If
native SSH asks for authentication, type it only into the native SSH console
prompt. The Python helper never reads PowerShell stdin, never asks for a
password, and does not put a credential in a command argument, environment
variable, file, or receipt. SSH stderr is inherited by the console; SSH stdin
is a pipe for the protocol and archive. OpenSSH's ability to read a password
from the Windows console with stdin redirected has not been verified here. If
the native prompt is not usable, or remote `READY` does not arrive within 180
seconds, the helper stops and sends no 42-file archive. Do not enter a password
at a Python prompt or retry automatically. Host-key checking is strict, and
OpenSSH host-key updates are disabled; an unknown host key fails closed.

The helper checks the approved transfer-manifest SHA-256, frozen candidate
SHA-256, coordinator directive, all 42 source lengths and hashes, and the
source path allowlist before starting SSH. Local sources must be regular files
inside the workspace, with no symlink or reparse-point component. It excludes
all files outside the 42 manifest entries and rejects instruction files, Git
paths, and non-source/model-like file extensions.

The SSH protocol first waits for remote capacity admission. The remote Python
command opens `/data/robot/p001` component by component without following
symlinks, verifies the `robot` account, measures project bytes without reading
file contents, and checks free bytes and inodes. Project bytes are the sum of
`st_size` for non-directory entries, with symlinks counted only by their own
entry size and never followed. The non-E165 baseline excludes `e165`; E165
usage is measured separately and rejects symlinks, non-regular files, and hard
links. Admission requires the remaining 128 MiB E165 allowance to fit the
project's available bytes, the 100 GB project cap, and at least 256 free
inodes. After transfer, at least 128 free inodes must remain.

The remote Python source is zlib-compressed and base64-encoded in the SSH
command. Before spawning OpenSSH, the helper estimates the Windows command-line
length conservatively and rejects anything at or above 30,000 characters; it
also checks the final Windows-quoted command line against that bound.

After the first `READY`, the helper sends only the approved manifest metadata
and the dynamically bound CHECKS spec metadata. The remote side rechecks the
baseline and destination paths, then emits `TRANSFER_READY`. Only then does
the helper stream the in-memory tar archive. The remote validates the full
archive, exact 42 paths, lengths, and SHA-256 values before writing. It uses
exclusive file creation and does not overwrite or delete existing evidence.
The archive is never saved to disk. The helper then creates
`/data/robot/p001/e165/o001/stage_checks.json` with
`resource_authorized: true`, the measured non-E165 baseline, and its UTC
admission time. It creates no execution-authorization file and starts no
supervisor, test, model, fit, score, audit, or environment interaction.

On success or failure, it attempts to write a unique local JSON receipt under
this directory. The receipt contains only protocol status, capacity values,
paths, byte counts, and hashes. A failure after archive transmission can leave
partial remote files; those are preserved, and the helper does not retry.

This is static implementation evidence only. The interactive Windows OpenSSH
authentication path, the remote protocol, and the remote filesystem behavior
remain unverified until the user runs the helper under the already authorized
preparation scope.
