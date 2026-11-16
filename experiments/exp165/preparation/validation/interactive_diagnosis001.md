# E165 CHECKS preparation: first login attempt

The manually started helper wrote
`interactive_prepare_receipt_20261029T153419Z.json`. It reported stdout EOF
before the remote capacity READY frame. It did not send the archive, start
CHECKS, or create an execution authorization. The receipt does not establish
whether authentication or remote Python started, and did not record the SSH
exit code.

The user subsequently completed a separate manual password-authenticated SSH
command and obtained `AUTH_OK`; that supports the availability of the account
and password at that time. It does not prove that password prompting works
when the helper pipes SSH stdin or that remote capacity is sufficient.

Local-only configuration inspection accepted the helper's full OpenSSH
arguments. A no-network proxy probe on Windows OpenSSH 9.5p2 showed that
`RemoteCommand=none` did not cause the proposed command-conflict failure.
The precise reason for the original EOF is still unknown.

The local helper was changed only to include the SSH process exit status and
whether it had to be terminated in later receipts. This does not alter the
approved transfer manifest, controlled stage specifications, resource caps,
or the rule requiring independent authorization to run CHECKS. No automatic
retry, server modification, historical model read, or experiment occurred in
this diagnosis.
