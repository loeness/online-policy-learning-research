# Superseded diagnostic v1

The v1 candidate norm must not be used for comparison with the mechanism pilot. It averaged two 27-coefficient effective task matrices without averaging across their three bands, while the pilot candidate is `common_mean_effective`: the mean across two slots and three bands, cast to float32, with 9 coefficients. The v1 cross-dimension candidate comparison is invalid. The stored shared-parent norms remain valid.

The original v1 `summary.json` and `checkpoint_magnitudes.csv` are preserved unchanged. The producer code used for v1 is preserved as `diagnose_retention_content_v1_snapshot.py` with SHA-256 `198b2fea83249100b5a736f3a83ee94e48bc4bf687630f378b8665874c2a17c3`. The v1 summary contains no producer-code hash, so there is no stored summary hash to compare against.
