# C1 tensor alternatives: execution summary

All runs used low capacity. Validation used seed 919; formal runs used seeds 181, 193, and 211. Each tensor method was selected from the three frozen candidates; the tree reference used the Step 2 selection.

| Arm | Validation scores (candidates 0 / 1 / 2) | Selected | Formal score by seed | Mean formal score | Mean final success | Mean history success |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `tensor_learned` | 0.9406, 0.8160, 0.8486 | candidate 0 `{'learning_rate': 0.003, 'ent_coef': 0.06}` | 181: 0.9980, 193: 0.9981, 211: 1.0000 | 0.9987 | 1.0000 | 1.0000 |
| `output_mixer` | 0.4927, 0.6634, 0.4131 | candidate 1 `{'learning_rate': 0.01, 'ent_coef': 0.06}` | 181: 1.0000, 193: 0.8805, 211: 0.9954 | 0.9586 | 0.9568 | 0.9935 |
| `hierarchical_output_mixer` | 1.0000, 0.9952, 0.9625 | candidate 0 `{'learning_rate': 0.003, 'ent_coef': 0.06}` | 181: 1.0000, 193: 1.0000, 211: 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `tree_reference` | Prior Step 2 selection | `{'learning_rate': 0.01, 'ent_coef': 0.06}` | 181: 0.9875, 193: 1.0000, 211: 0.9943 | 0.9939 | 1.0000 | 1.0000 |

Completed 21 of 21 jobs; failures: 0.
Training interactions: 1,047,552; evaluation interactions: 287,232.
Summed training CPU time: 0.27 hours; summed measured training elapsed time: 0.28 hours.
All formal runs stayed within both configured caps: True.

The detailed seed-level ledger, candidate results, byte peaks, checkpoints, and source hashes remain in `frozen_settings.json` and the per-run result JSON files. `execution.log` contains the complete progress stream.
