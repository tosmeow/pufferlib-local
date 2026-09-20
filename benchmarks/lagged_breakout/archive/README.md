# Earlier experiments

These trials are retained for provenance and possible reuse. The selected
policies and final scores are in [experiment_03_gpu](../experiment_03_gpu/README.md).

| Directory | Trial |
| --- | --- |
| `experiment_02_continued/` | Earlier CPU continuation for seed 42 |
| `calibration_gpu_large_batch/` | GPU batch-size calibration |
| `calibration_lr002/` | Lower learning-rate calibration |
| `calibration_velocity/` | Velocity input normalization calibration |
| `calibration_velocity_source/` | Source checkpoint for that calibration |

The cleanup moved files without changing their contents. Historical JSON and
logs retain the paths recorded when those runs happened.
[relocations.json](relocations.json) maps original paths to their new locations
and records SHA-256 hashes; paths are relative to the repository root. Use these
new locations when reopening an archived run.

The final experiment keeps its own history beside the active runs:

- `experiment_03_gpu/history/phases/`: invocation settings and intermediate results.
- `experiment_03_gpu/history/training_logs/`: console logs.
- `experiment_03_gpu/history/evaluations/`: the earlier comparison with a shorter episode cap.
- `experiment_03_gpu/lag_N/seed_S/history/`: previous configurations, checkpoint backups, and superseded reports.
