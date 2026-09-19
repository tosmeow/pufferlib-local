# Lagged Breakout benchmarks

Start with the [high-lag extension](experiment_05_high_lag/README.md) for the
latest lag-4 and lag-8 results, and the
[four-policy refinement](experiment_04_refinement/README.md) for lag 0 and 2
and the preceding before/after comparison. The
[reference experiment](experiment_03_gpu/README.md) contains the earlier
12-policy comparison, plots, and training protocol.

| Directory | Role |
| --- | --- |
| [viewer_latest](viewer_latest/README.md) | Launch all four latest completed seed-42 policies together |
| [experiment_05_high_lag](experiment_05_high_lag/README.md) | Completed 50-million-step extension of lags 4 and 8, with a fresh 1,024-episode audit |
| [experiment_04_refinement](experiment_04_refinement/README.md) | Completed 50-million-step extension of the four seed-42 policies, with paired scores and finishing times |
| [experiment_03_gpu](experiment_03_gpu/README.md) | Final 12 policies: lags 0, 2, 4, 8 and training seeds 42, 43, 44 |
| [experiment_01](experiment_01/README.md) | Original short training run; source checkpoints for continuation |
| [archive](archive/README.md) | Earlier CPU continuation and calibration trials |

Within `experiment_03_gpu`, `summary.csv` and `summary.json` contain the final
comparison. `final_evaluation.json` identifies the common-original evaluation
under `evaluations/`. Each `lag_N/seed_S/` directory contains its selected
`policy.bin`, matching `config.json`, resumable state, and final audit reports.
Periodic validation reports are in `validation/`; older configurations and
checkpoint backups are in `history/`.

Experiment-wide `history/phases/` holds training invocations and partial result
snapshots. Those snapshots describe their particular training phase; use
`summary.json` for the final scores across all policies.

```bash
# Four policies, original dynamics, same environment seed.
.venv/bin/python scripts/watch_lagged_breakout.py \
  --experiment benchmarks/lagged_breakout/viewer_latest --game-seed 100000

# Rebuild the summary from the saved audits; no training or rollout required.
.venv/bin/python scripts/summarize_lagged_breakout.py \
  --output benchmarks/lagged_breakout/experiment_03_gpu
```

See the [environment documentation](../../ocean/lagged_breakout/README.md) for
training, continuation, and evaluation commands.
