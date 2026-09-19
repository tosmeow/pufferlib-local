# Benchmarks

Saved experiment data, trained policies, and analysis notebooks. Run the commands
below from the repository root.

| Environment | Start here | Contents |
| --- | --- | --- |
| Lagged Breakout | [Experiment index](lagged_breakout/README.md) | Final comparison, 12 trained policies, baseline, and training history |
| Queue reactive | [Analysis index](queue_reactive/README.md) | Constant-alpha drift data and notebook |

To reopen the four-agent Breakout comparison:

```bash
.venv/bin/python scripts/watch_lagged_breakout.py \
  --experiment benchmarks/lagged_breakout/viewer_latest
```

The viewer set references lags 0 and 2 from `experiment_04_refinement` and lags
4 and 8 from `experiment_05_high_lag`.
Current results and runnable checkpoints stay in the experiment directories;
earlier trials live under `archive/`, and intermediate training artifacts under
`history/` and `validation/`. The September 2026 cleanup preserved the original
checkpoints, observations, scores, and logs.
