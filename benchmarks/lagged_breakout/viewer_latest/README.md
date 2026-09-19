# Latest four-policy viewer set

This folder references the latest completed seed-42 policies as of 2026-09-06.
It contains directory links, not copies of model weights.

| Training lag | Experiment |
| --- | --- |
| 0, 2 | [experiment_04_refinement](../experiment_04_refinement/README.md) |
| 4, 8 | [experiment_05_high_lag](../experiment_05_high_lag/README.md) |

From the repository root:

```bash
.venv/bin/python scripts/watch_lagged_breakout.py \
  --experiment benchmarks/lagged_breakout/viewer_latest
```

The four games start automatically, all using original dynamics (lag 0,
volatility 0) and the same game seed. Space pauses or resumes; R replays the
seed; N advances to the next seed. Use the speed menu to change playback speed.
Close the window to stop.

These references stay pinned to the experiments above until explicitly updated.
