# Queue-reactive benchmarks

Open [constant_alpha_drift.ipynb](constant_alpha_drift.ipynb) to inspect the stored
10-second mid-price response across constant-alpha scales. It reads
[constant_alpha_drift.json](constant_alpha_drift.json), which includes individual
sequence drifts as well as aggregate statistics. The notebook finds the data
when launched from this directory or the repository root.

To regenerate the data, build the queue-reactive backend, then run from the
repository root:

```bash
puffer build queue_reactive --cpu
python scripts/benchmark_queue_constant_alpha_drift.py
```

The separate `scripts/benchmark_queue_alpha_drift.py` measures responses to
stochastic alpha and writes `alpha_drift_results.json` in this directory.
See the [environment documentation](../../ocean/queue_reactive/README.md) for
the definitions and benchmark options.
