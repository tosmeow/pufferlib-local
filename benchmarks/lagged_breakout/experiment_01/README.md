# Initial training baseline

This short run contains the original policies for lags 0, 2, 4, 8 and training
seeds 42, 43, 44. Each policy was trained for about one million steps. These
checkpoints are the source used by the later continuation experiments.

- `lag_N/seed_S/`: policy, configuration, training log, and training result.
- `evaluations/`: the baseline comparison on the common original game.

For the completed training and final comparison, see
[experiment_03_gpu](../experiment_03_gpu/README.md).
