# Lagged Breakout — completed continuation experiment

All 12 policies (lags 0, 2, 4, 8; training seeds 42, 43, 44) met the empirical near-ceiling target. This is not a proof of mathematical optimality. Recorded 2026-09-05.

## Final scores

Scores below average the three trained policies at each lag. The maximum episode return is 864. Own-environment scores use 256 previously unseen episodes per policy, with volatility 6 and the training lag. Original-game scores use 64 common episodes per policy at lag 0 and volatility 0. Every episode completed.

| Training lag (microsteps) | Own environment | Original game |
| --- | ---: | ---: |
| 0 | 862.70 | 863.89 |
| 2 | 863.11 | 863.17 |
| 4 | 863.88 | 446.97 |
| 8 | 856.43 | 185.59 |

The larger audit scores for individual policies range from 855.42 to 864.00. Higher-lag policies perform near the ceiling under their own execution delay, but transfer poorly to the original game. Command-switch rates on the original game are not monotone in the training lag.

## Training and selection

- The source `experiment_01` checkpoints were preserved. Training used adaptive budgets, smaller learning rates, and velocity input normalization. Configurations before each resume are archived in each run's `history/` directory.
- Five stalled runs were initialized from stronger policies with the same training seed, then trained at their target lag. `warm_start_phases.json` records source checkpoints, hashes, source training counts, and adoption steps. These are fine-tuned policies sharing pretraining across lags; this is an exploratory convergence comparison, not a fixed-budget experiment from independent initializations.
- Qualification required a mean return of at least 850 on three consecutive validation checkpoints and on a separate 64-episode check. Validation used seeds starting at 200000, initially 32 episodes; lag 8, seed 44 expanded to 128, with its incumbent re-scored before selection continued.
- The 300000 qualification cohort was reused during retries. Final 256-episode audits used seeds starting at 400000, which were kept out of checkpoint selection. All 3,072 audit episodes completed, and every policy averaged at least 850.
- Original-game scores were not used to choose checkpoints. The final common evaluation uses seeds starting at 100000 and a 120,000-decision cap. Four episodes of lag 4, seed 43 exceeded the earlier 30,000-decision cap; they completed by decision 40,685. All 768 final original-game episodes completed. The earlier reports remain archived.
- Observations expose lag and pending commands. Evaluating at lag 0 also changes those inputs to the unlagged actuator state. The transfer results therefore do not isolate the paper’s smoothing mechanism.

## Artifacts

- [Per-policy scores and behavior](summary.csv), [aggregates](summary.json), and [learning curves](learning_curves.png).
- [Return and behavior comparison](policy_comparison.png).
- [Common-original episodes](evaluations/evaluation_1788639761288245000/episodes.csv), [trajectories](evaluations/evaluation_1788639761288245000/trajectories.csv), and [evaluation settings and raw results](evaluations/evaluation_1788639761288245000/results.json). [final_evaluation.json](final_evaluation.json) identifies this evaluation and its cohort sizes.
- `lag_N/seed_S/policy.bin`: selected policy. `config.json`: matching model/environment settings. `training_state.pt`: latest resumable state, which may differ from the selected policy. Native environment state is not serialized.
- `audit_training_environment.json`: the independent 256-episode audit and checkpoint hash. `holdout_original_environment.json`: the final common-original evaluation.
- `lag_N/seed_S/validation/`: periodic validation snapshots, including cohort recalibration.
- `lag_N/seed_S/history/`: earlier configurations, checkpoint backups, and superseded reports.
- `history/phases/`, `history/training_logs/`, and `warm_start_phases.json`: training provenance. Phase result files are intermediate snapshots; the final comparison is in `summary.json`.
- `history/evaluations/`: the earlier 30,000-decision-cap comparison.
- [Environment and runner instructions](../../../ocean/lagged_breakout/README.md).
