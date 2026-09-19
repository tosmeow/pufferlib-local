# Further training of the four viewer policies

Completed 2026-09-05. Each run received **50,069,504 additional decisions**.

## Results

The before/after scores below use the same 512 fresh episodes at each policy's
own lag and volatility 6. Winning times are simulated seconds, averaged over
successful full clears; win counts are included because failures do not have a
time to reach 864.

| Training lag | Score before → after | Wins before → after (of 512) | Mean winning time before → after |
| --- | ---: | ---: | ---: |
| 0 | 860.46 → 864.00 | 492 → 512 | 274.7 s → 275.6 s |
| 2 | 863.88 → 864.00 | 511 → 512 | 271.2 s → 266.0 s |
| 4 | 863.77 → 861.37 | 511 → 500 | 270.1 s → 270.9 s |
| 8 | 852.38 → 860.52 | 452 → 473 | 271.0 s → 264.9 s |

On seeds where both policies won, mean finishing-time changes were +0.79,
−5.11, +0.48, and −6.42 seconds respectively. Their paired standard errors
were 1.97, 2.02, 2.51, and 2.16 seconds. Lags 2 and 8 show modest speed gains
in this sample, about 2%; lags 0 and 4 show no clear speed improvement.

The extension improved audit scores at lags 0 and 8, and removed the one sampled
failure at lag 2. Lag 4's previous checkpoint scored higher on the audit.
These finite-sample results do not establish full convergence. The new policies
were selected on validation scores, and the original experiment remains intact.

Transfer to the original, unlagged, zero-volatility game uses the same 64 seeds
before and after:

| Training lag | Original-game score before → after |
| --- | ---: |
| 0 | 863.67 → 864.00 |
| 2 | 861.50 → 835.70 |
| 4 | 564.45 → 453.06 |
| 8 | 154.23 → 125.28 |

All 4,096 own-setting before/after audit episodes and all 256 new original-game
episodes completed within their caps. Better performance at the training lag
did not produce better transfer for the delayed policies.

## Protocol

This run continues lags 0, 2, 4, and 8 with training seed 42 from the selected
policies in [experiment_03_gpu](../experiment_03_gpu/README.md). That experiment
remains the reference comparison, with its checkpoints preserved.

Each policy receives another 50 million decisions (rounded up to a rollout
batch), at its own lag and volatility 6. Training restarts from its selected
weights with fresh optimizer momentum, retaining the original step counters.
All four use a learning rate of 0.0001, 4,096 environments, and a 16,384-sample
minibatch on the Mac GPU. Lag 0's velocity encoder is normalized with a
function-preserving weight conversion.

Checkpoint selection uses 256 episodes at the training lag, starting at seed
200000. The incumbent is re-evaluated on that larger cohort before new candidates
are compared. Reaching the old 850/864 target does not stop this extension.

After training, the old and new selected policies are compared on the same 512
previously unused episodes starting at seed 500000. Original-game evaluation
uses 64 common episodes starting at seed 100000 and a 120,000-decision cap.
Audit and original-game scores do not select checkpoints. These empirical
checks can measure improvement and stability; they cannot prove full convergence.

There is no explicit time penalty or completion bonus. Training discounts
per-decision rewards with gamma 0.9721246598992744 after clipping rewards to
[-1, 1]. Decisions are 4/60 seconds apart, so a reward delayed by one simulated
second has about 65% of the weight. This favors earlier brick rewards, but
checkpoint selection ranks undiscounted score rather than full-clear time.

## Artifacts

- [refinement_plan.json](refinement_plan.json): exact commands, budgets, and parent checkpoint hashes.
- `progress.json`: current stage of the run.
- `lag_N/seed_42/status.json`: training progress and validation history.
- `history/training_logs/`: training and evaluation output.
- [Paired audit](comparison/paired_audit.json): before/after scores, win rates, standard errors, and checkpoint hashes.
- [Completion times](comparison/completion_times.json): winning times, win counts, and changes on paired winning seeds for both environments.
- [Summary CSV](summary.csv), [summary JSON](summary.json), and [evaluation reference](final_evaluation.json): final results.

The recorded driver is [history/run_refinement.py](history/run_refinement.py).
To watch the new policies after training:

```bash
.venv/bin/python scripts/watch_lagged_breakout.py \
  --experiment benchmarks/lagged_breakout/experiment_04_refinement
```
