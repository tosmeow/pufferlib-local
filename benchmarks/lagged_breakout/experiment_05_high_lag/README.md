# Further training for lags 4 and 8

Completed 2026-09-05. Both runs received **50,069,504 additional decisions**.

## Results

Before and after are evaluated on the same 1,024 fresh episodes at each policy's
training lag and volatility 6. These seeds differ from the preceding refinement's
audit, so the baseline scores also differ from that earlier report.

| Lag | Score before → after | Full clears before → after | Mean winning time before → after |
| --- | ---: | ---: | ---: |
| 4 | 863.01 → 863.56 | 1,014/1,024 → 1,023/1,024 | 267.7 s → 269.6 s |
| 8 | 856.25 → 857.37 | 925/1,024 → 942/1,024 | 268.1 s → 266.6 s |

Winning times are simulated seconds and exclude failed episodes. On seeds where
both versions won, the mean change was +2.10 seconds for lag 4 (standard error
1.57) and −1.71 seconds for lag 8 (standard error 1.47). Neither provides a clear
speed improvement in this pass.

The observed score gains were +0.55 and +1.12, with paired standard errors 0.73
and 1.84 respectively. Those small gains should not be read as established
improvements in expected score. Lag 4's full-clear rate was 99.9% in this audit;
lag 8 still failed to clear 82 of 1,024 episodes. Neither run sustained the
stricter 863.5 validation target for the required three final checks, although
lag 4's selected checkpoint was close to the score ceiling on the final audit.

On the common original game, mean scores changed from 564.45 to 547.03 for lag 4
and from 125.28 to 129.25 for lag 8. All 4,096 own-setting before/after audit
episodes and all 128 new original-game episodes completed within their caps.

This extension trains only the seed-42 policies at lags 4 and 8. The lag-0 and
lag-2 policies in [experiment_04_refinement](../experiment_04_refinement/README.md)
remain unchanged.

Lag 4 starts from its stronger selected checkpoint in
[experiment_03_gpu](../experiment_03_gpu/README.md). Lag 8 starts from the improved
checkpoint in experiment_04_refinement. Both retain the latest cumulative
training counters, including the earlier lag-4 refinement whose weights were
discarded. Optimizer momentum is restarted from the selected weights.

## Protocol

- Another 50 million decisions per policy, rounded up to a rollout batch.
- Learning rate 0.00005, halved from the last extension; 4,096 environments and a
  16,384-sample minibatch on the Mac GPU.
- Training remains at each policy's own lag and volatility 6. Discounting and
  rewards are unchanged.
- Checkpoint selection uses 512 validation episodes starting at seed 200000.
  The selected incumbent is re-scored on this expanded cohort first.
- The 863.5/864 target is an empirical progress marker; this pass runs to the
  full step budget even if the target is reached.
- Before/after audits use 1,024 common episodes starting at seed 600000. These
  seeds were not used by the earlier experiments. Earlier audits informed which
  parents to continue, so their cohorts are not reused as this pass's audit.
- Original-game evaluation uses the same 64 seeds starting at 100000, with a
  120,000-decision cap. Audit and original-game scores do not select checkpoints
  within this extension.
- The final comparison includes win counts and time to reach 864, including
  paired finishing-time differences on seeds where both versions win.

## Artifacts

- [refinement_plan.json](refinement_plan.json): exact commands, source hashes,
  separate policy/state parents, and step budgets.
- `progress.json`: current stage; `lag_N/seed_42/status.json`: per-policy progress.
- `history/training_logs/`: training and evaluation output.
- [Paired audit](comparison/paired_audit.json) and
  [completion times](comparison/completion_times.json): score and timing
  comparisons, win counts, standard errors, and checkpoint hashes.
- [Summary CSV](summary.csv), [summary JSON](summary.json), and
  [evaluation reference](final_evaluation.json): final results.

The recorded driver is [history/run_refinement.py](history/run_refinement.py).
Only lags 4 and 8 are stored here; the unchanged lag-0 and lag-2 policies remain
in experiment_04_refinement.
