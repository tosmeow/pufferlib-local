# Rock / counter / self-play — 2026-09-27

Equal training mixture; hidden size 64, four layers, seed 73, learning rate
0.0015, 199,950,336 steps. Evaluation: 64 × 1,000 rounds per opponent.

| Opponent | Mean round reward | Near-optimal streak starts: median (range) |
|---|---:|---|
| Rock | +0.99892 | Round 2 (2–3) |
| Counter | +0.99300 | Round 7 (6–32) |
| Uniform | +0.00463; 95% CI includes zero | Uniformity criterion never met |

Streak = three rounds with every action probability within 0.01 of its target;
confirmation occurs two rounds after the listed start. Counter had two later
departures; every match stayed within tolerance from round 38 onward. Both
predictable bots were beaten every round after round 50.

**Self-play did not yield a uniform fallback.** Against uniform, average action
probabilities were 27.7% / 42.8% / 29.5%, with mean per-round entropy 0.291 nats
(uniform: 1.099). No match met the three-round uniformity criterion. This is
one seed; exploitability was not measured. Counter adaptation was slower than
the previous scripted mixture (median round 5).

Against itself, rounds 101–1,000 also remained nonuniform: average probabilities
9.6% rock / 47.0% paper / 43.4% scissors; mean entropy 0.172 nats. Near-zero
self-play reward therefore does not demonstrate Nash convergence.

Jobs **897** (training) and **898** (diagnostic) completed successfully.
CPU/GPU parity passed (max probability error 3.97e-6). Key artifact hashes,
publication, and scratch cleanup verified. Snapshot:
`dcd8509cc960f37aa2f30b9c92c0b72e8d9f6f0f`.

[Training outputs and checkpoints](/cluster/users/tosma/projects/pufferlib-local/jobs/d9442613-8e10-4449-bae0-dea177c152cf/artifacts)
· [Per-match adaptation results](/cluster/users/tosma/projects/pufferlib-local/jobs/849b5b90-6fde-42c4-b1cc-7389f4b3ae47/artifacts)

## Longer monitored runs — submitted 2026-09-27

Fresh seed-73, hidden-64, four-layer runs; LR 0.0015 with the existing annealing
schedule, 2 billion requested steps each. Remaining probability is split equally
between rock and counter. These runs are not yet conclusions; use Slurm and
receipts for current state.

| Self-play share | Slurm job | Durable job directory |
|---|---:|---|
| 80% | 902 | `/cluster/users/tosma/projects/pufferlib-local/jobs/a0e819c5-e4ba-4889-844b-68fc1449656e` |
| 95% | 903 | `/cluster/users/tosma/projects/pufferlib-local/jobs/96d3d5b1-5bad-4393-8a03-de7bfe621be7` |
| One-third (duration control) | 904 | `/cluster/users/tosma/projects/pufferlib-local/jobs/57dbf8d6-aec5-4a99-abfd-e6e96ef08044` |

All use snapshot `07b10e9ac68cb1cef2465c9d03ee806f7c69ef38`. Live checkpoint
metrics appear in `rps_monitor_history.json` in each job directory. Final outputs
are published under `artifacts`; inspect `monitor_history.json`, individual
`monitor/STEP/summary.json` files, and final `evaluation` / `parity` reports.
No adaptive stopping is enabled. Warnings identify possible collapse or failure
to learn; they do not establish a mechanism. Monitoring evaluates every ~200M
steps and can lag training while CPU evaluation completes.

Validation: job 900 failed before training because the headless test compilation
had no raylib header. Fixed with an empty test-only header (RPS has no graphics
calls). Job 901, snapshot `aab880e7badf503d8167cc0ffee57fa1b725af71`, then
completed with exit 0, publication and scratch cleanup confirmed. Sanitized
weighted-opponent tests, two training updates, both checkpoint evaluations,
final evaluation/adaptation and CPU/GPU parity passed (max error 7.49e-8).
Checkpoint, INI and JSON hashes matched the publication index. Smoke outputs:
`/cluster/users/tosma/projects/pufferlib-local/jobs/02770aec-a0d8-4918-8fef-63a42435500e/artifacts`.
The full runs add a durable copy of monitoring history to that tested driver.

### Observations from the longer runs

All three reached 1,999,962,112 training steps. CPU checkpoint evaluations
used the same 64 seeded matches per opponent at each checkpoint. Final warm
self-play entropy excludes rounds 1–100; uniform entropy is 1.098612 nats.

| Self-play share | Final self-play entropy | Rock reward | Counter reward |
|---|---:|---:|---:|
| One-third | 3.43e-62 | +0.499703 | -0.343156 |
| 80% | 3.79e-6 | +0.998188 | -0.733453 |
| 95% | 1.97e-5 | +0.995172 | +0.493734 |

Monitoring captured deterioration, not merely a poor final endpoint. At 200M,
the 80% run scored +0.976 against counter; by 800M it scored -0.0012 with
self-play entropy 0.0002. The 95% run oscillated: counter reward -0.9733 at
600M, +0.4648 at 800M, then -0.9925 at 1.6B. The one-third control lost
almost every round against both scripted bots at 1B, before partial recovery.
All checkpoints remain available; no early stopping was applied.

These are provisional single-seed observations. More steps and larger
self-play shares did not prevent low-entropy collapse in these runs. They do
not isolate the cause. Extending the budget also extends cosine LR annealing:
at 200M in a 2B run the LR is still about 97.6% of its initial value, whereas
the earlier 200M run was ending its schedule. The same seed does not ensure
identical stochastic trajectories across concurrent training runs.

Verification limitation: jobs 902 and 903 completed training and CPU evaluation
but failed their final strict CPU/GPU parity assertion. Maximum errors were
4.16923e-5 and 1.37955e-5 respectively (tolerance 1e-5); no tolerance was relaxed.
Thus their overall job state is failed / exit 1, not successful. Publication
and scratch cleanup succeeded; 28 checkpoint/config/JSON hashes per job were
verified against the artifact index. GPU training logs independently reported
final aggregate entropy approximately 0.000 and 0.005 respectively, supporting
low-entropy behavior without validating every CPU rollout or reward estimate.
Resolve parity before treating the CPU measurements as fully validated.

Job 904 likewise finished training/evaluation but failed parity: maximum error
1.803891e-4 versus tolerance 1e-5. Slurm failed / exit 1; publication and scratch
cleanup confirmed, and all 28 checkpoint/config/JSON hashes verified. Its GPU
training log also reported final aggregate entropy 0.000. All three research
jobs are terminal; none remains active. Next: investigate CPU/GPU numerical
agreement on these saturated models, then consider a lower LR or a fixed
decay schedule and entropy-strength comparison before spending more steps.
These are proposed diagnostics, not user-agreed scientific conclusions.
