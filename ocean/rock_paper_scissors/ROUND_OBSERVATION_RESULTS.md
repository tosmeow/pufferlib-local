# Normalized round observation — 2026-09-27

User hypothesis: exposing elapsed match history may help the recurrent policy
learn when to trust its accumulated state. Feature: `completed_rounds / 999`,
zero at opening, one before the final action, reset each match. It is history
length, not a direct confidence estimate. No opponent identity is supplied.

Paired fresh runs, 200M requested steps each, one-third self-play with the
rest split equally between rock and counter, hidden 64, four layers, seed 73,
LR 0.0015 with cosine decay, entropy coefficient 0.001. Both use seven inputs;
the control holds input seven at zero, retaining identical parameter count.
Single-seed exploratory comparison, not evidence of repeatability. Checkpoint
evaluations every 763 updates (~50M steps), 64 seeded 1,000-round matches per
opponent. Warm self-play metrics exclude rounds 1–100.

Snapshot for all three jobs: `134fff94964429bd6c11bf1f1e55c4267a030f35`.

| Run | Job | Durable output directory |
|---|---:|---|
| Two-update smoke | 905 | `/cluster/users/tosma/projects/pufferlib-local/jobs/4bc469a8-41d0-43a1-8b31-c28a7963c3b2/artifacts` |
| Round observation on | 906 | `/cluster/users/tosma/projects/pufferlib-local/jobs/693ce272-38fb-4625-9df3-1f7d81a756b7/artifacts` |
| Round observation off | 907 | `/cluster/users/tosma/projects/pufferlib-local/jobs/2cd3d2c5-d5f3-4f5f-b6eb-c7010d2c4146/artifacts` |

Job 905 completed with exit 0, publication and scratch cleanup confirmed;
checkpoint/config/JSON artifact hashes verified against publication index.
Sanitized opponent/clock tests, native training, checkpoint evaluations,
adaptation analysis and CPU/GPU parity passed (maximum probability error
7.36e-8 versus tolerance 1e-5). Tests check both players' clock progression
through a full match, reset, and the constant-zero control. Old six-input
evaluations have explicit dimension support; that compatibility path was not
rerun in this experiment.

Research runs submitted; read Slurm and receipts for live state. Outputs retain
`monitor_history.json`, per-checkpoint trajectories/summaries, final evaluation,
adaptation, parity, resolved configs and weights. Crash-surviving monitor
history also exists one directory above artifacts.

## Completed comparison

Both models reached 199,950,336 actual agent steps. Reported rewards are mean
per-round rewards; warm entropy is conditional action entropy, not entropy
of the aggregate action frequencies. Uniform entropy is 1.098612 nats.

| Metric | Clock on (906) | Clock off (907) |
|---|---:|---:|
| Warm self-play entropy | 0.309434 | 0.202631 |
| Warm self-play mean TV from uniform | 0.540532 | 0.590855 |
| Rock reward | +0.997406 | +0.998547 |
| Counter reward | +0.976922 | +0.986406 |
| Median rock first near-optimal streak start | Round 3 | Round 2 |
| Median counter first near-optimal streak start | Round 8 | Round 15 |
| Counter matches with later tolerance violations | 24/64 | 0/64 |

Adaptation criterion remains three consecutive rounds with maximum probability
error <=0.01 against the optimal one-hot distribution, excluding the opening.
The clock-on policy adapted earlier at the median against counter but was less
stable: sustained near-optimal starts ranged from round 8 to 732, versus rounds
12–28 for the clock-off policy. Neither met the uniform-bot uniformity criterion.

| Checkpoint (~M steps) | Clock-on self-play entropy | Clock-off self-play entropy |
|---|---:|---:|
| 50 | 0.061710 | 0.311225 |
| 100 | 0.032305 | 0.039804 |
| 150 | 0.259928 | 0.012595 |
| 200 | 0.309434 | 0.202631 |

Provisional interpretation: clock-on produced higher final self-play entropy
and lower TV in this seed, but remained far from uniform and had slightly worse
scripted-bot rewards. It did not prevent early low entropy or establish calibrated
confidence. This is a mixed result, not a general improvement or Nash convergence.
Replicate across seeds and resolve control parity before stronger claims.

Validation: 906 completed / application exit 0, final CPU/GPU max probability
error 7.1256e-6 (passes 1e-5 tolerance). 907 finished training and evaluation but
failed final parity: 2.36901e-5 >1e-5; Slurm failed / application exit 1. No
tolerance was relaxed. Treat the clock-off CPU measurements/comparison as
provisional. Both published outputs and cleaned scratch; 16 checkpoint/config/
JSON artifact hashes per run were verified against publication indexes.
No jobs remain active. Full source and command identities are pinned in the
snapshot and per-job `protocol.json` and resolved training INI.

Final checkpoint SHA-256:
- Clock on: `c55445bdc4d47670b95deaf9bf606872a22693d96661f4699eebcb2b3b8f63a5`.
- Clock off: `e6d284dbafd0d7accebbe10abbca9a011e00880c2c405264ce51ab9299372fe3`.
