# RPS bot training results — 2026-09-27

Operational measurements from six completed runs; interpretations below are provisional, not user-agreed scientific conclusions. See [protocol](BOT_EXPERIMENT.md).

Each run trained a separate fresh policy, seed 73, for 199,950,336 actual agent steps (200 million requested, rounded down to complete rollout batches). The primary comparison uses learning rate 0.0015. Evaluation uses 64 independently seeded 1,000-round matches per opponent with sampled actions and recurrent state. Reward is +1 for a win, 0 for a tie, -1 for a loss.

## Matching-settings comparison

| Trained against | Slurm job | Mean round reward against its training bot | Approximate 95% interval |
|---|---:|---:|---|
| Uniform | 889 | -0.002422 | [-0.008944, +0.004100] |
| Counter previous move | 888 | +0.998969 | [+0.998784, +0.999154] |
| Always rock after opening | 890 | +0.998875 | [+0.998677, +0.999073] |

## Observed behavior

- Uniform-trained agent: mean action probabilities against uniform play were
  28.29% rock, 38.52% paper, and 33.19% scissors. Its reward interval includes
  zero. Training against a fixed uniform bot does not require the learner to
  become uniform: every strategy has expected reward zero against that bot.
- Counter-trained agent: on evaluated trajectories it chose scissors after
  its own rock, paper after scissors, and rock after paper, with probabilities
  extremely close to one. This produces a repeating winning cycle. Marginal
  action frequencies near one-third do not imply random or Nash play; mean
  action entropy here was only 1.30e-7 nats per round.
- Rock-trained agent: paper probability was numerically 1.0 on evaluated
  trajectories. It exploited the predictable opponent as expected.

The two predictable bots have theoretical optimal expected reward 0.999 over
1,000 rounds, since their opening move is uniform. The observed rewards are
consistent with that value. This describes these runs, not convergence across
all seeds or behavior in every unvisited state.

## Cross-opponent mean round rewards

| Training opponent | Evaluation: uniform | Evaluation: counter | Evaluation: rock |
|---|---:|---:|---:|
| Uniform | -0.002422 | -0.009062 | +0.049016 |
| Counter previous move | +0.000078 | +0.998969 | +0.000078 |
| Always rock after opening | +0.000625 | -0.999141 | +0.998875 |

## Baseline at learning rate 0.015

| Training opponent | Slurm job | Mean round reward against its training bot |
|---|---:|---:|
| Uniform | 885 | -0.002375 |
| Counter previous move | 886 | -0.998828 |
| Always rock after opening | 887 | +0.998875 |

The higher-rate counter run collapsed to rock. The lower-rate run succeeded; this one-seed comparison supports trying the lower rate, but does not establish its reliability across seeds. No runs were discarded.

## Provenance and verification

All six jobs reached Slurm COMPLETED / exit 0. Publication and scratch cleanup were confirmed. Final checkpoint, summary, resolved INI, and provenance hashes were checked against each publication receipt. Full trajectories remain in shared storage. Each artifact directory also contains intermediate checkpoints, resolved training settings, and hardware/compiler/source identities.

Jobs 885–889 took 33–41 seconds each including build and evaluation; observed
peak RSS was approximately 0.74–0.98 GiB. Training dashboards reported about
0.52 GiB VRAM. No remaining jobs or execution blockers for this comparison.

### Job 885 — Uniform, learning rate 0.015

- Snapshot: `f6837552f1becaa05cee44cace995a77c5fdb184`.
- [Artifacts](/cluster/users/tosma/projects/pufferlib-local/jobs/a07288b0-c53e-4fe9-b361-50e995ec8702/artifacts)
- [Evaluation summary](/cluster/users/tosma/projects/pufferlib-local/jobs/a07288b0-c53e-4fe9-b361-50e995ec8702/artifacts/evaluation/summary.json)
- [Final checkpoint](/cluster/users/tosma/projects/pufferlib-local/jobs/a07288b0-c53e-4fe9-b361-50e995ec8702/artifacts/checkpoints/rock_paper_scissors/bot_0_seed_73/0000000199950336.bin)
- Checkpoint SHA-256: `fff2eefeda197cc111274bff848ceca66138434be4775fe683f64e97363b9f12`.

### Job 886 — Counter previous move, learning rate 0.015

- Snapshot: `2dd4641887622173747a31f21a593d4bd1e857fd`.
- [Artifacts](/cluster/users/tosma/projects/pufferlib-local/jobs/6a51be62-079b-4f1e-bc7e-adc3c5eec9bf/artifacts)
- [Evaluation summary](/cluster/users/tosma/projects/pufferlib-local/jobs/6a51be62-079b-4f1e-bc7e-adc3c5eec9bf/artifacts/evaluation/summary.json)
- [Final checkpoint](/cluster/users/tosma/projects/pufferlib-local/jobs/6a51be62-079b-4f1e-bc7e-adc3c5eec9bf/artifacts/checkpoints/rock_paper_scissors/bot_1_seed_73/0000000199950336.bin)
- Checkpoint SHA-256: `347791af25e0eeb267778b661c4c7aa8757315e7ff71ab5f5649c8106cd12a75`.

### Job 887 — Always rock after opening, learning rate 0.015

- Snapshot: `2dd4641887622173747a31f21a593d4bd1e857fd`.
- [Artifacts](/cluster/users/tosma/projects/pufferlib-local/jobs/0ba5428e-3d43-4390-be82-507090f9d7a8/artifacts)
- [Evaluation summary](/cluster/users/tosma/projects/pufferlib-local/jobs/0ba5428e-3d43-4390-be82-507090f9d7a8/artifacts/evaluation/summary.json)
- [Final checkpoint](/cluster/users/tosma/projects/pufferlib-local/jobs/0ba5428e-3d43-4390-be82-507090f9d7a8/artifacts/checkpoints/rock_paper_scissors/bot_2_seed_73/0000000199950336.bin)
- Checkpoint SHA-256: `bb197b2e4331b4466921a8727a2a90234185ea6addec23a37acaee7e0d137afc`.

### Job 888 — Counter previous move, learning rate 0.0015

- Snapshot: `6378bb069c4cc54bb4e64a21321ec9308b219812`.
- [Artifacts](/cluster/users/tosma/projects/pufferlib-local/jobs/c2c2a3fd-9f3d-4645-8257-c6d42c438541/artifacts)
- [Evaluation summary](/cluster/users/tosma/projects/pufferlib-local/jobs/c2c2a3fd-9f3d-4645-8257-c6d42c438541/artifacts/evaluation/summary.json)
- [Final checkpoint](/cluster/users/tosma/projects/pufferlib-local/jobs/c2c2a3fd-9f3d-4645-8257-c6d42c438541/artifacts/checkpoints/rock_paper_scissors/bot_1_seed_73/0000000199950336.bin)
- Checkpoint SHA-256: `11cf47fedd920ce3ada9d94f8e02c3ffeeb280e0454ebf5a0bd05ae5a3d02254`.

### Job 889 — Uniform, learning rate 0.0015

- Snapshot: `6378bb069c4cc54bb4e64a21321ec9308b219812`.
- [Artifacts](/cluster/users/tosma/projects/pufferlib-local/jobs/503e6996-c7b2-4465-bdaf-ef080edb3440/artifacts)
- [Evaluation summary](/cluster/users/tosma/projects/pufferlib-local/jobs/503e6996-c7b2-4465-bdaf-ef080edb3440/artifacts/evaluation/summary.json)
- [Final checkpoint](/cluster/users/tosma/projects/pufferlib-local/jobs/503e6996-c7b2-4465-bdaf-ef080edb3440/artifacts/checkpoints/rock_paper_scissors/bot_0_seed_73/0000000199950336.bin)
- Checkpoint SHA-256: `ec8372a532c0e19f74edb32533bb49518d67d7e08ce9874fff1d6216718c8ab2`.

### Job 890 — Always rock after opening, learning rate 0.0015

- Snapshot: `6378bb069c4cc54bb4e64a21321ec9308b219812`.
- [Artifacts](/cluster/users/tosma/projects/pufferlib-local/jobs/1821276a-86a8-4797-b5a5-f3905e5d95c0/artifacts)
- [Evaluation summary](/cluster/users/tosma/projects/pufferlib-local/jobs/1821276a-86a8-4797-b5a5-f3905e5d95c0/artifacts/evaluation/summary.json)
- [Final checkpoint](/cluster/users/tosma/projects/pufferlib-local/jobs/1821276a-86a8-4797-b5a5-f3905e5d95c0/artifacts/checkpoints/rock_paper_scissors/bot_2_seed_73/0000000199950336.bin)
- Checkpoint SHA-256: `b7428d2b858d04687b187ff02f9fabccdb7d63d1ab7cb160939169c187554308`.

## Limitations and next steps

Intervals describe evaluation variability across 64 matches, not uncertainty across training seeds. The lower learning rate was selected after observing baseline failure; this is an exploratory result. The probe uses the repository CPU inference implementation, previously checked against GPU inference in job 860; parity was not rerun for these six weights. No held-out benchmark was accessed.

Next: discuss the observed strategies; repeat the matching-settings comparison across independent training seeds before claiming robust convergence.

## Mixed-opponent capacity comparison — 2026-09-27

Job 893 trained one fresh mixed-opponent policy with hidden size 64, four
layers, seed 73, learning rate 0.0015, and 199,950,336 completed steps. Each
match draws a hidden opponent uniformly and keeps it fixed for 1,000 rounds.
Comparison is with the earlier 16-unit mixed policy, job 892, using identical
training budget and evaluation seeds. Evaluation: 64 matches per preset.

| Opponent | Hidden 16 mean round reward | Hidden 64 mean round reward | Hidden 64 approximate 95% interval |
|---|---:|---:|---|
| Uniform | +0.000734 | +0.000766 | [-0.004454, +0.005985] |
| Counter | +0.498813 | +0.996422 | [+0.996254, +0.996590] |
| Rock | +0.998516 | +0.998422 | [+0.998292, +0.998551] |

| Opponent | Rounds 1–5 | Rounds 1–10 | Rounds 1–50 | Rounds 51–1,000 |
|---|---:|---:|---:|---:|
| Uniform | -0.025000 | -0.029688 | +0.012500 | +0.000148 |
| Counter | +0.290625 | +0.642188 | +0.928438 | +1.000000 |
| Rock | +0.684375 | +0.842188 | +0.968438 | +1.000000 |

Both predictable bots were beaten on every evaluated round after round 50
(60,800 rounds per bot). This is an observed improvement in one seed, not proof
that 16 units are incapable or that 64 units converge reliably across seeds.
No memory ablation was performed. Training stopped at its fixed budget; final
two binned mixed training scores were approximately 665 and 665 per match.

The CPU evaluation probe now takes dimensions from the resolved training INI
and checks checkpoint size. CPU/GPU probability parity passed for this model:
10,000 rows, max error 3.0167e-6, tolerance 1e-5. The GPU probe now clears the
full configured recurrent state. Shell syntax and diff checks passed.

- Slurm job **893**: COMPLETED, application exit 0, Slurm exit 0:0.
- Snapshot: `c8ba6fc84bd021685670dd4303e13b745078f905`.
- [Artifacts](/cluster/users/tosma/projects/pufferlib-local/jobs/61b5e806-d4b7-4b00-a4e7-ab4e6b2b271a/artifacts)
- [Evaluation summary](/cluster/users/tosma/projects/pufferlib-local/jobs/61b5e806-d4b7-4b00-a4e7-ab4e6b2b271a/artifacts/evaluation/summary.json)
- [Checkpoint](/cluster/users/tosma/projects/pufferlib-local/jobs/61b5e806-d4b7-4b00-a4e7-ab4e6b2b271a/artifacts/checkpoints/rock_paper_scissors/bot_3_seed_73/0000000199950336.bin)
- Checkpoint SHA-256: `61b3105ae83f9c89b24f67fcff8dd2f5ed9f3ed3463c6f748c6507388b42b2dd`.
- Final checkpoint, summary, parity report, resolved INI, and provenance hashes
  verified against publication receipt; publication and scratch cleanup confirmed.
- Elapsed 56 seconds; peak RSS 785,360 KiB; training VRAM about 0.6 GiB.
- [16-unit comparison artifacts, job 892](/cluster/users/tosma/projects/pufferlib-local/jobs/5cd5dd29-f145-435f-b38e-dccf664fe3b7/artifacts)

Next: discuss this single-seed comparison, assess seed variability, and consider
a memory ablation. No training remains active for this experiment.

## Time to near-optimal probabilities — 2026-09-27

User-defined criterion: three consecutive rounds with maximum absolute error
across the three action probabilities at most 0.01 relative to the optimal
one-hot distribution. For rock, the target is paper; for counter it is
`(previous_self + 2) % 3`. Round numbers are one-based. The uniform opening is
excluded because it has no uniquely optimal action. This analyzes the existing
64-unit policy trajectories from job 893; no training or new matches were run.

| Opponent | First qualifying streak begins | Median start | Third round confirms the streak | Later failures |
|---|---|---:|---|---:|
| Rock | Round 2: 42 matches; round 3: 22 | 2 | Rounds 4–5 (median 4) | 0/64 |
| Counter | Round 4: 28; round 5: 35; round 7: 1 | 5 | Rounds 6–9 (median 7) | 0/64 |

Every deterministic-bot match stayed within tolerance through round 1,000
after its first qualifying streak began. Thus median adaptation began after
one observed round against rock and four observed rounds against counter.
The streak criterion is retrospective: confirmation requires two more rounds.
This describes these 64 seeded trajectories, not every possible history or
explicit posterior identification by the network.

Closeness to `(1/3, 1/3, 1/3)` was measured separately against uniform: zero of
64 matches met the three-round criterion. That is not failure of optimality;
every learner action distribution has expected reward zero against uniform.

- Analysis job **894**, snapshot `9a8ce3657012f46179a3c109f6b2a3adcb9492e1`.
- Slurm COMPLETED / exit 0:0; application exit 0; publication and scratch cleanup confirmed.
- [Summary and per-match timings](/cluster/users/tosma/projects/pufferlib-local/jobs/c2e1f0f9-8430-4f41-8ca2-e70bf27024c8/artifacts)
- Source: job 893 `evaluation/trajectories.csv`, SHA-256
  `54d199b89abbca07d8620a84f2f00c086cc51296adccfe0e040bcc473e7579c1` verified before analysis.
- Both diagnostic output hashes verified against publication receipt. Python
  syntax checked; row counts, probability validity, and match lengths asserted.
- Resources: one CPU, 512 MiB RAM, no GPU, 4 GiB scratch, two-minute limit;
  49 MB source streamed, only 192,000 booleans retained plus Python overhead.
