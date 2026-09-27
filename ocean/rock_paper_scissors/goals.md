## Steps completed

- Architecture design was a 4x smaller than usual minGRU because of simplicity of the policies we knew towards which we wanted to converge.

- Trained rock paper scissors environment in self-play, which eventually converged to Nash equilibrium: indifferent of recent actions, always plays uniform at random across the 3 different actions.

## Next steps to explore

- No longer self-playing: want to hard-code various agent policies: players with different than the Nash equilibrium behavior but indifferent to our behavior, players that draw their actions based on what we have last done.

- Study then the equilibrium compared to what I'd expect on paper.

## Opponent selection

`rock_paper_scissors.h` defines probability tables and samples the scripted
opponent using the learner's previous move. Two learner slots share policy 0
with separate recurrent states. In scripted matches each plays its own bot;
in self-play they play each other. Each observation contains that learner's
own previous move and its opponent's previous move.

Set `[env] bot_policy` in `config/rock_paper_scissors.ini`, or pass
`--env.bot_policy=N` to `bash scripts/rps.sh train` in a Slurm submission:

- `0`: uniform.
- `1`: counter the learner's previous move.
- `2`: always rock after the opening.
- `3`: choose uniformly among the three scripted bots at each match reset.
- `4`: current-policy self-play.
- `5`: equal rock/counter/self-play mixture (default). Select at match reset
  and keep it fixed for all 1,000 rounds, across training rollout chunks.

Mode 5 now accepts `[env] selfplay_share` (default one-third); the remaining
share is split equally between rock and counter. For longer monitored runs,
submit `python3 scripts/rps_monitor.py SHARE` through Slurm (see the longer-run
protocol in BOT_EXPERIMENT.md). Check `rps_monitor_history.json` in the job
directory during execution; final artifacts retain all evaluated trajectories,
checkpoint hashes, summaries, training logs, and CPU/GPU parity.

Scripted presets open uniformly; self-play uses the policy's sampled opening.
Each slot receives the requested one-third mixture in expectation, so neither
slot trains on an ignored action. Keep `vec.num_policies=1` and
`selfplay.enabled=0` (the latter controls the historical-opponent pool).
Use an even `vec.total_agents`, since each environment now has two slots.
Logs average scores/performance over both learner perspectives; self-play
scores cancel, so use the separate bot evaluations to assess exploitation.
Uniform/rock/counter probability-table evaluations remain available.

Adding a preset in C requires a rebuild; selecting
an existing preset through the config does not. The run script rebuilds before
training and forwards its arguments to the trainer.

## Running and evaluating

One entry point: `bash scripts/rps.sh help`. Submit GPU work through Slurm:

```bash
cluster run --gpu 1 --cpus 2 --memory 2G --scratch 4 --time 00:10:00 -- \
    bash scripts/rps.sh run 5 200000000 0.0015
```

`run` trains, evaluates all bots, measures adaptation, and checks CPU/GPU parity.
`train` accepts the usual trainer flags. `eval CHECKPOINT HIDDEN [LAYERS [SHA256]]`
reevaluates a saved float32 model; `parity` checks inference alone. CPU-only
`analyze summary` and `analyze adaptation` reuse existing outputs. No command
implicitly submits jobs; invoke it inside `cluster run`.

RPS now has four files in `scripts/`: `rps.sh`, `rps_analysis.py`, and the two
native inference helpers `rps_policy_probe.c` / `rps_gpu_probe.cu`. The six old
wrappers/analysis scripts were consolidated. Their exact prior versions remain
in immutable job snapshots (e.g. job 897); restore a snapshot to a new checkout
for historical reproduction. Published experiment outputs were not removed.

Removed entry points: `scripts/rps_run.sh`, `scripts/rps_bot_experiment.sh`,
`scripts/rps_policy_eval.sh`, `scripts/rps_gpu_check.sh`,
`scripts/rps_policy_summary.py`, and `scripts/rps_adaptation.py`. All six were
verified byte-for-byte against job 897's snapshot before removal was finalized.

## Worklog

- 2026-09-27: Jobs 902–904 each reached 1,999,962,112 training steps. Monitoring
  found severe self-play entropy collapse and scripted-bot reward deterioration
  at all three mixture shares; see SELFPLAY_RESULTS.md for checkpoint history
  and provisional final measurements. All three failed their final strict
  CPU/GPU parity assertion (not training); artifacts were published, cleanup
  confirmed, and 28 key hashes per job verified. GPU training logs also showed
  very low final entropy. No runs remain active. Next: diagnose parity on the
  saturated checkpoints, discuss LR-schedule/entropy controls, then replicate
  promising settings across seeds. Do not describe these jobs as fully validated.

- 2026-09-27: At user request, added configurable self-play share and live
  checkpoint monitoring; submitted 2B-step runs at 80%, 95%, and one-third
  self-play (jobs 902–904), snapshot `07b10e9ac68cb1cef2465c9d03ee806f7c69ef38`.
  Weighted selection tests and full monitoring/evaluation/parity smoke passed
  in job 901 after correcting job 900's headless build include. See
  SELFPLAY_RESULTS.md for paths, verification and protocol. Full-run results
  pending at submission; no conclusion about collapse yet. Next: inspect live
  entropy and bot reward histories, verify final receipts/parity, then compare
  duration and mixture effects and discuss seed replication.

- 2026-09-26: Added scripted opponents, single-learner integration, and config
  selection. C++ unit tests with AddressSanitizer and UndefinedBehaviorSanitizer
  passed in Slurm job 861 (exit 0), snapshot
  `77874359602e9dc976d06cade79ef7a7da754c72`. Receipt/log directory:
  `/cluster/users/tosma/projects/pufferlib-local/jobs/37de9b64-8068-4196-8ab9-91b27ac8f7c0/`.
  Publication and scratch cleanup confirmed; no output artifacts were declared
  by the test. The test uses an empty raylib header because this environment
  has no graphics calls. Run-script argument forwarding was added after that
  snapshot and passed `bash -n`; `git diff --check` passed. Full trainer build
  and training: not run. Next: select an opponent, submit training, then compare
  learned behavior with the expected best response.

- 2026-09-27: Trained and evaluated separate policies against all three bots.
  Baseline jobs 885–887 used learning rate 0.015; counter training collapsed.
  Matching follow-up jobs 888–890 used 0.0015 and obtained near-optimal scores
  against both predictable bots, with approximately zero against uniform.
  Each policy received 199,950,336 agent steps, seed 73. All six jobs completed
  with exit 0, verified key artifact hashes, publication, and scratch cleanup.
  See [protocol](BOT_EXPERIMENT.md) and [results/provenance](BOT_RESULTS.md)
  for snapshot commits, durable output paths, confidence intervals, and the
  retained failed-learning baseline. These are single-seed observations, not
  agreed general conclusions. Next: discuss the observed policies and repeat
  with additional training seeds before claiming robust convergence.

- 2026-09-27: Added `bot_policy=3` (now the default): select an opponent on
  match reset, keeping it fixed across rollout chunks. Fixed-bot modes remain
  available. Existing and new seeded-selection/match-stability unit tests
  passed with sanitizers in Slurm job 891, snapshot
  `d3b21f6d390b27a41adf7489ae7eb4430e87914d`, exit 0; publication and scratch
  cleanup confirmed (no artifacts). Receipt/log directory:
  `/cluster/users/tosma/projects/pufferlib-local/jobs/4ce02c84-25ac-4e39-b814-19a670891bc3/`.
  Training in mixed-opponent mode: not run. Next: train one shared recurrent
  policy and measure adaptation during the first rounds of each match.

- 2026-09-27: Reran mixed-opponent training with the user's hidden-size increase
  to 64, keeping four layers, seed 73, learning rate 0.0015, and ~200M steps.
  Job 893, snapshot `c8ba6fc84bd021685670dd4303e13b745078f905`, completed
  with exit 0 and verified publication, key hashes, and cleanup. Updated the
  evaluation probes for configurable architecture and verified CPU/GPU parity
  (max probability error 3.0167e-6). Observed mean rewards: uniform +0.000766,
  counter +0.996422, rock +0.998422; both predictable bots scored +1 on all
  evaluated rounds after round 50. See [results](BOT_RESULTS.md) for artifacts,
  checkpoints, comparison with job 892, and single-seed limitations. Next:
  discuss the improvement, assess seed variability, and consider memory ablation.

- 2026-09-27: Measured the user's three-round / 0.01 probability-error adaptation
  criterion on job 893's verified trajectories. Against rock the first lasting
  qualifying streak began at rounds 2–3 (median 2); against counter at rounds
  4–7 (median 5). All 64 matches per deterministic bot remained within tolerance
  thereafter. Diagnostic job 894, snapshot
  `9a8ce3657012f46179a3c109f6b2a3adcb9492e1`, completed; exit, output hashes,
  publication and cleanup verified. See [results](BOT_RESULTS.md) for exact
  definitions, per-match artifacts, and the uniform-optimality distinction.
  Next: discuss timing, assess variability across seeds, and test memory dependence.

- 2026-09-27: Implemented `bot_policy=5` (default): equal rock/counter/current-
  policy self-play per learner slot, selected once per match. Both slots share
  policy 0 with independent histories/state; scripted modes give each its own
  bot, self-play pairs them. Preserved modes 0–3 and added pure self-play mode 4.
  Unit tests cover all self-play action pairs, mirrored observations, independent
  counter histories and RNG draws, seeded selection/proportions, and both-slot
  terminals/resets. First test job 895 failed to compile a mixed-enum initializer
  in the test harness; corrected to an explicit integer array. Job 896 then
  passed ASan/UBSan tests, native compilation, two training updates, and terminal
  evaluation. Snapshot `3dfa7f1950207a0b62eab7acce5cd71b9647b0f9`; exit 0,
  all output hashes, publication, and scratch cleanup verified. Outputs:
  `/cluster/users/tosma/projects/pufferlib-local/jobs/f7e2a2bf-6a6c-438a-bcc9-35316c3ea037/artifacts`.
  Resources: two CPUs, one GPU, 2 GiB RAM, 4 GiB scratch, three minutes, based
  on prior sub-1-GiB RSS and ~0.6-GiB VRAM. Full research training: not run.
  Next: train mode 5 at the established settings, evaluate the fixed bots, and
  compare adaptation and exploitability with the earlier scripted mixture.

- 2026-09-27: Mode-5 training job 897 and adaptation diagnostic 898 completed,
  snapshot `dcd8509cc960f37aa2f30b9c92c0b72e8d9f6f0f`; exits, key hashes,
  publication, cleanup, and CPU/GPU parity verified. ~200M steps, hidden 64,
  seed 73, LR 0.0015. [Short results and outputs](SELFPLAY_RESULTS.md): near-
  optimal rock/counter reward, but no uniform fallback observed. One-seed
  observation; exploitability not tested. Resources: 2 CPUs/2 GiB/1 GPU/4 GiB
  scratch for training (64 s, peak RSS 784,756 KiB); 1 CPU/512 MiB/no GPU/4 GiB
  scratch for streaming diagnostics (4 s). No jobs remain active. Next: discuss
  the result, measure exploitability, and consider historical self-play opponents.

- 2026-09-27: Consolidated eight RPS workflow files into four: one launcher,
  one Python analysis module, and two native inference helpers. Removed the
  six redundant entry points listed above; exact originals are recoverable
  from job 897's snapshot (also backed up at `/tmp/rps-cleanup-zfypgog6`).
  Existing results and Modal scripts preserved. Job 899, snapshot
  `da28c86d331897f3162b12562d21a9312df6a14d`, passed a two-update training
  smoke run, full saved-checkpoint evaluation, CPU/GPU parity, and byte-exact
  comparisons of summary/adaptation results against jobs 897/898. Exit 0,
  key output hashes, publication, and cleanup verified. Outputs:
  `/cluster/users/tosma/projects/pufferlib-local/jobs/ed4dc470-ac86-41fc-94ae-c8247ce03030/artifacts`.
  Used 2 CPUs, 2 GiB RAM, 1 GPU, 4 GiB scratch, five-minute limit based on prior
  sub-1-GiB RSS and ~0.6-GiB VRAM. Shell syntax, CLI help, and diff checks passed.
  Next: use `scripts/rps.sh` for subsequent training/evaluation; research
  conclusions are unchanged by this cleanup.
