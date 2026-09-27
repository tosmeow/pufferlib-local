# Separate training against scripted opponents

## Round-observation comparison — 2026-09-27

User requested normalized within-match step count and a shorter training rerun.
Add input 7: completed rounds / 999 (zero at opening, one before final move,
reset to zero at match reset). This exposes history length, not calibrated
confidence or opponent identity. Default `env.round_observation=1`; setting 0
keeps the seventh input zero for a same-architecture ablation. Legacy six-input
checkpoints require explicit OBS_SIZE=6 in eval/parity and cannot initialize
the new seven-input network.

Fresh paired runs: 200M requested agent steps, seed 73, equal rock/counter/
self-play mixture, hidden 64, four layers, LR 0.0015, existing cosine schedule
and entropy coefficient. Evaluate checkpoints every 763 updates (~50M steps)
and final CPU/GPU parity without relaxing tolerance. Compare warm self-play
entropy/TV, bot rewards and adaptation timing. One seed is exploratory.

Submit through Slurm: `python3 scripts/rps_monitor.py 0.3333333333333333
--steps 200000000 --interval 763 --round-observation 1` (or 0 for control).
First run a two-update clock smoke test. Each job requests one GPU, 3 CPUs,
2 GiB RAM, 4 GiB scratch, 10 minutes. The extra 64 encoder parameters add only
256 bytes to weights; prior usage ~0.75 GiB RSS / 0.6 GiB VRAM remains the basis
for resource estimates. Four checkpoint evaluations plus final output are
roughly 250 MiB; no dataset is loaded. Outputs use the existing durable history
and publication workflow. Full previous study remains in SELFPLAY_RESULTS.md.

## Longer weighted self-play comparison — 2026-09-27

User requested longer runs, higher self-play shares, and collapse monitoring.
Plan: fresh seed-73 runs at 2 billion requested agent steps with self-play
shares 1/3, 0.8, and 0.95; split the remaining matches equally between rock
and counter. Hold hidden size 64, four layers, LR 0.0015, entropy coefficient
0.001, and existing cosine LR schedule fixed. The long 1/3 run is a duration
control. These are exploratory single-seed comparisons.

`scripts/rps_monitor.py SHARE` runs sanitized opponent tests, training, checkpoint
evaluations every 3052 updates (~200M agent steps), and final evaluation/parity.
Monitor warm self-play entropy and total-variation distance from uniform,
plus rock/counter rewards. Warnings flag entropy below 0.5 nats, a checkpoint
entropy drop greater than 0.2 nats, or scripted reward below 0.9. These are
diagnostic thresholds, not convergence tests or automatic stopping rules.
Full checkpoint trajectories and summaries are retained. Uniform entropy is
ln(3); high entropy alone does not prove convergence or robustness.

Per run: 1 GPU, 3 CPUs (2 training plus evaluator), 2 GiB RAM, 4 GiB scratch,
15-minute limit. Prior 64-unit training measured ~0.75 GiB peak RSS and
0.6 GiB VRAM; allow evaluator/compiler headroom. No dataset or loaders.
Ten 200-KiB checkpoints and ~0.6 GiB evaluation output fit the bounded budget;
outputs are written to CLUSTER_RESULTS_DIR for publication. A separate normal
`rps_monitor_history.json` below CLUSTER_JOB_DIR provides crash-surviving progress.
First run a two-update smoke test with the same driver before full submissions.

## Protocol — 2026-09-27

Train one fresh policy per bot (uniform, previous-move counter, always rock after
opening), using seed 73 and 200 million requested agent steps each. Keep the
existing float32 6-input, 16-hidden, four-layer minGRU and all other training
settings identical. This is an initial single-seed comparison, not an estimate
of variability across training seeds. No external dataset or held-out data is
used. The environment, input identity, and probability tables are pinned by
each submission's source snapshot.

Current entry point: `bash scripts/rps.sh run BOT_ID [TIMESTEPS] [LEARNING_RATE]` submitted
through `cluster run`. It records checkpoints, resolved training INI/logs,
evaluation trajectories and probabilities, summaries, and checkpoint/source
hashes under `$CLUSTER_RESULTS_DIR`.

Request per job: one GPU, two CPUs, 4 GiB RAM, 4 GiB scratch, ten minutes.
Prior job 858 reported approximately 0.6 GiB host RAM, 0.5 GiB VRAM, and
9.6 million steps/sec. The model has 3,232 float32 parameters (~13 KiB weights);
rollouts contain 1,024 agents x 64 steps. The RAM request adds room for native
compilation and CUDA overhead; scratch covers the source/dependencies/build
and bounded CSV output. There are no data loaders or resident datasets.

Evaluation: CPU PufferNet inference with sampled softmax actions, recurrent
state carried across rounds and reset between matches. The existing CPU/GPU
inference parity procedure is available as `bash scripts/rps.sh parity` and in prior
job 860. Each trained policy plays 64 independent seeded matches of 1,000
rounds against every preset, plus the existing diagnostic opponents. The exact
training header supplies preset probabilities and opening behavior; evaluation
uses a separate RNG from training. Report mean round reward and approximate
95% intervals across match means, alongside conditional action probabilities.
These intervals quantify evaluation sampling, not training-seed uncertainty.

Analytical reference (not experimental conclusions): against uniform play,
every learner policy has expected reward zero. Against always rock, play paper
after the opening. Against the counter, choose `(previous_self + 2) % 3` after
the opening. Both predictable bots permit expected match-average reward 0.999,
because the first of 1,000 rounds is uniform.

## Worklog

- 2026-09-27: Prepared independent training/evaluation wrapper and extended the
  existing probe with opt-in exact preset evaluation. Shell syntax and diff
  checks passed. Training submissions and verified results will be recorded
  below. Next: inspect training completion, evaluate the final policies, and
  compare against the analytical references.

- 2026-09-27: Baseline jobs 885–887 used learning rate 0.015. The counter agent
  saturated on rock and lost almost every round (evaluation mean -0.998828).
  Follow-up job 888 changed only the learning rate to 0.0015, started fresh,
  and obtained +0.998969 against the counter. Submitted matching lower-rate
  uniform/rock comparisons as jobs 889–890. Reduced RAM requests from 4 GiB
  to 2 GiB after the baseline runs used approximately 0.74–0.98 GiB peak RSS.
  These are operational observations; the final comparison and provenance
  belong in the accompanying results record. No seed sweep was performed.

- 2026-09-27: All six runs completed and key output hashes were verified against
  publication receipts. See [BOT_RESULTS.md](BOT_RESULTS.md) for final scores,
  conditional behavior, cross-opponent evaluation, checkpoint links, exact
  snapshots and limitations. No training remains active for this comparison.

## Mixed-opponent follow-up — 2026-09-27

Train one fresh policy with `bot_policy=3`, seed 73, 200 million requested
steps, and learning rate 0.0015. Each match draws one of the three presets
uniformly, without revealing its identity. Preserve recurrent state across
64-step rollout chunks and reset it between 1,000-round matches. Other network
and training settings match the previous comparison.

Request one GPU, two CPUs, 2 GiB RAM, 4 GiB scratch, ten minutes, based on the
previous 0.74–0.98 GiB peak RSS and 0.52 GiB VRAM. No external input dataset.
Evaluate 64 matches against each preset using the existing independent CPU
probe, additionally reporting mean rewards over rounds 1, 1–5, 1–10, 1–50,
and 51–1,000. These are descriptive adaptation measurements, not proof that
the recurrent memory is necessary; no memory ablation or seed sweep is planned
for this initial run.

## Capacity comparison — 2026-09-27

User changed hidden size from 16 to 64. Rerun mixed-opponent training from
scratch, retaining four layers, seed 73, learning rate 0.0015, and 200 million
requested steps. The CPU evaluator now takes architecture dimensions from the
resolved training INI; verify checkpoint size and CPU/GPU probability parity
for the larger network before interpreting its results. Sampling, matches,
evaluation seeds, and early-round reporting remain the same.

Request one GPU, two CPUs, 4 GiB RAM, 4 GiB scratch, ten minutes. The network
has 49,792 float32 parameters (~195 KiB), versus 3,232 previously; hidden
activations grow fourfold and recurrent matrix storage sixteenfold. The
previous ~0.74 GiB host RSS and ~0.52 GiB VRAM leave room for the increased
buffers on one 16-GiB GPU. RAM includes compiler/evaluator overhead; no dataset
is loaded, and source/build/CSV scratch remains bounded as before. This is a
single-seed capacity comparison, not a guarantee of improved optimization.
