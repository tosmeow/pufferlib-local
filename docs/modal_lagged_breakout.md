# Lagged Breakout on Modal

Edit this checkout on the Mac, launch a remote Linux/T4 job, then download its
policies and logs. The launcher uses the existing PyTorch comparison runner with
`--device cuda` and builds the native extension with `--float` on Modal. Game
simulation and training both run remotely. Your Mac needs neither CUDA nor Docker.

## Local setup

Install the Modal client in its own environment, leaving the training environment
unchanged. From the repository root, using the `uv` already available on this Mac:

```bash
uv venv .venv-modal --python .venv/bin/python
uv pip install --python .venv-modal/bin/python 'modal>=1.5.5,<2'
.venv-modal/bin/modal setup
```

`modal setup` signs in to your Modal account. The following commands launch paid
remote compute using that account. To see the launcher options locally:

```bash
.venv-modal/bin/modal run scripts/modal_lagged_breakout.py --help
```

## First remote check

```bash
.venv-modal/bin/modal run scripts/modal_lagged_breakout.py --smoke
```

The first invocation builds the Linux image and compiles the extension. Subsequent
invocations reuse cached image layers when their inputs have not changed.

The smoke check requests one T4, one CPU core and 4 GiB RAM, with a ten-minute
execution timeout. It performs a 1,024-decision lag-4 training run, executes a
short CPU evaluation, and loads the saved weights onto the CPU to
check that they are finite. Its single evaluation episode is capped at 64
decisions; a capped-episode message is expected and is not a performance result.

Verified on 2026-09-06 in
[`smoke_20260906_095119_287423cb`](../benchmarks/lagged_breakout/smoke_20260906_095119_287423cb/modal_run.json):
the Tesla T4 completed all 1,024 training decisions in about 1.16 seconds.
The recorded Function interval was about 10.1 seconds, excluding image building
and initial container startup. The downloaded checkpoint also loaded on the Mac
CPU with finite weights. This tiny check is not a throughput benchmark.

As checked on 2026-09-06, T4 is Modal's lowest-priced GPU at $0.000164/second
(about $0.59/hour), versus $0.000222/second for L4 (about $0.80/hour).
Ten minutes of T4 GPU time costs about $0.098; adding the requested CPU and RAM
gives about $0.112 at those rates. These are estimates, not a total-spend cap:
image builds, any extra resource use, and storage are separate. The verified
smoke check finished well before the timeout; future startup and execution times
can vary. There is no automatic fallback to a more expensive GPU.
See [current Modal pricing](https://modal.com/pricing).

## Run a comparison

```bash
.venv-modal/bin/modal run --detach scripts/modal_lagged_breakout.py \
  --run-name modal_trial_01 \
  --lags 0,2,4,8 --seeds 42 \
  --timesteps 1000000 \
  --agents 4096 --minibatch-size 16384
```

Lags and seeds are comma-separated launcher arguments. The timestep budget is
**per policy** and rounds up to a complete rollout. This example runs four
policies sequentially on one GPU, then evaluates them on lag 0, volatility 0.
One million decisions is an initial trial, not a convergence guarantee. Training
volatility defaults to 6; other experiment settings come from the current checkout.

Omit `--run-name` to get a unique generated name. An existing output directory
causes an error, including if a previous attempt failed. There are no automatic
retries. `--detach` lets the remote invocation continue if the client disconnects;
the launch output includes a Modal dashboard link and the download command.

Full comparisons request one T4, four CPU cores and 16 GiB RAM, with a six-hour
timeout. Both training and the runner's final CPU evaluations hold this
allocation. For longer experiments, use the timed mode below.

## Four T4s, five minutes per policy

`--seconds` switches to independent jobs: one lag/seed per T4, with up to four
running concurrently. For example, continue the four policies from the initial
comparison already stored on the Modal Volume:

```bash
.venv-modal/bin/modal run --detach scripts/modal_lagged_breakout.py \
  --run-name breakout_t4_5min_01 \
  --source-run modal_20260906_141333_119624d8 \
  --lags 0,2,4,8 --seeds 42 --seconds 300 \
  --agents 32768 --minibatch-size 131072 --threads 4 \
  --learning-rate 0.005
```

Omit `--source-run` to start fresh policies. Each job reads its matching parent
on the Volume; no checkpoint upload is needed. Parent environment settings,
including volatility, and the network architecture are preserved. The requested
agent count, horizon, minibatch size and learning rate replace the parent's
values. The learning rate follows a cosine decay to 10% over this run's time
budget. The example uses 0.005 instead of the initial comparison's 0.1; this is
an explicit training setting, not a throughput optimization.

The timer starts after trainer setup. It includes training, periodic logging
and checkpoint overhead, stops at the end of the update crossing 300 seconds,
then saves the final checkpoint. Startup, image building and final saving add
time. The larger outer Function timeout is a failure backstop, not the training
timer. The four containers may start at slightly different times; each gets its
own full budget. Scheduling depends on T4 availability and account concurrency.

Add `--autotune` to measure several agent/minibatch sizes and CPU thread counts
on each allocated T4 before training. Calibration performs one warmup update and
three measured updates per candidate, records end-to-end steps/second in
`throughput.json`, then discards those temporary policies. It selects the fastest
measured candidate and reloads the original policy before starting the requested
training time. Calibration is additional billable time. For repeated runs, reuse
the selected `--agents`, `--minibatch-size` and `--threads` without `--autotune`.
This is a bounded throughput search, not a guarantee of the global optimum or
100% GPU utilization. Larger batches also change the frequency of policy updates;
equal wall-clock budgets are not equal sample budgets or a convergence guarantee.

Measured on 2026-09-06 with the current float32 PyTorch trainer, horizon 64 and
the same replay ratio, the best sampled throughput at each size was:

| Agents | Minibatch | CPU threads | Training steps/second |
| ---: | ---: | ---: | ---: |
| 4,096 | 16,384 | 1 | 650,000 |
| 8,192 | 32,768 | 1 | 808,000 |
| 16,384 | 65,536 | 4 | 885,000 |
| 32,768 | 131,072 | 4 | 920,000 |

The last configuration sampled 97% GPU utilization. A separate 32-second check
ran all four lags on four distinct T4 UUIDs concurrently, using 65,536 agents and
a 262,144 minibatch. It reached 916,000–926,000 SPS after the first reporting
interval, with 98–99% sampled GPU utilization. Since doubling the batch offered
little additional throughput, the example uses 32,768 agents. The four jobs
finished after 32.8–33.2 seconds; every downloaded checkpoint loaded on the Mac
with finite weights, matching optimizer checkpoint weights and saved CUDA RNG.

Raw results: [batch calibration](../benchmarks/lagged_breakout/t4_timed_calibration_20260906/lag_0/seed_42/throughput.json)
and [four-GPU verification](../benchmarks/lagged_breakout/t4_parallel_check_20260906/lag_0/seed_42/modal_run.json).
These short checks measure throughput, not final policy quality. Historical Mac
logs show about 44,000 SPS for the initial CPU experiment and roughly
270,000–340,000 SPS in the MPS training runs, using smaller batches.

Each job writes its own `lag_N/seed_S/` directory, including `modal_run.json`,
`console.log`, `config.json`, `policy.bin`, `training_state.pt`, `training.jsonl`
and `training_result.json`. The manifest includes the GPU UUID. Results record
new steps, parent steps, total steps and actual training duration. Checkpoints
are saved at the start, approximately every 30 seconds and at completion, with
explicit Volume commits. A handled termination signal finishes the current
update and saves; abrupt termination can lose work since the last commit.

The initial comparison saved only weights, so its first continuation starts a
fresh optimizer. Continuing a timed run also restores its optimizer and saved
random-number-generator states. Environments and recurrent rollout state reset
on each invocation; this is not a bit-for-bit continuation of the simulation.
Use a new `--run-name` for every invocation, including continuations.

The four five-minute T4 allocations cost about **$0.20 for GPUs**, or **$0.30
including the requested four CPU cores and 16 GiB per worker**, at the prices
checked above. Setup, calibration, checkpoint saving, additional resource usage
and storage are extra. GPUs are released after saving; timed jobs do not retain
the GPU allocation for CPU evaluation.

Download and evaluate on the Mac:

```bash
mkdir -p benchmarks/lagged_breakout
.venv-modal/bin/modal volume get breakout-results /breakout_t4_5min_01 \
  benchmarks/lagged_breakout
.venv/bin/python scripts/compare_lagged_breakout.py --evaluate-only \
  --output benchmarks/lagged_breakout/breakout_t4_5min_01 \
  --eval-episodes 32 --max-eval-steps 120000
```

Evaluation uses the same common lag-0, volatility-0 game as the initial comparison.

## Continue the best local policies

The source bundle `local_best_seed42_20260906_153321` is uploaded to
`breakout-results`. It contains the selected `policy.bin` and matching config for
each of the four seed-42 policies. Selection uses the highest recorded audited
return at each policy's training lag among its saved local experiment versions:

| Lag | Local experiment | Audited mean return |
| ---: | --- | ---: |
| 0 | `experiment_04_refinement` | 864.00 |
| 2 | `experiment_04_refinement` | 864.00 |
| 4 | `experiment_03_gpu` | 863.97 |
| 8 | `experiment_04_refinement` | 860.52 |

The [selection manifest](../benchmarks/lagged_breakout/local_best_seed42_20260906_153321/selection.json)
records original paths and checkpoint hashes, verified against the audit reports.
These are the selected best weights; three of their adjacent optimizer files
contain different, later weights. The bundle therefore imports weights only,
and each continuation starts fresh optimizer momentum. Original local artifacts
remain in their existing directories.

Launched on 2026-09-06 as `local_best_t4_5min_01`:

```bash
.venv-modal/bin/modal run --detach scripts/modal_lagged_breakout.py \
  --run-name local_best_t4_5min_01 \
  --source-run local_best_seed42_20260906_153321 \
  --lags 0,2,4,8 --seeds 42 --seconds 300 \
  --agents 4096 --minibatch-size 16384 --horizon 64 --threads 4 \
  --learning-rate 0.0001
```

This preserves the normalized velocity encoder and uses the smaller batch sizes
from the successful local training. The learning rate decays from 0.0001 to
0.00001 over the five-minute window. For another launch, change `--run-name`;
reusing the same source starts from the same best local weights. Using a completed
timed run as `--source-run` instead continues its final weights and optimizer.

Starting and final checkpoints are evaluated on the same test seeds: 256 episodes
at their own lag/volatility using seeds 600000–600255, and 64 common-game episodes
using seeds 100000–100063. This measures whether the continuation improves on its
actual starting policies.

## Download and watch

Each launch prints its exact download command. For the named example:

```bash
mkdir -p benchmarks/lagged_breakout
.venv-modal/bin/modal volume get breakout-results /modal_trial_01 \
  benchmarks/lagged_breakout

.venv/bin/python scripts/watch_lagged_breakout.py \
  --experiment benchmarks/lagged_breakout/modal_trial_01 \
  --policy-seed 42 --game-seed 100000
```

The viewer expects all four lags (0, 2, 4, 8) and a locally built lagged-Breakout
extension. If needed, rebuild it with `bash scripts/build_macos_cpu.sh lagged_breakout`.
A smoke run contains only lag 4 and is not a four-panel viewer experiment.
Checkpoints load on the CPU regardless of the training device. Seeds do not
guarantee identical Linux/macOS trajectories; the environment uses platform-specific
`rand_r` serve randomness.

Pass the existing benchmark **parent** directory to `modal volume get`: the
client creates `modal_trial_01` inside it. With Modal 1.5, a nonexistent local
destination can cause an `Is a directory` error during recursive downloads.

On the Volume, each run contains:

- `modal_run.json`: command, status, GPU, package versions and uploaded source hashes.
- `console.log`: combined experiment stdout/stderr; also streamed to Modal's logs.
- `lag_N/seed_S/`: the runner's `config.json`, `policy.bin`, `training.jsonl` and
  `training_result.json`.
- `evaluations/`: common-game results and CSVs from the existing runner.

Modal persists Volume writes in the background, and the launcher explicitly
commits before training and when the Function exits normally or raises a handled
exception. A hard interruption may leave `modal_run.json` marked `running`;
check the Modal dashboard for the authoritative invocation status.

Without `--seconds`, this launcher starts fresh sequential comparisons and saves
weights only after each policy finishes. Timed mode provides periodic optimizer
checkpoints and continuation from a parent run already on the Volume.

## What is uploaded and built

`SOURCE_PATTERNS` in `scripts/modal_lagged_breakout.py` explicitly selects the
Python trainer, CUDA/native source, Breakout headers, configuration and launcher
files. Local edits are included even when uncommitted. Benchmarks, credentials,
virtual environments, `.git`, local binaries and rendering assets are excluded.
Rendering remains local. Changing included files requires another invocation;
edits do not affect a running job.

The image uses Python 3.12, the NVIDIA CUDA 12.8.1 development image, PyTorch
2.9.1 with CUDA 12.8 wheels, and this checkout's `pyproject.toml` dependencies.
The project is installed from the uploaded checkout, not a released PufferLib
package. Other Python dependency versions are recorded per run rather than fully
locked.

`scripts/build_modal_lagged_breakout.sh` supplies linker names for the CUDA wheel
libraries and builds for T4's `sm_75` architecture. It uses the CUDA NVML stub only
at link time, so the image can build without a GPU; runtime uses Modal's actual
driver. Changing the GPU model also requires selecting a compatible build target.
T4 is suitable for this float32 workflow; the default BF16 native trainer is a
separate workflow requiring a GPU with appropriate BF16 support. A lower hourly
rate does not guarantee a lower total training cost; benchmark throughput before
choosing hardware for long runs.

References: [Modal images](https://modal.com/docs/guide/images),
[CUDA setup](https://modal.com/docs/examples/install_cuda),
[Volumes](https://modal.com/docs/guide/volumes),
[run CLI](https://modal.com/docs/cli/latest/run).
