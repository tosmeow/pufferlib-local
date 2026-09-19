# Lagged Breakout

A native PufferLib C environment for training at different execution delays and
comparing the resulting policies on a common Breakout game. It reuses
`ocean/breakout/breakout.h` for the game rules, collision geometry, scoring,
serves, and renderer. Actions are `0 = NOOP`, `1 = LEFT`, `2 = RIGHT`.

## Time, delay, diffusion, and rewards

One **microstep** is one original Breakout physics frame, `dt = 1/60` second.
Each policy decision advances `frameskip` microsteps. Changing the lag does not
change the decision frequency, physical timestep, or discount factor.

| Setting | Meaning | Default |
| --- | --- | --- |
| `frameskip` | Microsteps per policy decision, from 1 to 64 | 4 |
| `action_lag` | Microsteps before a command first affects the paddle, from 0 to 64 | 0 |
| `volatility` | Ball position diffusion, in pixels / sqrt(second) | 0 |
| `reward_interval` | Microsteps between reward payouts; a positive multiple of `frameskip` | 4 |
| `seed` | Environment seed; vector slot `i` uses `seed + i` | 0 |

The policy command is held for the entire decision interval and this command
stream passes through a FIFO delay of `action_lag` microsteps. A command sent at
microstep `k` first applies during microstep `k + action_lag`. Before the first
command arrives, the paddle receives NOOP. Commands already in flight execute
in order; a later command does not overwrite an earlier one. For example, with
`frameskip=4` and `action_lag=3`, the first RIGHT command produces
`NOOP, NOOP, NOOP, RIGHT`; a subsequent LEFT produces
`RIGHT, RIGHT, RIGHT, LEFT`. A delay may span multiple policy decisions.
The physical delay is `tau = action_lag / 60` seconds.

Between collisions, the ball follows the discrete process

```text
position[k+1] = position[k] + drift[k] * dt + volatility * sqrt(dt) * Z[k]
Z[k] ~ N(0, I_2), independently across microsteps and environments.
```

The stored ball velocities use the original game's pixels-per-microstep units.
Noise perturbs the displacement for one microstep; it is never accumulated into
the velocity. Thus the mean motion stays on the current straight line, and the
per-coordinate position variance grows as `volatility^2 * elapsed_seconds` away
from collisions. The standard deviation accumulated over a lag is
`volatility * sqrt(action_lag / 60)`. For `volatility=6` and lag 4 this is about
1.55 pixels per coordinate.

The sampled displacement goes through the existing swept collision solver.
Walls and bricks reflect the drift on the collision axes; paddle hits choose a
new drift using the original bounce rule. This is a discrete noisy billiard
model, not an exact sampler of reflected Brownian motion at collision surfaces.
The inherited solver resolves one collision per microstep, so small volatility
relative to the game geometry is the intended regime.

Brick collisions earn the original points. Points accumulate and are paid at
the configured reward timestamps, with zero reward between payouts. The default
returns the points earned in each four-microstep decision interval. A larger
`reward_interval`, such as 12, pays every third decision when `frameskip=4`.
This schedules **delivery of accumulated brick rewards**; it does not introduce
the paper's periodic shots or evaluate a new reward landscape at those times.

An episode ending between payout times flushes all unpaid points. Pending
commands are then cleared, and the returned observation is the new episode's
reset state. The current decision stops at termination. This avoids the original
Breakout wrapper's behavior of continuing the remaining frames into the next
episode. Apart from that terminal-step bookkeeping, lag 0 and volatility 0
recover the original physics and rewards when `reward_interval=frameskip`.
At frameskip 1, parity also holds across complete episode boundaries.
Losing a single ball keeps pending commands; only a full episode reset clears
them. The original serve frame ignores discrete paddle movement.

## Observations and reproducibility

All lag values have the same 190-float observation layout and the same three
actions. Checkpoints therefore load without changing the network architecture
when evaluation lag or volatility changes.

| Indices | Contents |
| --- | --- |
| 0–117 | Original Breakout observations, in the original order and units |
| 118 | `action_lag / 64` |
| 119 | `volatility / 100` |
| 120 | `frameskip / 64` |
| 121 | Paddle-hit counter modulo four, divided by four |
| 122 | Current ball speed / maximum ball speed |
| 123 | Last applied command: LEFT = -1, NOOP = 0, RIGHT = 1 |
| 124 | Reward-clock phase, `reward_tick / reward_interval` |
| 125 | Unpaid reward / 864 |
| 126–189 | Next 64 pending microstep commands, earliest first; unused entries are zero |

Exposing pending commands makes the actuator state observable. The hit counter
and ball speed expose the original game's speed-increase state. The queue has
a fixed capacity; changing that capacity changes the checkpoint format.

Serve randomness retains Breakout's `rand_r` stream. Diffusion has a separate
per-environment SplitMix64/Box–Muller stream, so changing volatility does not
consume additional serve draws. Zero volatility consumes no noise draws.
Recreating a vector with identical settings and seeds reproduces trajectories
on the same platform. `reset()` starts a new episode and continues the random
streams; recreate the vector to replay seeds from the beginning. Serve draws
use the platform's `rand_r`, so bitwise cross-platform reproducibility is not
promised. Set `--env.seed` to control environment randomness independently of
the trainer's seed.

## Build and train

From the repository root on this Mac:

```bash
bash scripts/build_macos_cpu.sh lagged_breakout
.venv/bin/python -m pufferlib.pufferl train lagged_breakout \
  --env.action-lag 4 --env.volatility 6 \
  --vec.total-agents 256 --train.minibatch-size 8192 \
  --train.total-timesteps 1000000 --torch.device cpu
```

Use `--torch.device mps` for Apple GPU training. On a CUDA machine, build with
`bash build.sh lagged_breakout` for the native trainer, or add `--float` for the
PyTorch backend. `puffer train lagged_breakout` works with the installed CLI.
As in the rest of this checkout, a build selects the single active native
environment in `pufferlib._C`.

For the standalone keyboard demo:

```bash
CC='/opt/homebrew/opt/llvm/bin/clang -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk' \
  bash build.sh lagged_breakout --local
./lagged_breakout 4 6
```

Use the arrow keys or A/D. The window title shows lag and volatility.

## Train at several lags, compare on the original game

The experiment runner uses PufferLib's PyTorch trainer with the native C vector
environment. It needs a `--cpu` or `--float` build, and initializes a fresh policy
for each `(lag, seed)` pair. Training settings, volatility, reward timing, and
initial seeds are held fixed across lag levels.

```bash
.venv/bin/python scripts/compare_lagged_breakout.py \
  --output benchmarks/lagged_breakout/experiment_01 \
  --lags 0 2 4 8 --seeds 42 43 44 --volatility 6 \
  --timesteps 1000000 --device cpu --eval-episodes 32
```

One million steps is a starting budget, not a convergence guarantee. Increase it
and inspect training curves before interpreting behavior. Default training
hyperparameters come from the original Breakout configuration; they have not
been retuned for diffusion or lag.

By default, **every saved policy is evaluated with lag 0, volatility 0, and the
original reward timing**, using the same environment seeds and deterministic
argmax actions. This uses the original physics inside `lagged_breakout` while
keeping the observation layout compatible with all trained agents. A checkpoint
from the original 118-input `breakout` policy is not directly compatible.

Outputs include:

- `lag_N/seed_S/config.json`, `policy.bin`, `training.jsonl`, and `training_result.json`.
- A separate `evaluations/evaluation_TIMESTAMP/results.json` with settings, per-policy means,
  standard errors across evaluation episodes, and raw episode results.
- `episodes.csv` with returns, lengths, command-switch rates, immediate LEFT/RIGHT
  reversal rates, NOOP fractions, and paddle travel measured at decision times.
- `trajectories.csv` with the first evaluation seed's paddle/ball positions and
  selected actions, up to `--trace-steps` decisions, for plotting behavior.

Evaluation takes the **first episode from each vector slot**, so fast-finishing
policies are not overrepresented. Episodes reaching `--max-eval-steps` are marked
as capped and excluded from completed-episode summaries; increase the cap before
comparing returns if censoring occurs. Paddle travel excludes observed ball/reset
teleports and is a decision-time measurement; the native `paddle_distance` log
measures movement per microstep. Standard errors within one trained policy do
not measure training uncertainty: compare results across training seeds too.

To evaluate the saved agents on a second common game, without training again:

```bash
.venv/bin/python scripts/compare_lagged_breakout.py \
  --output benchmarks/lagged_breakout/experiment_01 --evaluate-only \
  --eval-lag 0 --eval-volatility 6 --eval-episodes 64
```

Each evaluation gets a new output directory. To watch a PyTorch checkpoint:

```bash
.venv/bin/python -m pufferlib.pufferl eval lagged_breakout --slowly \
  --load-model-path benchmarks/lagged_breakout/experiment_01/lag_4/seed_42/policy.bin \
  --env.action-lag 0 --env.volatility 0 --torch.device cpu
```

## Continue existing policies

The [benchmark index](../../benchmarks/lagged_breakout/README.md) links the final
comparison, original baseline, and archived calibration trials.

To watch four trained policies play the original game together in a native
desktop window:

```bash
.venv/bin/python scripts/watch_lagged_breakout.py \
  --experiment benchmarks/lagged_breakout/experiment_03_gpu \
  --policy-seed 42 --game-seed 100000
```

The 2×2 view shows training lags 0, 2, 4, and 8, all evaluated with action lag 0
and volatility 0. Each game has its own native PufferLib vector initialized with
the **same environment seed**. All four advance on the same simulation clock;
their actions can then produce different trajectories. Replaying recreates the
vectors and clears recurrent policy state, reproducing the same random streams.
Finished games wait for the others before an optional synchronized next seed.

Use the shared pause, step, replay, next-seed, and speed controls. Each panel's
policy-seed selector loads the corresponding checkpoint and restarts all games
together. Space pauses, R replays, and N advances the shared seed. This viewer
uses Tkinter and the actual C environment, and requires no browser or web server.
For a non-graphical smoke check, add `--headless-steps 250`.

For a longer run from the saved lag/seed weights:

```bash
.venv/bin/python scripts/continue_lagged_breakout.py \
  --source benchmarks/lagged_breakout/experiment_01 \
  --output benchmarks/lagged_breakout/archive/experiment_02_continued \
  --additional-steps 100000000 --eval-interval 5000000 --workers 4
```

The source experiment is preserved. This runner validates each policy at its own
training lag and volatility, saves the best validation checkpoint as `policy.bin`,
and writes live progress to each run's `status.json`. `latest.bin` is the latest
evaluated policy, while `training_state.pt` contains the latest policy, optimizer,
training counters, and Python/NumPy/Torch RNG states. Repeating the command resumes
these saved training states. Native environment states are not serialized, so a
resume starts new episodes and is not a bitwise continuation of the rollout.
`--additional-steps` is the cumulative continuation-step limit relative to the
source checkpoint: resuming with 200 million after a 100-million-step run adds
up to another 100 million steps.
Use `--continue-after-target` to train up to that cumulative step limit even
when a saved policy already passed the empirical target. The same flag prevents
early stopping when the target is reached during the new training phase.
The original comparison runner saved weights only; its first continuation must
start with fresh optimizer momentum.

The default learning-rate schedule runs over the additional-step budget and
retains a 10% learning-rate floor. Four independent CPU jobs run concurrently with
one Torch/OpenMP thread each. The default near-ceiling criterion is a mean return
of at least 850 out of 864 on three consecutive validation checks. This is an
empirical target, not a proof of optimality. Exhausting the budget below that
target is reported explicitly. Validation uses environment seeds starting at
200000; final evaluation at the training lag uses separate seeds starting at
300000, and common-original evaluation uses seeds starting at 100000. These final
reports are `holdout_training_environment.json` and
`holdout_original_environment.json` in each run directory.

Periodic evaluation snapshots go in each run's `validation/` directory. Previous
configurations and checkpoint backups go in `history/`. Each continuation
invocation records its manifest and partial results under the experiment's
`history/phases/`, with a timestamp to preserve earlier invocations. Selected
policies, resumable states, and final reports keep their existing run paths.

The longer calibration on this Mac found that the original small-batch learning
rate was unstable. A larger Apple-GPU batch and a lower rate reached near-maximum
returns. The initial GPU phase used:

```bash
.venv/bin/python scripts/continue_lagged_breakout.py \
  --source benchmarks/lagged_breakout/experiment_01 \
  --output benchmarks/lagged_breakout/experiment_03_gpu \
  --additional-steps 100000000 --anneal-steps 50000000 \
  --eval-interval 5000000 --learning-rate 0.02 \
  --workers 1 --device mps --agents 4096 --minibatch-size 65536
```

`--anneal-steps` keeps the learning-rate decay separate from the run's step
limit: the rate reaches its 10% floor after that many steps and stays there.
Completed policies with passing validation and fresh-seed scores are skipped
when resuming the queue. Earlier configurations are archived before each resume.
SIGTERM/SIGINT requests a checkpoint and pause after the current update. Apple
GPU jobs need direct GPU access; a sandbox can report MPS unavailable even on an
otherwise supported Mac.

For unfinished policies, the follow-up phase uses `--normalize-velocity`, a
16384-sample minibatch, learning rate 0.005, and a 200-million-step limit. This
selects `LaggedBreakoutEncoder`, which multiplies observation entries 4 and 5 by
60 before the encoder, giving physical velocity / 512 rather than the inherited
per-frame displacement / 512. Existing encoder weights in those two columns are
divided by 60, preserving the policy's forward function at conversion. Optimizer
momentum is restarted in the new parameter coordinates. The observations, state
dynamics, and rewards are unchanged. Converted checkpoints record their encoder
in `config.json`; always load that configuration when evaluating them. Checkpoints
that already passed the validation and fresh-seed targets are preserved.

```bash
.venv/bin/python scripts/continue_lagged_breakout.py \
  --source benchmarks/lagged_breakout/experiment_01 \
  --output benchmarks/lagged_breakout/experiment_03_gpu \
  --additional-steps 200000000 --anneal-steps 50000000 \
  --eval-interval 5000000 --learning-rate 0.005 \
  --workers 1 --device mps --agents 4096 --minibatch-size 16384 \
  --normalize-velocity --eval-episodes 32 --holdout-episodes 64
```

The comparison script reads each checkpoint's saved encoder automatically.
For the rendered CLI evaluation of a converted checkpoint, specify the encoder:

```bash
.venv/bin/python -m pufferlib.pufferl eval lagged_breakout --slowly \
  --load-model-path benchmarks/lagged_breakout/experiment_03_gpu/lag_4/seed_42/policy.bin \
  --torch.encoder LaggedBreakoutEncoder --torch.device cpu \
  --env.action-lag 0 --env.volatility 0
```

The continuation is an exploratory convergence run with adaptive training
budgets and archived tuning stages. A fixed-budget causal comparison of lag
should repeat the experiment with a common training schedule across all seeds.

For a stalled run, `--restart-from-best` restores its best validation policy and
restarts optimizer momentum while retaining the cumulative counters and history.
`--warm-start /path/to/policy.bin` instead adapts a compatible trained policy;
use exactly one lag and seed, and keep the source's adjacent `config.json`.
The source is recorded in the run configuration, and the previous full training
state is archived before either restart. In this exploratory run, some stalled
policies were adapted from the already trained lag-0 policy with the same
training seed; lag 8, seed 44 subsequently adapted its same-seed lag-4 policy.
Such runs share pretraining across lags and should be described
as fine-tuned policies, rather than independent training from scratch.
`--gamma` optionally changes the reward discount and is preserved on subsequent
resumes. A short 0.995-discount trial on lag 2, seed 43 did not improve its saved
best; that run returned to the original 0.9721246598992744 discount.

For lag 8, seed 44, checkpoint selection was expanded from 32 to 128 validation
episodes after the smaller validation set selected a policy that missed the
fresh-seed target. When the validation cohort size changes, the runner re-scores
the incumbent checkpoint and resets the validation streak before comparing new
candidates. This keeps checkpoint scores comparable within the new cohort.

To audit qualified checkpoints on a larger, separate seed set and refresh the
CSV/JSON summaries:

```bash
.venv/bin/python scripts/audit_lagged_breakout.py \
  --output benchmarks/lagged_breakout/experiment_03_gpu --episodes 256 --seed 400000
.venv/bin/python scripts/summarize_lagged_breakout.py \
  --output benchmarks/lagged_breakout/experiment_03_gpu
```

Audits are saved as `audit_training_environment.json` beside each policy and
include its SHA-256 digest. Matching completed audits are reused. The summary
uses a matching audit when available, otherwise the 64-episode qualification
report; its CSV records the report and episode count for each score. Add
`--plots` to the summary command in a Python environment with Matplotlib to
render `learning_curves.png` and `policy_comparison.png`. The latter compares
own-environment returns, common-original returns, and command-switch rates;
error bars describe variation across training seeds. `warm_start_phases.json`
records the pretraining sources and marks their adoption on the learning curves.

## Relationship to `main.pdf`

Sections 4.1 and 5 of the accompanying paper motivate examining uncertainty on
the scale `sigma * sqrt(tau)` and measuring whether latency-trained policies
become less reactive. This environment supplies execution delay, drifted
diffusion, a fixed decision/reward clock, and controlled policy comparisons.

Breakout has action-dependent future trajectories, collisions, and potentially
multiple pending decisions. It is not a direct implementation of the paper's
single-pending-action reduction or its one-dimensional multimodal reward example.
No smoothed-reward surrogate is implemented, and smoother or more robust learned
behavior is an empirical hypothesis here, not a guaranteed consequence of the
paper's equivalence. Keep training volatility fixed across lag levels, include
lag 0, and evaluate both the deterministic original game and a common noisy game
to distinguish transfer from performance under execution uncertainty.

## Tests

```bash
.venv/bin/python -m unittest tests.test_lagged_breakout -v
```

The C harness checks 100,000-frame original parity, all supported lag values,
delivery within a decision, reward timestamps, terminal payout/reset, diffusion
mean/covariance and linear variance growth, seeded reproducibility, and noisy
wall contacts. Python tests exercise native vector buffers, independent seeds,
and release-build rejection of invalid settings. Native vector tests require
`lagged_breakout` to be the currently built environment.
