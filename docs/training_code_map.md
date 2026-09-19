# Training Algorithms: CPU and GPU Code Map

This map describes the **working tree inspected on 2026-09-06**, based on commit `1b0ec4b3` with local changes. Paths are relative to the repository root; line numbers are navigation hints for this snapshot. Function names are the durable search targets. This documents current behavior and proposed extension points; it does not change training behavior.

The active learning algorithm is **PPO-style actor-critic training with GAE/V-trace-style advantages, prioritized trajectory replay, and Muon gradient updates**. There are two trainer implementations: native CUDA and PyTorch. The CPU extension supplies environments and an advantage kernel; PyTorch owns the CPU loss, backpropagation, and optimizer. Apple MPS also uses that PyTorch trainer. The direct Metal port is not connected to training yet.

**On this page**

- [Which implementation actually runs?](#which-implementation-actually-runs)
- [Source ownership at a glance](#source-ownership-at-a-glance)
- [One training update, from rollout to descent](#one-training-update-from-rollout-to-descent)
- [Exactly which optimizer is used?](#exactly-which-optimizer-is-used)
- [Training curves: what exists and where to extend it](#training-curves-what-exists-and-where-to-extend-it)
- [Backend differences to resolve before algorithm comparisons](#backend-differences-to-resolve-before-algorithm-comparisons)
- [Where to add curves, optimizers, and algorithms](#where-to-add-curves-optimizers-and-algorithms)
- [Verification and navigation](#verification-and-navigation)

## Which implementation actually runs?

The decision is in `pufferlib/pufferl.py:187`, `_resolve_backend`:

```text
puffer train ENV
  -> load_config / train / _train                  pufferlib/pufferl.py
  -> _resolve_backend(args)
     |
     +-- --slowly OR compiled _C.gpu == 0
     |     -> torch_pufferl.PuffeRL
     |        -> _resolve_device(args['torch']['device'])
     |           auto: CUDA -> Apple MPS -> CPU
     |        -> PyTorch policy, autograd, and Python Muon
     |        -> compiled _C.VecEnv for environment steps
     |
     +-- otherwise
           -> _C.create_pufferl / _C.rollouts / _C.train
           -> bindings.cu -> pufferlib.cu -> models.cu + muon.cu
           -> native CUDA policy, manual gradients, and CUDA Muon
```

**Execution matrix**

| Selected path | How selected | Policy, loss, optimizer | Advantage implementation |
|----|----|----|----|
| PyTorch on CPU | CPU build, or float CUDA build with `--slowly`; `--torch.device cpu` | PyTorch CPU operations; `pufferlib/muon.py` | `bindings_cpu.cpp` when exported; otherwise PyTorch recurrence |
| PyTorch on Apple GPU | Usually CPU build; `--torch.device mps` or `auto` | PyTorch MPS operations; `pufferlib/muon.py` | PyTorch recurrence on MPS |
| PyTorch on NVIDIA GPU | Float CUDA build with `--slowly --torch.device cuda`; also possible with a CPU extension and CUDA-enabled PyTorch | PyTorch CUDA operations; `pufferlib/muon.py` | Native CUDA kernel when exported; otherwise PyTorch recurrence on CUDA |
| Native NVIDIA GPU | CUDA extension, without `--slowly` | `src/pufferlib.cu`, `src/models.cu`, `src/muon.cu` | CUDA scalar or vectorized kernel according to horizon and precision |
| Direct Apple Metal | Separate development library | Partial model kernels only | No integrated advantage/loss/optimizer training path |

**A CPU build does not force CPU learning.** It sets `_C.gpu = 0` and selects the PyTorch trainer automatically, even without `--slowly`. That trainer's device default is `auto`. To force CPU learning, set `--torch.device cpu`. Conversely, `--torch.device` does not select the trainer: with a CUDA build, use `--slowly` to make that setting take effect.

`build.sh:408` compiles `src/bindings.cu` for native CUDA; `build.sh:439` compiles `src/bindings_cpu.cpp` for `--cpu`. Both produce the environment-specific `pufferlib._C` extension. The macOS helper `scripts/build_macos_cpu.sh` invokes the CPU build. The PyTorch trainer's import guard at `torch_pufferl.py:21` requires `precision_bytes == 4`: CPU builds already satisfy this; CUDA builds need `--float` for PyTorch. An unavailable/unimportable CUDA extension is not automatically rebuilt or replaced by a CPU extension.

The vector interface and policy device are separate. In `PuffeRL.create_pufferl` (`torch_pufferl.py:498`), `vec_gpu` is true only when the policy uses CUDA and the loaded extension supports CUDA. MPS and CPU policies use host buffers. This method also forces `num_buffers = 1`.

**Environment simulation still runs on CPU in these training paths.** `gpu_vec_step` transfers actions to the host, calls the CPU environment steps, and transfers observations/rewards/terminals back. The native trainer overlaps the same CPU environment work with CUDA policy work using multiple buffers and worker threads. `gpu` in the vector interface describes CUDA buffers/transfers, not CUDA simulation of `c_step`.

## Source ownership at a glance

The source links below open files in this repository.

**Main source files**

| File | Responsibility and useful symbols |
|----|----|
| [pufferlib/pufferl.py](../pufferlib/pufferl.py) | Shared orchestration: `_resolve_backend`, `_train`, `load_config`, `unroll_nested_dict`. Logging, dashboard, checkpoints, sweeps. |
| [src/bindings_cpu.cpp](../src/bindings_cpu.cpp) | CPU pybind11 interface: `create_vec`, `cpu_vec_step_py`, `vec_log`, `py_puff_advantage_cpu`. No native trainer or optimizer. |
| [pufferlib/torch_pufferl.py](../pufferlib/torch_pufferl.py) | PyTorch trainer: `PuffeRL.rollouts`, `train`, `log`; `compute_puff_advantage`, `sample_logits`, `load_policy`. |
| [pufferlib/muon.py](../pufferlib/muon.py) | PyTorch optimizer: `Muon.step`, `zeropower_via_newtonschulz5`. |
| [src/bindings.cu](../src/bindings.cu) | Native Python API: `create_pufferl`, `rollouts`, `train`, `puf_log`, `puf_eval_log`, weight I/O; also CUDA vector/advantage bindings used by the PyTorch CUDA path. |
| [src/pufferlib.cu](../src/pufferlib.cu) | Native trainer: `train_impl`, `net_callback_wrapper`, `puff_advantage_cuda`, `prio_replay_cuda`, `ppo_loss_fwd_bwd`, `build_policy`, `create_pufferl_impl`. Owns training buffers. |
| [src/muon.cu](../src/muon.cu) | Native descent: `muon_init`, `muon_step`, gradient norm clipping, momentum, Newton-Schulz iterations, weight update, NCCL reduction. |
| [src/models.cu](../src/models.cu) | Native policy forward/backward, encoder/decoder, MinGRU recurrent scans, model parameter and activation registration. |
| [pufferlib/models.py](../pufferlib/models.py) | PyTorch policy components, including `MinGRU`, `MLP`, `LSTM`, default and environment-specific encoders/decoders. |
| [src/kernels.cu](../src/kernels.cu) | CUDA tensor operations, cuBLAS GEMM wrappers, precision selection, initialization, allocation. `src/tensor.h` supplies tensor shapes/types. |
| [src/vecenv.h](../src/vecenv.h) | Shared C environment runtime: `create_static_vec`, `static_omp_threadmanager`, `cpu_vec_step`, `gpu_vec_step`, `static_vec_log`. Calls environment `c_step` and `my_log`. |
| [src/ocean.cu](../src/ocean.cu) | Custom native neural encoder selection, currently `nmmo3` via `create_custom_encoder`. Includes convolution support from `src/cudnn_conv2d.cu`; this file is model code. |
| `ocean/<environment>/` | Simulation, observations, rewards, termination, and environment logs. `binding.c` defines action/observation metadata and `my_log`. Some environments also link C++ simulation code. |
| [src/puffernet.h](../src/puffernet.h) | Standalone CPU policy inference, used by environment executables. `forward_puffernet` is not the CPU training/backpropagation fallback. |
| [src/metal/metal_backend.h](../src/metal/metal_backend.h) | Direct Metal C API. `metal_backend.mm` manages Metal dispatch and MPS matrix multiplication; `kernels.metal` has primitive/MinGRU kernels. See `src/metal/README.md` for completed and planned stages. |

## One training update, from rollout to descent

Let `B = total_agents`, `T = horizon`, and `M = minibatch_size` for one trainer instance. A rollout collects `B*T` agent steps. One call to `train` performs `int(replay_ratio * B*T / M)` optimizer steps, each sampling `M/T` complete trajectories **with replacement**. `M` must be divisible by `T`; choose settings giving at least one minibatch. The current configuration validator does not check that last condition.

**Ordered execution map**

| Stage | Native CUDA | PyTorch CPU / MPS / CUDA |
|----|----|----|
| Step 1: Collect rollout | `bindings.cu:135` `rollouts` -\> `vecenv.h:253` `static_omp_threadmanager` -\> `pufferlib.cu:597` `net_callback_wrapper`. `policy_forward` and CUDA `sample_logits` produce actions. | `torch_pufferl.py:293` `PuffeRL.rollouts` -\> `policy.forward_eval` and Python `sample_logits` under `torch.no_grad` -\> `VecEnv.cpu_step` or `gpu_step`. |
| Step 2: Prepare training data | `pufferlib.cu:1498` `train_impl` transposes `RolloutBuf` from `[T,B,...]` to `[B,T,...]`. | `torch_pufferl.py:342` `PuffeRL.train` transposes its tensors. Both trainers clamp rewards to `[-1,1]` and reset stored ratios to 1. |
| Step 3: Recompute advantages | `pufferlib.cu:1380` `puff_advantage_cuda` selects the CUDA kernel. | `torch_pufferl.py:554` `compute_puff_advantage` dispatches by tensor device and available extension functions. |
| Step 4: Sample trajectories | `pufferlib.cu:1241` `prio_replay_cuda` -\> priority reduction, CDF, cuRAND sampling, importance weights; `select_copy` at line 1447 gathers `TrainGraph` tensors. | `PuffeRL.train` computes priorities, calls `torch.multinomial(..., replacement=True)`, indexes tensors. |
| Step 5: Evaluate loss and its derivatives | `models.cu:789` `policy_forward_train` -\> `pufferlib.cu:1048` `ppo_loss_fwd_bwd` -\> `ppo_loss_compute` at line 787. | `self.policy(mb_obs)` -\> score stored actions with `sample_logits` -\> build scalar loss in `PuffeRL.train`. |
| Step 6: Backpropagate | `models.cu:798` `policy_backward` propagates manually assembled logits/value/log-standard-deviation gradients through decoder, recurrent network, encoder. | `loss.backward()` uses PyTorch autograd through `models.py`. |
| Step 7: Clip and update weights | `muon.cu:159` `muon_step`; BF16 mode then casts float32 master weights back to policy precision in `train_impl`. | `clip_grad_norm_` -\> `self.optimizer.step()` -\> `self.optimizer.zero_grad()`; optimizer is `pufferlib.muon.Muon`. |
| Step 8: Refresh replay estimates | `pufferlib.cu:1671` copies sampled rows' new ratios and values back into training rollout buffers. | `self.ratio[idx] = ratio.detach()` and `val[idx] = newvalue.detach().float()` update the same estimates. |
| Step 9: Publish metrics | `bindings.cu:14` `puf_log` reads accumulated device losses. | `torch_pufferl.py:453` `PuffeRL.log` publishes the latest update's losses. Both return dictionaries consumed by `pufferl._train`. |

Stages 3--8 repeat for every minibatch. Advantages use the refreshed values and ratios on the next pass; behavior-policy log probabilities stay fixed for the rollout. Replay here reuses the current rollout batch. Neither trainer owns a persistent, cross-rollout replay buffer.

### The actor-critic objective

Both loss implementations combine a clipped policy objective, a clipped value objective, and an entropy bonus. In code-oriented notation:

```text
ratio = exp(new_logprob - rollout_logprob)
adv = priority_correction * (advantage - mean) / (std + 1e-8)
policy_loss = mean(max(-adv * ratio,
                       -adv * clamp(ratio, 1-clip_coef, 1+clip_coef)))
returns = advantage + stored_value
clipped_value = stored_value + clamp(new_value - stored_value,
                                     -vf_clip_coef, vf_clip_coef)
value_loss = 0.5 * mean(max((new_value - returns)**2,
                             (clipped_value - returns)**2))
total_loss = policy_loss + vf_coef * value_loss - ent_coef * mean(entropy)
```

Advantage normalization is over the sampled minibatch. The priority correction weights the policy term. Priorities are based on `sum(abs(advantages), time)**prio_alpha`; sampling weights add an epsilon. The correction is `(B * prio_probs[selected])**(-beta)` with `beta = prio_beta0 + (1-prio_beta0)*prio_alpha*epoch/total_epochs`. The native sampling CDF and PyTorch multinomial implementation are separate; matching hyperparameters alone does not imply identical sampled batches.

The default policy uses an encoder, a MinGRU recurrent network, and action/value heads. The recurrent architecture is distinct from the learning algorithm and the optimizer. Native recurrent kernels are `mingru_gate` (rollout), `mingru_scan_forward` and `mingru_scan_backward` in `models.cu`. PyTorch uses `MinGRU.forward_eval` and `MinGRU.forward_train` in `models.py:228`. Selecting `MLP` or `LSTM` changes the policy architecture, not the PPO-style objective.

## Exactly which optimizer is used?

**Muon is hardcoded in both trainers.** Native initialization is `src/pufferlib.cu:2057` (`muon_init`), and its update call is at line 1648. PyTorch initialization is `pufferlib/torch_pufferl.py:256` and the update is inside `PuffeRL.train`. There is no active Adam/AdamW training branch or algorithm/optimizer selector in the shared configuration.

The implementation in `src/muon.cu` mirrors the sequence in `pufferlib/muon.py:85`:

1.  Clip the global parameter-gradient norm to `max_grad_norm`. Native multi-GPU training first averages gradients with `ncclAllReduce`.
2.  Update a momentum buffer: `m = beta1*m + gradient`; form the Nesterov update `gradient + beta1*m`.
3.  For tensors with at least two dimensions, flatten to a matrix `[first_dimension, remaining_elements]`, normalize its Frobenius norm, and perform five Newton-Schulz iterations to approximately orthogonalize the update. Both files contain the same five coefficient triples.
4.  Scale matrix updates by `sqrt(max(1, rows/columns))`. Lower-dimensional parameters use the momentum update directly.
5.  Apply `weight = weight*(1-lr*weight_decay) - lr*update`.

Native matrix operations go through `cublasGemmExDense` / `puf_addmm_nn` in `src/kernels.cu:262`. Native master weights and momentum are float32; gradient and Newton-Schulz working buffers use build-selected `precision_t` (BF16 by default, float32 with `--float`). PyTorch uses its tensor operations on the selected device; this trainer does not enable an AMP/autocast mode.

**Configuration that reaches descent**

| Setting in `[train]` | Current effect |
|----|----|
| `learning_rate`, `anneal_lr`, `min_lr_ratio` | Base rate and cosine schedule in both trainers. Native updates a device scalar read by the captured update; PyTorch updates the first parameter group's rate. The current optimizer has one group. |
| `beta1` | Muon momentum coefficient, default 0.95. |
| `beta2` | Parsed into native `HypersT`, but unused by either trainer's Muon. |
| `eps` | Accepted/stored but does not reach the actual Muon normalization. Newton-Schulz norm floors are hardcoded to `1e-7`; norm clipping uses `1e-6` in native code and the PyTorch clipping helper's default. |
| `max_grad_norm` | Global gradient clipping before momentum/orthogonalization. |
| Weight decay | Supported by the Muon implementations, but the trainer wiring uses zero. There is no shared `train.weight_decay` setting. |

## Training curves: what exists and where to extend it

The shared seam is **\`\`pufferlib/pufferl.py:285\`\` in \`\`\_train\`\`**, immediately after `backend.log` / `backend.eval_log` returns and before filtering or downsampling. A curve logger can serve both trainers there. `unroll_nested_dict` (line 43) converts nested fields to slash-separated keys.

```text
Environment Log -> vecenv.h static_vec_log -> binding my_log ----+
                                                               |
Native PPO kernels -> losses_puf -> bindings.cu puf_log ---------+-->
PyTorch train -> self.losses -> PuffeRL.log ---------------------+   _train
                                                                   |
                        flattened keys + agent_steps + uptime <----+
                           |           |             |
                        dashboard   wandb.log    all_logs in memory
                                                    |
                                            downsample at run end
                                                    |
                                        logs/ENV/RUN_ID.json
```

### Current access options

- The regular CLI writes config plus column-oriented `metrics` arrays to `logs/<env_name>/<run_id>.json` at the end of a successfully logged run. `agent_steps` is the main curve x-axis; `uptime` is a time-based axis. `[sweep] downsample` controls compression even for ordinary training; its default is 5. This file is a compressed summary, not a live history.
- `--wandb` sends flattened records from `_train` at the logging cadence, using `agent_steps` as the step. Training logging is rate-limited to about one call per 0.6 seconds, with a final training log forced.
- Direct backend access already works: call `backend.rollouts(trainer)`, `backend.train(trainer)`, then `backend.log(trainer)`. The native `train` wrapper returns an empty dictionary and PyTorch `train` returns `None`; metrics come from `log`.
- Local experiment scripts already demonstrate richer access: `scripts/compare_lagged_breakout.py:44` (`train_policy`) writes `training.jsonl`; `scripts/continue_lagged_breakout.py:61` (`train_run`) adds periodic validation and resumable optimizer state. These are experiment-specific implementations, not shared CLI facilities.

For an existing CLI JSON file, curve coordinates can be read without loading the trainer:

```python
import json
from pathlib import Path

run = json.loads(Path("logs/ENV/RUN_ID.json").read_text())
metrics = run["metrics"]
score_curve = list(zip(metrics["agent_steps"], metrics["env/score"]))
# Use env/episode_return instead when that is the desired environment metric.
```

### Metric names and aggregation differ

**Flattened loss fields currently returned by `log`**

| Quantity               | Native CUDA     | PyTorch                   |
|------------------------|-----------------|---------------------------|
| Policy loss            | `loss/policy`   | `loss/policy_loss`        |
| Value loss             | `loss/value`    | `loss/value_loss`         |
| Entropy                | `loss/entropy`  | `loss/entropy`            |
| Total objective        | `loss/total`    | Not emitted               |
| Old approximate KL     | `loss/old_kl`   | `loss/old_approx_kl`      |
| Approximate KL         | `loss/kl`       | `loss/approx_kl`          |
| Clipped ratio fraction | `loss/clipfrac` | `loss/clipfrac`           |
| Mean ratio             | Not emitted     | `loss/importance`         |
| Explained variance     | Not emitted     | `loss/explained_variance` |

Native `ppo_loss_reduce` (`pufferlib.cu:979`) accumulates minibatch means and a count in `losses_puf`; `puf_log` averages and clears that buffer. Thus its loss record covers updates since the previous log. PyTorch stores only the latest `train` call's minibatch averages in `self.losses`. Neither currently emits the effective learning rate, entropy coefficient, gradient norm, or optimizer update norm through the regular loss dictionary.

Environment metrics follow `Log` fields updated in the environment and `my_log` in its binding. For example, `ocean/breakout/binding.c:30` exports `score`, `episode_return`, `episode_length`, and `perf`. `vecenv.h:696` aggregates fields and divides by `Log.n`; `static_vec_log` then clears environment counters. Add new environment metrics at those producers, not inside the PPO loss.

There are three details to preserve when adding curves:

- PyTorch calls `self._vec.log()` after **every rollout**, clearing environment counters then. If the CLI skips logs, earlier rollouts' environment summaries are overwritten. Native environment counters accumulate until `puf_log`.
- `_train` only sends W&B records and appends training history when the configured `env/<sweep.metric>` exists. A new general logger should retain available loss/performance records even before the first episode completes.
- The final CLI summary combines training performance with evaluation environment metrics. Native `puf_eval_log` returns only cumulative, non-clearing environment metrics; PyTorch aliases `eval_log = log` and continues its per-rollout clearing behavior. A new curve schema should explicitly label `train` versus `eval` and avoid carrying stale training fields into evaluation records.

## Backend differences to resolve before algorithm comparisons

These are observations from source, not claims of numerical equivalence.

**The advantage recurrence currently differs across CUDA branches.** All implementations iterate backward from `T-2` and use rewards/terminal flags at `t+1`. The last advantage remains zero in the zero-initialized buffer; the loss still processes all `T` entries. With `q = 1 - done[t+1]`, `rho = min(ratio[t], rho_clip)` and `c = min(ratio[t], c_clip)`, the shared recursion is:

```text
advantage[t] = delta + gamma * gae_lambda * c * q * advantage[t+1]

CPU C++, portable PyTorch, scalar CUDA:
    delta = rho * reward[t+1] + gamma * value[t+1] * q - value[t]

Vectorized CUDA:
    delta = rho * (reward[t+1] + gamma * value[t+1] * q - value[t])
```

Sources: `bindings_cpu.cpp:40`, `torch_pufferl.py:532`, `pufferlib.cu:1265` (scalar), and `pufferlib.cu:1314` (vectorized). `puff_advantage_cuda` and the Python CUDA binding both select the vectorized branch when `T` is divisible by `16 / sizeof(precision_t)`: 4 for float32, 8 for BF16. The default horizon 64 takes that branch. When `rho != 1`, the formula can therefore depend on device, extension build, and horizon. For example, with reward 1, current/next values 2/4, gamma 0.9, rho 0.5, and no terminal, scalar delta is 2.1 while vectorized delta is 1.3. Fixing this needs an explicit choice of intended recurrence and a parity test covering non-unit ratios and both CUDA dispatch branches.

Other material differences:

- **Entropy scheduling:** `anneal_ent_coef` / `min_ent_coef_ratio` reach native `train_impl:1563` through `HypersT` and a device scalar, including CUDA graph replay. PyTorch always uses `config['ent_coef']` in its loss.
- **Policy structure:** the native default encoder/decoder are bias-free, and the decoder fuses policy/value output columns. PyTorch defaults use biased linear layers and separate value/action modules. Initializations also differ. Equal `hidden_size`/`num_layers` does not establish model equivalence.
- **Recurrent state:** native rollout/minibatch state resetting is controlled by `reset_state`. PyTorch resets rollout state on every `rollouts` call; its standard recurrent training paths start from zero state.
- **Action semantics:** native sampling/loss supports environment action masks via `MY_ACTION_MASK` buffers. The current PyTorch vector/policy path does not consume those buffers. Conversely, PyTorch's `QueueReactiveMarketDistribution` has conditional sampling/log-probability/ entropy logic in `_sample_queue_reactive_market`; native generic action heads do not implement that conditional distribution.
- **Checkpoints:** native `bindings.cu:180` saves flat float32 master weights; PyTorch `torch_pufferl.py:480` saves a model `state_dict`. Both regular APIs save weights without optimizer/RNG/training-loop state, and the formats are different despite the shared `.bin` extension. Compare curves from resumed runs with this distinction recorded.

## Where to add curves, optimizers, and algorithms

The following are **proposed changes**, not existing configuration options.

**Extension map**

| Desired extension | First implementation point | Native/backend work needed |
|----|----|----|
| Live training curves and export | Add one metric-record adapter/logger in `pufferl._train` after the backend log call. Normalize loss keys and include run ID, phase, agent steps, update count, backend, device, precision, config, seed. Append JSONL/CSV before downsampling; expose a callback/iterator for UI. | Reuse the same sampled record for all consumers: reading native `log` again clears accumulators. Align aggregation windows and expose optimizer/schedule scalars through both backend log methods. |
| Another descent optimizer | Introduce an optimizer factory where `PuffeRL.__init__` creates `Muon`. Prototype AdamW/SGD using PyTorch on any supported device. Give each optimizer its own relevant configuration fields. | Add optimizer selection/state allocation beside `muon_init` and dispatch beside `muon_step`. Preserve gradient reduction/clipping, float32 master weights, and device-resident schedules for CUDA graphs. |
| Another actor-critic loss or advantage rule | Factor `PuffeRL.train` into advantage, sampling, objective, and update functions. Make the recurrence choice explicit in `compute_puff_advantage`; add a trainer/algorithm selection seam. | Update `train_impl`, `ppo_loss_fwd_bwd`/`ppo_loss_compute` and both CUDA advantage branches. Native loss changes require updating the manually implemented derivatives, not just scalar loss reporting. |
| Off-policy algorithm with long-lived replay | Add an algorithm-specific trainer with its own replay storage, collection/update cadence, model heads, and targets. | Extend or replace `RolloutBuf`, `TrainGraph`, `PPOBuffersPuf` and model/backprop interfaces. The current per-rollout replay does not supply target networks or a persistent replay store. |
| New policy architecture | Implement compatible modules in `models.py` and select via `[torch] network/encoder/decoder` as consumed by `load_policy`. | `build_policy` currently selects MinGRU directly. Extend its `Encoder`/`Network`/`Decoder` interfaces, registration, activation allocation, and forward/backward kernels. `[torch]` names do not configure native models. |
| Complete direct Metal training | Follow `src/metal/README.md`: finish model integration, then native rollout/sampling/advantage/PPO, then Muon and training submission. | Add a CLI backend adapter after those pieces exist. PyTorch MPS remains the integrated Apple GPU training path meanwhile. |
| Faithful continuation and curve comparisons | Add a versioned full-training checkpoint schema; use the continuation experiment script as a reference for optimizer/counter/RNG persistence. | Native serialization also needs optimizer state, counters/schedules, and RNG state. Record whether environment/recurrent state is restored or restarted. Standard `save_weights` alone cannot provide this. |

Configuration flows from `config/default.ini` plus environment overrides through `load_config`. New native settings additionally need to be copied in `bindings.cu:create_pufferl` and represented in `HypersT`; adding an INI field alone does not implement native behavior. Existing PyTorch network selection uses a module constructor contract, so a class in `models.py` must match the arguments supplied by `load_policy` to be selectable.

A practical order is to establish the shared metric stream, settle advantage and aggregation parity, then prototype optimizer/loss choices in PyTorch. Port validated choices into CUDA and the developing Metal backend with an explicit record of which backends support each algorithm.

## Verification and navigation

This document is primarily a source trace. It does not establish end-to-end CPU/CUDA learning parity or certify the unfinished Metal trainer.

Relevant existing checks and their scope:

- `tests/test_torch_devices.py` checks device selection and CPU/PyTorch advantage parity; optional MPS checks compare advantages and policy/optimizer behavior. Its advantage fixture has horizon 17 and does not test vectorized CUDA against the scalar recurrence.
- `tests/test_ent_coef_anneal.py` exercises entropy annealing through native CUDA graphs; it does not test PyTorch entropy scheduling.
- `tests/test_muon.py` is a standalone PyTorch-versus-HeavyBall comparison script requiring `heavyball`. It is not a CUDA `muon_step` parity test.
- `tests/test_metal_backend.py` tests the direct Metal primitives and model kernels, not a complete reinforcement-learning update.

For new algorithm work, compare fixed rollout tensors and identical initial weights before comparing noisy learning curves. Include non-unit importance ratios, terminal boundaries, aligned/unaligned horizons, discrete/continuous actions, loss derivatives, optimizer updates, and schedules after graph capture. Then compare full runs with the same evaluation protocol and explicit backend/precision metadata.

Useful source searches:

```bash
rg -n '_resolve_backend|def _train|unroll_nested_dict' pufferlib/pufferl.py
rg -n 'train_impl|puff_advantage|prio_replay_cuda|ppo_loss_' src/pufferlib.cu
rg -n 'muon_init|muon_step|muon_weight_update' src/muon.cu src/pufferlib.cu
rg -n 'def train|compute_puff_advantage|self.optimizer' pufferlib/torch_pufferl.py
rg -n 'puf_log|losses_dict|save_weights' src/bindings.cu
rg -n 'static_vec_log|static_vec_eval_log|c_step' src/vecenv.h
```
