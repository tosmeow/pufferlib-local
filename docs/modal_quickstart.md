# Modal quickstart

Run the native PufferLib 5.0 trainer on a remote NVIDIA T4 while keeping your
checkout local.

## Once per machine

```bash
uv venv .venv-modal --python .venv/bin/python
uv pip install --python .venv-modal/bin/python 'modal>=1.5.5,<2'
.venv-modal/bin/modal setup
```

## Check the launcher

```bash
.venv-modal/bin/modal run scripts/modal_native.py --help
```

## Short smoke test

This builds the native CUDA binary remotely and runs 1,024 training steps:

```bash
.venv-modal/bin/modal run scripts/modal_native.py \
  --env breakout \
  --run-name breakout_smoke \
  --smoke
```

## Real training

```bash
.venv-modal/bin/modal run --detach scripts/modal_native.py \
  --env breakout \
  --run-name breakout_modal_01 \
  --overrides '--train.total_timesteps=55000000 --vec.total_agents=4096'
```

Use a new `--run-name` each time. Training, evaluation, matching, and sweeps
are selected with `--mode train|eval|match|sweep`.

## Results

Results are persisted in the Modal Volume `pufferlib-5-results`:

```bash
mkdir -p benchmarks/modal_native
.venv-modal/bin/modal volume get pufferlib-5-results \
  /breakout_modal_01 benchmarks/modal_native
```

Each run contains `console.log`, `modal_run.json`, native logs, and
checkpoints.

## What the launcher does

- Uploads the current native 5.0 source, including uncommitted source edits.
- Excludes local binaries, virtual environments, checkpoints, and benchmarks.
- Builds `./puffer` in a CUDA 12.8 Linux image for a T4 (`sm_75`).
- Runs the selected native command remotely and streams its output.
- Saves logs, checkpoints, and a run manifest to the persistent Volume.

The old lagged-breakout/Python trainer is not part of current 5.0, so this
launcher cannot run that experiment until it is ported to the native 5.0 API.
