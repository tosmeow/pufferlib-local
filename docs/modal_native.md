# Native 5.0 training on Modal

The old 4.0 Modal launcher used the Python/PyTorch trainer. PufferLib 5.0
uses the native CUDA trainer, so this checkout has a separate launcher that
uploads the current source, builds `./puffer` on a Linux CUDA image, and runs
the native `train`, `eval`, `match`, or `sweep` command on a T4.

The source upload includes uncommitted source edits but excludes local
virtual environments, binaries, checkpoints, and benchmark artifacts.

## Setup

The Modal client can live in its own environment:

```bash
uv venv .venv-modal --python .venv/bin/python
uv pip install --python .venv-modal/bin/python 'modal>=1.5.5,<2'
.venv-modal/bin/modal setup
```

Check the local command interface without launching a remote job:

```bash
.venv-modal/bin/modal run scripts/modal_native.py --help
```

## Smoke training

This launches a short paid T4 job and stores its output in the
`pufferlib-5-results` Modal Volume:

```bash
.venv-modal/bin/modal run scripts/modal_native.py \
  --env breakout --run-name breakout_smoke --smoke
```

The smoke override is 1,024 training steps with 64 agents, a 16-step horizon,
and one CPU thread. It checks the remote native build and training path; it is
not a performance benchmark.

## Real training

Configuration values are passed as a quoted string of native PufferLib
overrides:

```bash
.venv-modal/bin/modal run --detach scripts/modal_native.py \
  --env breakout \
  --run-name breakout_modal_01 \
  --overrides '--train.total_timesteps=55000000 --vec.total_agents=4096'
```

The command uses one T4, four CPUs, and 16 GiB of RAM. Use a new run name for
each invocation; the launcher refuses to overwrite an existing Volume path.

To download results locally:

```bash
mkdir -p benchmarks/modal_native
.venv-modal/bin/modal volume get pufferlib-5-results \
  /breakout_modal_01 benchmarks/modal_native
```

The downloaded directory contains `modal_run.json`, `console.log`, logs, and
native checkpoints. The existing 4.0 lagged-breakout launcher is not copied
because lagged-breakout and the Python trainer are not present in the current
5.0 branch; that environment must first be ported to the native 5.0 API.
