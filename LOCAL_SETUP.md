# Local Setup

This checkout is a private local fork based on PufferLib `4.0`.

## Environment

```bash
uv --cache-dir .uv-cache venv --python /opt/anaconda3/bin/python3.12 .venv
uv --cache-dir .uv-cache pip install -e .
```

Activate it when working interactively:

```bash
source .venv/bin/activate
```

## Apple Silicon CPU Build

The fast CUDA backend is for Linux/NVIDIA machines. On this Mac, use the CPU vector backend plus the PyTorch fallback:

```bash
brew install llvm libomp raylib
puffer build breakout
.venv/bin/puffer train breakout --slowly
```

For quick smoke tests:

```bash
.venv/bin/puffer train breakout --slowly \
  --train.total-timesteps 256 \
  --vec.total-agents 32 \
  --vec.num-buffers 1 \
  --vec.num-threads 4 \
  --train.horizon 8 \
  --train.minibatch-size 64 \
  --policy.hidden-size 16 \
  --policy.num-layers 1 \
  --checkpoint-dir /private/tmp/pufferlib-checkpoints \
  --log-dir /private/tmp/pufferlib-logs
```

The build script has local macOS compatibility adjustments:

- skips x86-only AVX/FMA flags on ARM
- prefers Homebrew raylib on macOS
- links raylib dynamically on macOS
- retargets `_C` to the venv's bundled PyTorch `libomp.dylib`

## Useful Endpoints

- Docs: https://puffer.ai/docs.html
- Source: https://github.com/PufferAI/PufferLib/tree/4.0
- Docker/GPU environment: https://github.com/PufferAI/PufferTank
- Support: https://discord.gg/puffer

## Local Sphinx Docs

```bash
uv --cache-dir .uv-cache pip install sphinx furo sphinx-copybutton
scripts/build_docs.sh
```

Open `docs/_build/html/index.html` after building.
