#!/usr/bin/env bash
# Run inside the launcher's Linux CUDA image, from the repository root.
set -euo pipefail

# PyTorch's CUDA wheels can contain only versioned shared libraries. build.sh
# links with -lcudnn/-lnccl, so supply the corresponding linker names.
python - <<'PY'
from importlib import import_module
from pathlib import Path

for module, library, soname in (
    ("nvidia.cudnn", "libcudnn.so", "libcudnn.so.9"),
    ("nvidia.nccl", "libnccl.so", "libnccl.so.2"),
):
    directory = Path(next(iter(import_module(module).__path__))) / "lib"
    target = directory / soname
    if not target.is_file():
        raise FileNotFoundError(target)
    link = directory / library
    if not link.exists():
        link.symlink_to(soname)
PY

# Image builders have no GPU. T4 uses sm_75; do not use build.sh's default
# -arch=native. The NVML stub is for linking only: never put it in LD_LIBRARY_PATH.
export NVCC_ARCH=sm_75
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
bash build.sh lagged_breakout --float
