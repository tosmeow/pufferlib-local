#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${1:?usage: build_modal_native.sh ENV}

# The image builder has no GPU, so compile for the T4 requested by the
# launcher instead of using build.sh's default -arch=native.
export NVCC_ARCH=${NVCC_ARCH:-sm_75}

# build.sh links against libnvidia-ml.so, which is supplied as a CUDA stub in
# the devel image. Keep the stub available to the host linker.
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"

# The pip NCCL wheel normally ships only the versioned shared object. Provide
# the unversioned linker name expected by build.sh.
python - <<'PY'
from importlib import import_module
from pathlib import Path

directory = Path(next(iter(import_module("nvidia.nccl").__path__))) / "lib"
target = directory / "libnccl.so.2"
link = directory / "libnccl.so"
if not target.is_file():
    raise FileNotFoundError(target)
if not link.exists():
    link.symlink_to(target.name)
PY

# Float32 is the portable native-training choice for the T4 image.
bash build.sh "$ENV_NAME" --float
