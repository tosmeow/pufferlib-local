#!/usr/bin/env bash
set -euo pipefail

# Siva's RTX 5070 Ti needs a compiler with native sm_120 support (CUDA 12.8+).
# Leave the shared system compiler unchanged; select a compatible toolkit.
cuda_candidates=()
if [[ -n "${CUDA_HOME:-${CUDA_PATH:-}}" ]]; then
    cuda_candidates+=("${CUDA_HOME:-${CUDA_PATH}}")
else
    cuda_candidates+=(/usr/local/cuda /usr/local/cuda-* /opt/cuda /opt/cuda-*)
    if command -v nvcc >/dev/null 2>&1; then
        cuda_candidates+=("$(dirname "$(dirname "$(command -v nvcc)")")")
    fi
fi

cuda_selected=""
for cuda_candidate in "${cuda_candidates[@]}"; do
    if [[ -x "$cuda_candidate/bin/nvcc" ]] &&
        "$cuda_candidate/bin/nvcc" --list-gpu-code | grep -qx 'sm_120'; then
        cuda_selected="$cuda_candidate"
        break
    fi
done
if [[ -z "$cuda_selected" ]]; then
    echo "RPS debug requires CUDA Toolkit 12.8+ with sm_120 support on Siva." >&2
    echo "Ask the cluster operator to provision it, then set CUDA_HOME to its installation directory." >&2
    exit 1
fi
export CUDA_HOME="$cuda_selected"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVCC_ARCH=sm_120
echo "Using CUDA_HOME=$CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version

./build.sh rock_paper_scissors rps_run --float

./rps_run train \
    --train.gpus=1 \
    --train.total_timesteps=2000000000 \
    --train.horizon=64 \
    --train.minibatch_size=8192 \
    --vec.total_agents=1024 \
    --vec.num_buffers=1 \
    --vec.num_threads=2 \
    --vec.num_policies=1 \
    --vec.hist_policy_percent=0 \
    --selfplay.enabled=0 \
    --base.reset_every_horizon=0 \
    --base.checkpoint_interval=100 \
    --base.eval_episodes=10000 \
    --base.checkpoint_dir="${CLUSTER_RESULTS_DIR}/checkpoints" \
    --base.log_dir="${CLUSTER_RESULTS_DIR}/logs"