#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root, inside a Slurm job.
usage() {
    cat <<'HELP'
Usage: bash scripts/rps.sh COMMAND ...
  run BOT [STEPS [LR]]              Train fresh, evaluate, and check parity.
                                   Defaults: 200M steps, LR 0.015, seed 73.
  train [--section.key=value ...]   Train only, with trainer overrides.
  eval CHECKPOINT HIDDEN [LAYERS [SHA256 [OBS_SIZE [ROUND_OBS]]]]
                                   Evaluate + adaptation + CPU/GPU parity.
  parity CHECKPOINT HIDDEN [LAYERS [SHA256 [OBS_SIZE [ROUND_OBS]]]]
                                   Check CPU/GPU inference only.
  analyze summary DIRECTORY [--output DIRECTORY]
  analyze adaptation CSV SHA256 [--output DIRECTORY]
  analyze parity DIRECTORY
GPU commands write to CLUSTER_RESULTS_DIR. Analyze commands need only a CPU.
HELP
}
command=${1:-help}
shift || true
case "$command" in
    help|-h|--help) usage; exit 0 ;;
    analyze) exec python3 scripts/rps_analysis.py "$@" ;;
    run|train|eval|parity) ;;
    *) usage >&2; exit 1 ;;
esac
root=${CLUSTER_RESULTS_DIR:?Submit through cluster run}
obs_size=7; round_obs=1
if [[ "$command" == run ]]; then
    bot=${1:?run needs BOT}; steps=${2:-200000000}; learning_rate=${3:-0.015}
    case "$bot" in 0|1|2|3|4|5) ;; *) echo 'BOT must be 0..5' >&2; exit 1 ;; esac
elif [[ "$command" == eval || "$command" == parity ]]; then
    checkpoint=${1:?checkpoint required}; hidden=${2:?hidden size required}; layers=${3:-4}
    obs_size=${5:-7}; round_obs=${6:-1}
    [[ "$obs_size" == 6 || "$obs_size" == 7 ]]
    [[ "$round_obs" == 0 || "$round_obs" == 1 ]]
    if [[ $# -ge 4 ]]; then printf '%s  %s\n' "$4" "$checkpoint" | sha256sum --check; fi
fi

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
    echo "RPS requires CUDA Toolkit 12.8+ with sm_120 support on Siva." >&2
    echo "Ask the cluster operator to provision it, then set CUDA_HOME to its installation directory." >&2
    exit 1
fi
export CUDA_HOME="$cuda_selected"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVCC_ARCH=sm_120
echo "Using CUDA_HOME=$CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version

./build.sh rock_paper_scissors build/rps_run --float

train() {
    build/rps_run train \
        --train.gpus=1 --train.total_timesteps=2000000000 \
        --train.horizon=64 --train.minibatch_size=8192 \
        --vec.total_agents=1024 --vec.num_buffers=1 --vec.num_threads=2 \
        --vec.num_policies=1 --vec.hist_policy_percent=0 --selfplay.enabled=0 \
        --base.reset_every_horizon=0 --base.checkpoint_interval=100 \
        --base.eval_episodes=10000 \
        --base.checkpoint_dir="$root/checkpoints" --base.log_dir="$root/logs" "$@"
}

compile_cpu_probe() {
    gcc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L -DRPS_BOT_EVAL -Isrc \
        -DRPS_HIDDEN_SIZE="$hidden" -DRPS_NUM_LAYERS="$layers" \
        -DRPS_OBS_SIZE="$obs_size" -DRPS_ROUND_OBSERVATION="$round_obs" \
        -Iraylib-5.5_linux_amd64/include scripts/rps_policy_probe.c -lm -o build/rps_policy_probe
}

parity() {
    mkdir -p "$root/parity"
    nvcc -O2 -arch=sm_120 -std=c++17 -DPRECISION_FLOAT -DRPS_OBS_SIZE="$obs_size" -I. -Isrc -Ivendor \
        -Iraylib-5.5_linux_amd64/include -I"$CUDA_HOME/include/cccl" \
        -Xcompiler=-fopenmp -Xcompiler=-Wno-narrowing --diag-suppress=2361 \
        scripts/rps_gpu_probe.cu raylib-5.5_linux_amd64/lib/libraylib.a \
        -L"$CUDA_HOME/lib64" -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
        -lm -Xlinker=-lpthread -lGL -o build/rps_gpu_probe
    build/rps_policy_probe "$checkpoint" "$root/parity/cpu.csv" unused parity
    build/rps_gpu_probe "$checkpoint" "$root/parity/gpu.csv" "$hidden" "$layers" "$round_obs"
    python3 scripts/rps_analysis.py parity "$root/parity"
}

case "$command" in
    train) train "$@"; exit 0 ;;
    run)
        run_id="bot_${bot}_seed_73"
        train --env.bot_policy="$bot" --base.seed=73 --base.run_id="$run_id" \
            --base.load_model_path=None --train.learning_rate="$learning_rate" \
            --train.total_timesteps="$steps"
        mapfile -t model_info < <(python3 - "$root" "$run_id" <<'PYCODE'
import configparser, pathlib, sys
root, run_id = pathlib.Path(sys.argv[1]), sys.argv[2]
c = configparser.ConfigParser(interpolation=None)
c.read(root/'logs/rock_paper_scissors'/f'{run_id}.ini')
checkpoint = max((root/'checkpoints/rock_paper_scissors'/run_id).glob('*.bin'),
                 key=lambda p: int(p.stem))
print(checkpoint)
print(c.getint('policy', 'hidden_size'))
print(c.getint('policy', 'num_layers'))
print(c.getint('env', 'round_observation'))
PYCODE
        )
        [[ ${#model_info[@]} == 4 ]]
        checkpoint=${model_info[0]}; hidden=${model_info[1]}; layers=${model_info[2]}; round_obs=${model_info[3]}
        ;;
esac
compile_cpu_probe
if [[ "$command" != parity ]]; then
    mkdir -p "$root/evaluation"
    build/rps_policy_probe "$checkpoint" "$root/evaluation/trajectories.csv" "$root/evaluation/probes.csv"
    python3 scripts/rps_analysis.py summary "$root/evaluation" > "$root/evaluation/summary_stdout.json"
    read -r trajectory_hash _ < <(sha256sum "$root/evaluation/trajectories.csv")
    python3 scripts/rps_analysis.py adaptation "$root/evaluation/trajectories.csv" "$trajectory_hash" \
        --output "$root/evaluation" > "$root/evaluation/adaptation_stdout.json"
fi
parity
{
    printf 'checkpoint=%s\nhidden_size=%s\nnum_layers=%s\n' "$checkpoint" "$hidden" "$layers"
    printf 'obs_size=%s\nround_observation=%s\n' "$obs_size" "$round_obs"
    printf 'evaluation_seed_base=20260926\nmatches_per_opponent=64\nrounds_per_match=1000\n'
    if [[ "$command" == run ]]; then
        printf 'bot_policy=%s\ntraining_seed=73\nrequested_timesteps=%s\nlearning_rate=%s\n' \
            "$bot" "$steps" "$learning_rate"
    fi
    sha256sum "$checkpoint" config/default.ini config/rock_paper_scissors.ini \
        ocean/rock_paper_scissors/rock_paper_scissors.h scripts/rps.sh \
        scripts/rps_analysis.py scripts/rps_policy_probe.c scripts/rps_gpu_probe.cu \
        src/puffercpu.c src/algo.cu
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    gcc --version | head -1
} > "$root/provenance.txt"
if [[ "$command" != parity ]]; then
    python3 - "$root/evaluation/summary.json" <<'PYCODE'
import json, sys
s = json.load(open(sys.argv[1]))
for name in ('bot_uniform', 'bot_counter', 'bot_rock'):
    print(name, 'mean reward:', s['groups'][name]['mean_reward'],
          '95% interval:', s['match_reward_ci95'][name])
PYCODE
fi
