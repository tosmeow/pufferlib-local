#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVCC_ARCH=sm_120
checkpoint=/cluster/users/tosma/projects/pufferlib-local/jobs/a06fc0ee-da58-4b12-8de4-575f32d5dc10/artifacts/checkpoints/rock_paper_scissors/1790433943269/0000001999962112.bin
printf '%s  %s\n' 6ca507f3f12ac323e180df5498ea00dc68ef3c0a9765c6fc66adcf716c1c31ed "$checkpoint" | sha256sum --check
# Also validate the normal optimized build and prepare its raylib dependency.
./build.sh rock_paper_scissors build/rps_eval_build --float
nvcc -O2 -arch=sm_120 -std=c++17 -DPRECISION_FLOAT -I. -Isrc -Ivendor \
    -Iraylib-5.5_linux_amd64/include -I"$CUDA_HOME/include/cccl" \
    -Xcompiler=-fopenmp -Xcompiler=-Wno-narrowing --diag-suppress=2361 \
    scripts/rps_gpu_probe.cu raylib-5.5_linux_amd64/lib/libraylib.a \
    -L"$CUDA_HOME/lib64" -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
    -lm -Xlinker=-lpthread -lGL -o build/rps_gpu_probe
gcc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L scripts/rps_policy_probe.c -lm -o build/rps_policy_probe
build/rps_policy_probe "$checkpoint" "$CLUSTER_RESULTS_DIR/cpu.csv" unused parity
build/rps_gpu_probe "$checkpoint" "$CLUSTER_RESULTS_DIR/gpu.csv"
python3 - "$CLUSTER_RESULTS_DIR" <<'PY'
import csv,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
def read(name):
    with (root/name).open() as f:
        return {(r['agent'],r['step']): [float(r['p_'+a]) for a in ['rock','paper','scissors']] for r in csv.DictReader(f)}
cpu,gpu=read('cpu.csv'),read('gpu.csv')
assert cpu.keys()==gpu.keys() and len(cpu)==10000
error=max(abs(x-y) for k in cpu for x,y in zip(cpu[k],gpu[k]))
result=dict(rows=len(cpu),max_absolute_probability_error=error,tolerance=1e-5,passed=error<1e-5,
    gpu_first_move=gpu[('0','0')],gpu_second_move=gpu[('0','1')])
(root/'parity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
assert result['passed']
PY
