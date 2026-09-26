#!/usr/bin/env bash
set -euo pipefail
checkpoint=/cluster/users/tosma/projects/pufferlib-local/jobs/a06fc0ee-da58-4b12-8de4-575f32d5dc10/artifacts/checkpoints/rock_paper_scissors/1790433943269/0000001999962112.bin
expected=6ca507f3f12ac323e180df5498ea00dc68ef3c0a9765c6fc66adcf716c1c31ed
printf '%s  %s\n' "$expected" "$checkpoint" | sha256sum --check
mkdir -p build
gcc -O2 -std=c11 -D_POSIX_C_SOURCE=200809L scripts/rps_policy_probe.c -lm -o build/rps_policy_probe
build/rps_policy_probe "$checkpoint" "$CLUSTER_RESULTS_DIR/trajectories.csv" "$CLUSTER_RESULTS_DIR/probes.csv"
python3 scripts/rps_policy_summary.py "$CLUSTER_RESULTS_DIR"
{
    printf 'source_job=858\nsource_snapshot=5a2e8ea3104d3e7dd706b373e06f73cb8d12ff26\ncheckpoint=%s\ncheckpoint_sha256=%s\n' "$checkpoint" "$expected"
    printf 'network=default CPU PufferNet, float32, obs=6, hidden=16, layers=4, actions=3\nseed_base=20260926\nmatches_per_opponent=64\nrounds_per_match=1000\n'
    gcc --version | head -1
    sha256sum src/puffercpu.c src/algo.cu scripts/rps_policy_probe.c scripts/rps_policy_summary.py
} > "$CLUSTER_RESULTS_DIR/provenance.txt"
