#!/usr/bin/env bash
set -euo pipefail

# Run from the project root, through Slurm on this cluster. This compiles only
# the small environment checks, not the CUDA trainer, and performs no training.
# Raylib headers are needed by pufferenv.h; no graphics library is linked.
test_output_dir="${CLUSTER_RESULTS_DIR:-build/nash_impact_checks}"
mkdir -p "$test_output_dir"
test_source="ocean/nash_impact/tests/test_nash_impact.c"
test_includes=(-Isrc -Iraylib-5.5_linux_amd64/include)
test_flags=(-O1 -g -Wall -Wextra -Wno-unused-function -Wno-missing-field-initializers
    -fsanitize=undefined -fno-sanitize-recover=all)

gcc -std=c11 -D_POSIX_C_SOURCE=200809L "${test_flags[@]}" "${test_includes[@]}" \
    "$test_source" -lm -o "$test_output_dir/test_nash_impact_c"
"$test_output_dir/test_nash_impact_c"
g++ -std=c++17 "${test_flags[@]}" "${test_includes[@]}" -x c++ \
    "$test_source" -lm -o "$test_output_dir/test_nash_impact_cpp"
"$test_output_dir/test_nash_impact_cpp"

for invalid_case in invalid_dates invalid_action invalid_population; do
    if "$test_output_dir/test_nash_impact_c" "$invalid_case"; then
        echo "Expected rejection of $invalid_case" >&2
        exit 1
    fi
done
echo "nash_impact: invalid inputs rejected"
