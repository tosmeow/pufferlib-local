#!/usr/bin/env python3
"""Measure how queue-reactive alpha intensity scaling reaches the mid-price.

The benchmark submits Noop for every environment.  For each replica it records
the unscaled OU alpha after a burn-in, then measures the mid-price change over
one or more simulated-time horizons.  Since the OU alpha is symmetric around
zero, the unconditional mean drift should be close to zero.  The informative
quantity is the alpha-aligned drift::

    sign(alpha_at_anchor) * (mid_at_T - mid_at_anchor)

Run after building the native CPU environment::

    puffer build queue_reactive --cpu
    python scripts/benchmark_queue_alpha_drift.py \
        --output /tmp/queue_alpha_drift.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np

from pufferlib import _C
from pufferlib.pufferl import load_config


ALPHA_OBSERVATION_INDEX = 21
TIME_REMAINING_OBSERVATION_INDEX = 25


def _float_view(address: int, count: int) -> np.ndarray:
    raw = (ctypes.c_float * count).from_address(address)
    return np.ctypeslib.as_array(raw)


def _summary(alpha: np.ndarray, delta: np.ndarray, threshold: float) -> dict:
    finite = np.isfinite(alpha) & np.isfinite(delta)
    unconditional = delta[finite]
    selected = finite & (np.abs(alpha) >= threshold)
    selected_alpha = alpha[selected].astype(np.float64)
    selected_delta = delta[selected].astype(np.float64)
    aligned = np.sign(selected_alpha) * selected_delta

    if aligned.size:
        aligned_mean = float(aligned.mean())
        aligned_se = (
            float(aligned.std(ddof=1) / math.sqrt(aligned.size))
            if aligned.size > 1 else 0.0
        )
        alpha_delta_slope = float(
            np.dot(selected_alpha, selected_delta)
            / np.dot(selected_alpha, selected_alpha)
        )
    else:
        aligned_mean = None
        aligned_se = None
        alpha_delta_slope = None

    return {
        "n": int(aligned.size),
        "alpha_aligned_drift": aligned_mean,
        "standard_error": aligned_se,
        "unconditional_drift": (
            float(unconditional.mean()) if unconditional.size else None
        ),
        "alpha_delta_slope": alpha_delta_slope,
    }


def run_configuration(
    *,
    alpha_scale: float,
    use_power_law_impact: bool,
    agents: int,
    burn_in: float,
    horizons: list[float],
    threshold: float,
    max_steps: int,
) -> list[dict]:
    episode_duration = burn_in + max(horizons) + 60.0
    with mock.patch.object(sys, "argv", [sys.argv[0]]):
        config = load_config("queue_reactive")
    config["vec"].update(total_agents=agents, num_buffers=1, num_threads=1)
    config["env"].update(
        alpha_scale=alpha_scale,
        use_alpha=1,
        use_power_law_impact=int(use_power_law_impact),
        strategy_impact=0,
        episode_duration_seconds=episode_duration,
    )

    vec = _C.create_vec(config, 0)
    if vec.obs_size <= TIME_REMAINING_OBSERVATION_INDEX:
        vec.close()
        raise RuntimeError("queue_reactive observation ABI is too small")
    if getattr(vec, "diagnostic_size", 0) != 1:
        vec.close()
        raise RuntimeError(
            "Rebuild queue_reactive: this benchmark requires the mid-price "
            "diagnostic added to the native CPU binding"
        )

    observations = _float_view(
        vec.obs_ptr, agents * vec.obs_size).reshape(agents, vec.obs_size)
    actions = np.zeros((agents, vec.num_atns), dtype=np.float32)
    mid = np.empty(agents, dtype=np.float32)
    anchor_time = np.full(agents, np.nan, dtype=np.float64)
    anchor_alpha = np.full(agents, np.nan, dtype=np.float64)
    anchor_mid = np.full(agents, np.nan, dtype=np.float64)
    deltas = np.full((len(horizons), agents), np.nan, dtype=np.float64)

    try:
        vec.reset()
        for _ in range(max_steps):
            vec.cpu_step(actions.ctypes.data)
            vec.diagnostic(mid.ctypes.data)

            simulated_time = episode_duration * (
                1.0 - observations[:, TIME_REMAINING_OBSERVATION_INDEX]
            )
            new_anchors = np.isnan(anchor_time) & (simulated_time >= burn_in)
            anchor_time[new_anchors] = simulated_time[new_anchors]
            anchor_alpha[new_anchors] = observations[
                new_anchors, ALPHA_OBSERVATION_INDEX]
            anchor_mid[new_anchors] = mid[new_anchors]

            for horizon_idx, horizon in enumerate(horizons):
                due = (
                    np.isfinite(anchor_time)
                    & np.isnan(deltas[horizon_idx])
                    & (simulated_time >= anchor_time + horizon)
                )
                deltas[horizon_idx, due] = mid[due] - anchor_mid[due]

            if np.all(np.isfinite(deltas)):
                break
        else:
            missing = np.count_nonzero(~np.isfinite(deltas[-1]))
            raise RuntimeError(
                f"{missing}/{agents} replicas did not reach the longest "
                f"horizon within {max_steps} steps; increase --max-steps"
            )
    finally:
        vec.close()

    rows = []
    for horizon, delta in zip(horizons, deltas):
        rows.append({
            "alpha_scale": alpha_scale,
            "power_law_impact": bool(use_power_law_impact),
            "horizon_seconds": horizon,
            **_summary(anchor_alpha, delta, threshold),
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scales", type=float, nargs="+",
        default=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
        help="Alpha intensity scales to compare",
    )
    parser.add_argument(
        "--horizons", type=float, nargs="+", default=[1.0, 5.0, 10.0],
        help="Forward simulated-time horizons in seconds",
    )
    parser.add_argument("--agents", type=int, default=2048)
    parser.add_argument("--burn-in", type=float, default=60.0)
    parser.add_argument(
        "--alpha-threshold", type=float, default=0.25,
        help="Only anchors with |alpha| at least this value enter aligned means",
    )
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument(
        "--impact-modes", choices=("off", "on"), nargs="+",
        default=["off", "on"],
        help="Compare isolated alpha and the current public-impact process",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("benchmarks/queue_reactive/alpha_drift_results.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(_C, "env_name", None) != "queue_reactive":
        raise RuntimeError(
            "The loaded native backend is not queue_reactive; run "
            "`puffer build queue_reactive --cpu` first"
        )
    if args.agents < 2:
        raise ValueError("--agents must be at least 2")
    if any(horizon <= 0 for horizon in args.horizons):
        raise ValueError("--horizons must all be positive")

    started = time.perf_counter()
    rows = []
    for impact_mode in args.impact_modes:
        for alpha_scale in args.scales:
            print(
                f"impact={impact_mode:>3} alpha_scale={alpha_scale:g}",
                flush=True,
            )
            rows.extend(run_configuration(
                alpha_scale=alpha_scale,
                use_power_law_impact=impact_mode == "on",
                agents=args.agents,
                burn_in=args.burn_in,
                horizons=args.horizons,
                threshold=args.alpha_threshold,
                max_steps=args.max_steps,
            ))

    payload = {
        "benchmark": "queue_reactive_no_agent_alpha_drift",
        "definition": (
            "sign(alpha_at_anchor) * "
            "(mid_at_anchor_plus_T - mid_at_anchor)"
        ),
        "agents_per_configuration": args.agents,
        "burn_in_seconds": args.burn_in,
        "alpha_threshold": args.alpha_threshold,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
