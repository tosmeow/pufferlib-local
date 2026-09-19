#!/usr/bin/env python3
"""Measure price transmission from a constant directional alpha.

Every replica starts from the same initial book, submits only Noop, and holds
raw alpha fixed for the complete experiment.  Public power-law impact is
disabled so the only exogenous directional input is::

    intensity_bias = alpha_scale * constant_alpha

For each scale, the output stores every replica's terminal-minus-initial
mid-price change plus its mean and standard error.
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


TIME_REMAINING_OBSERVATION_INDEX = 25


def _float_view(address: int, count: int) -> np.ndarray:
    raw = (ctypes.c_float * count).from_address(address)
    return np.ctypeslib.as_array(raw)


def summarize(drifts: np.ndarray) -> dict:
    drifts = np.asarray(drifts, dtype=np.float64)
    if drifts.ndim != 1 or drifts.size < 2 or not np.all(np.isfinite(drifts)):
        raise ValueError("drifts must be a finite one-dimensional sample")
    return {
        "mean_drift": float(drifts.mean()),
        "standard_error": float(drifts.std(ddof=1) / math.sqrt(drifts.size)),
        "sample_standard_deviation": float(drifts.std(ddof=1)),
        "minimum": float(drifts.min()),
        "q05": float(np.quantile(drifts, 0.05)),
        "median": float(np.median(drifts)),
        "q95": float(np.quantile(drifts, 0.95)),
        "maximum": float(drifts.max()),
    }


def run_scale(
    *,
    alpha_scale: float,
    constant_alpha: float,
    horizon: float,
    agents: int,
    max_steps: int,
) -> dict:
    episode_duration = horizon + 60.0
    with mock.patch.object(sys, "argv", [sys.argv[0]]):
        config = load_config("queue_reactive")
    config["vec"].update(total_agents=agents, num_buffers=1, num_threads=1)
    config["env"].update(
        use_alpha=0,
        use_constant_alpha=1,
        constant_alpha=constant_alpha,
        alpha_scale=alpha_scale,
        use_power_law_impact=0,
        strategy_impact=0,
        episode_duration_seconds=episode_duration,
    )

    vec = _C.create_vec(config, 0)
    if getattr(vec, "diagnostic_size", 0) != 1:
        vec.close()
        raise RuntimeError(
            "Rebuild queue_reactive: the constant-alpha benchmark requires "
            "the native mid-price diagnostic"
        )

    observations = _float_view(
        vec.obs_ptr, agents * vec.obs_size).reshape(agents, vec.obs_size)
    actions = np.zeros((agents, vec.num_atns), dtype=np.float32)
    initial_mid = np.empty(agents, dtype=np.float32)
    current_mid = np.empty(agents, dtype=np.float32)
    final_mid = np.full(agents, np.nan, dtype=np.float32)

    try:
        vec.reset()
        vec.diagnostic(initial_mid.ctypes.data)
        for _ in range(max_steps):
            vec.cpu_step(actions.ctypes.data)
            vec.diagnostic(current_mid.ctypes.data)
            simulated_time = episode_duration * (
                1.0 - observations[:, TIME_REMAINING_OBSERVATION_INDEX]
            )
            reached = np.isnan(final_mid) & (simulated_time >= horizon)
            final_mid[reached] = current_mid[reached]
            if np.all(np.isfinite(final_mid)):
                break
        else:
            missing = int(np.count_nonzero(~np.isfinite(final_mid)))
            raise RuntimeError(
                f"{missing}/{agents} replicas did not reach T={horizon:g}s "
                f"within {max_steps} steps; increase --max-steps"
            )
    finally:
        vec.close()

    drifts = (final_mid - initial_mid).astype(np.float64)
    return {
        "alpha_scale": alpha_scale,
        "effective_bias_magnitude": abs(alpha_scale * constant_alpha),
        "n": agents,
        **summarize(drifts),
        "sequence_drifts": drifts.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scales", type=float, nargs="+",
        default=[0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        help="Nonnegative multipliers for the fixed alpha",
    )
    parser.add_argument(
        "--constant-alpha", type=float, default=1.0,
        help="Fixed raw alpha; its sign selects the direction",
    )
    parser.add_argument("--horizon", type=float, default=10.0)
    parser.add_argument("--agents", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--output", type=Path,
        default=Path("benchmarks/queue_reactive/constant_alpha_drift.json"),
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
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.constant_alpha == 0:
        raise ValueError("--constant-alpha must select a nonzero direction")
    if any(scale < 0 for scale in args.scales):
        raise ValueError("--scales must be nonnegative")

    started = time.perf_counter()
    results = []
    for scale in args.scales:
        print(f"constant_alpha={args.constant_alpha:g} scale={scale:g}", flush=True)
        results.append(run_scale(
            alpha_scale=scale,
            constant_alpha=args.constant_alpha,
            horizon=args.horizon,
            agents=args.agents,
            max_steps=args.max_steps,
        ))

    payload = {
        "benchmark": "queue_reactive_constant_alpha_drift",
        "constant_alpha": args.constant_alpha,
        "horizon_seconds": args.horizon,
        "agents_per_scale": args.agents,
        "agent_action": "Noop",
        "power_law_impact": False,
        "drift_definition": "final_mid_price - initial_mid_price",
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
