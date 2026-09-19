"""Train one PufferLib policy per lag/seed and evaluate on a common game.

Run from the repo root with .venv/bin/python scripts/compare_lagged_breakout.py
--help. Checkpoints use the PyTorch backend (--cpu or --float native build).
"""

import argparse
import copy
import csv
import ctypes
import json
import math
from pathlib import Path
import random
import sys
import time
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def float_view(pointer, shape):
    size = math.prod(shape)
    return np.ctypeslib.as_array((ctypes.c_float * size).from_address(pointer)).reshape(shape)


def write_json(destination, payload):
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "standard_error": float(values.std(ddof=1) / math.sqrt(len(values)))
        if len(values) > 1 else None,
    }


def train_policy(config, destination):
    from pufferlib.torch_pufferl import PuffeRL

    destination.mkdir(parents=True, exist_ok=False)
    seed = config["train"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    write_json(destination / "config.json", config)
    trainer = PuffeRL.create_pufferl(copy.deepcopy(config))
    started = last_report = time.monotonic()
    try:
        with (destination / "training.jsonl").open("w") as log_file:
            while trainer.global_step < config["train"]["total_timesteps"]:
                trainer.rollouts()
                trainer.train()
                if time.monotonic() - last_report >= 5 or trainer.global_step >= config["train"]["total_timesteps"]:
                    report = trainer.log()
                    log_file.write(json.dumps(report) + "\n")
                    log_file.flush()
                    print(f"{destination.name}: {trainer.global_step:,} steps", flush=True)
                    last_report = time.monotonic()
            trainer.save_weights(str(destination / "policy.bin"))
            write_json(destination / "training_result.json", {
                "steps": trainer.global_step,
                "seconds": time.monotonic() - started,
            })
    finally:
        trainer.close()


def evaluate_policy(config, checkpoint, args):
    from pufferlib import _C
    from pufferlib.torch_pufferl import load_policy

    config = copy.deepcopy(config)
    config["vec"].update(total_agents=args.eval_episodes, num_buffers=1, num_threads=1)
    config["env"].update(
        action_lag=args.eval_lag, volatility=args.eval_volatility,
        seed=args.eval_seed, reward_interval=args.eval_reward_interval,
    )
    config["load_model_path"] = str(checkpoint)
    config["load_id"] = None
    # Evaluate all policies on CPU, with deterministic argmax actions. Each slot
    # contributes its first episode, avoiding bias toward shorter episodes.
    vec = _C.create_vec(config, 0)
    n = args.eval_episodes
    try:
        observations = float_view(vec.obs_ptr, (n, vec.obs_size))
        rewards = float_view(vec.rewards_ptr, (n,))
        terminals = float_view(vec.terminals_ptr, (n,))
        policy = load_policy(config, vec, device=torch.device("cpu")).eval()
        state = policy.initial_state(n, device=torch.device("cpu"))
        vec.reset()
        done = np.zeros(n, dtype=bool)
        returns = np.zeros(n, dtype=np.float64)
        lengths = np.zeros(n, dtype=np.int64)
        switches = np.zeros(n, dtype=np.int64)
        reversals = np.zeros(n, dtype=np.int64)
        noops = np.zeros(n, dtype=np.int64)
        distances = np.zeros(n, dtype=np.float64)
        previous = np.zeros(n, dtype=np.int64)
        trace = []
        with torch.inference_mode():
            for step in range(args.max_eval_steps):
                logits, _, state = policy.forward_eval(torch.from_numpy(observations), state)
                commands = logits.argmax(dim=-1).numpy()
                actions = np.ascontiguousarray(commands[:, None], dtype=np.float32)
                active = ~done
                if not done[0] and step < args.trace_steps:
                    trace.append({
                        "step": step,
                        "time_seconds": step * config["env"]["frameskip"] / 60.0,
                        "paddle_x": float(observations[0, 0] * 576),
                        "ball_x": float(observations[0, 2] * 576),
                        "ball_y": float(observations[0, 3] * 330),
                        "action": int(commands[0]),
                    })
                paddle_x = observations[:, 0].copy()
                balls = observations[:, 8].copy()
                switches += active & (lengths > 0) & (commands != previous)
                reversals += active & (lengths > 0) & (commands != previous) & (commands != 0) & (previous != 0)
                noops += active & (commands == 0)
                previous = commands.copy()
                lengths += active
                vec.cpu_step(actions.ctypes.data)
                returns += np.where(active, rewards, 0)
                continuous_round = active & (terminals == 0) & (balls == observations[:, 8])
                distances += np.where(continuous_round, np.abs(observations[:, 0] - paddle_x) * 576, 0)
                done |= terminals != 0
                if done.all():
                    break
        episodes = []
        for i in range(n):
            episodes.append({
                "environment_seed": args.eval_seed + i,
                "completed": bool(done[i]),
                "return": float(returns[i]),
                "decision_steps": int(lengths[i]),
                "action_switch_rate": float(switches[i] / max(1, lengths[i] - 1)),
                "direction_reversal_rate": float(reversals[i] / max(1, lengths[i] - 1)),
                "noop_fraction": float(noops[i] / max(1, lengths[i])),
                "paddle_distance_pixels": float(distances[i]),
            })
        complete = [episode for episode in episodes if episode["completed"]]
        # Capped episodes are retained in the raw output, never presented as
        # completed returns. Report the cap explicitly to reveal censoring.
        aggregates = {
            metric: summary([episode[metric] for episode in complete])
            for metric in ("return", "decision_steps", "action_switch_rate",
                           "direction_reversal_rate", "noop_fraction", "paddle_distance_pixels")
        } if complete else {}
        return {
            "completed_episodes": len(complete),
            "capped_episodes": n - len(complete),
            "summary": aggregates,
            "episodes": episodes,
            "trace": trace,
        }
    finally:
        vec.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Evaluate all saved lag_*/seed_*/policy.bin runs under --output")
    parser.add_argument("--lags", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--volatility", type=float, default=6.0)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--agents", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--reward-interval", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=100_000)
    parser.add_argument("--eval-lag", type=int, default=0)
    parser.add_argument("--eval-volatility", type=float, default=0.0)
    parser.add_argument("--eval-reward-interval", type=int, default=4)
    parser.add_argument("--max-eval-steps", type=int, default=30_000)
    parser.add_argument("--trace-steps", type=int, default=600)
    args = parser.parse_args()
    if any(lag < 0 or lag > 64 for lag in args.lags + [args.eval_lag]):
        parser.error("lags must be in [0, 64]")
    if any(not math.isfinite(sigma) or sigma < 0 for sigma in (args.volatility, args.eval_volatility)):
        parser.error("volatility must be finite and nonnegative")
    if len(set(args.lags)) != len(args.lags) or len(set(args.seeds)) != len(args.seeds):
        parser.error("lags and seeds must not contain duplicates")
    if any(seed < 0 or seed > 2**31 - 1 for seed in args.seeds + [args.eval_seed]):
        parser.error("seeds must be in [0, 2**31 - 1]")
    if any(value <= 0 for value in (args.agents, args.horizon, args.timesteps,
                                    args.minibatch_size, args.eval_episodes,
                                    args.max_eval_steps, args.torch_threads)):
        parser.error("training sizes, threads, and evaluation limits must be positive")
    if args.trace_steps < 0 or not 1 <= args.frameskip <= 64:
        parser.error("trace-steps must be nonnegative and frameskip must be in [1, 64]")
    for interval in (args.reward_interval, args.eval_reward_interval):
        if interval < args.frameskip or interval % args.frameskip:
            parser.error("reward intervals must be positive multiples of frameskip")
    if args.minibatch_size % args.horizon or args.minibatch_size > args.agents * args.horizon:
        parser.error("minibatch-size must be a multiple of horizon and <= agents * horizon")
    return args


def main():
    args = parse_args()
    from pufferlib import _C
    from pufferlib.pufferl import load_config

    if _C.env_name != "lagged_breakout" or _C.precision_bytes != 4:
        raise RuntimeError("Build lagged_breakout with --cpu or --float before running this experiment")
    torch.set_num_threads(args.torch_threads)
    args.output = args.output.resolve()
    if args.evaluate_only:
        checkpoints = sorted(args.output.glob("lag_*/seed_*/policy.bin"))
        if not checkpoints:
            raise FileNotFoundError(f"No lag/seed checkpoints in {args.output}")
    else:
        with mock.patch.object(sys, "argv", [sys.argv[0]]):
            base = load_config("lagged_breakout")
        base["slowly"] = True
        base["torch"]["device"] = args.device
        base["vec"].update(total_agents=args.agents, num_buffers=1, num_threads=1)
        base["env"].update(volatility=args.volatility, frameskip=args.frameskip,
                            reward_interval=args.reward_interval)
        base["train"].update(total_timesteps=args.timesteps, horizon=args.horizon,
                              minibatch_size=args.minibatch_size)
        destinations = [args.output / f"lag_{lag}" / f"seed_{seed}"
                        for lag in args.lags for seed in args.seeds]
        if any(destination.exists() for destination in destinations):
            raise FileExistsError("A requested run already exists; use a fresh output or --evaluate-only")
        checkpoints = []
        for lag in args.lags:
            for seed in args.seeds:
                config = copy.deepcopy(base)
                config["seed"] = config["train"]["seed"] = seed
                config["env"].update(action_lag=lag, seed=seed)
                destination = args.output / f"lag_{lag}" / f"seed_{seed}"
                print(f"Training lag={lag}, volatility={args.volatility}, seed={seed}", flush=True)
                train_policy(config, destination)
                checkpoints.append(destination / "policy.bin")

    evaluation_dir = args.output / "evaluations" / f"evaluation_{time.time_ns()}"
    evaluation_dir.mkdir(parents=True)
    results, rows, traces = [], [], []
    for checkpoint in checkpoints:
        config = json.loads((checkpoint.parent / "config.json").read_text())
        if config["env"]["frameskip"] != args.frameskip:
            raise ValueError("Evaluation frameskip must match every saved training configuration")
        label = {"train_lag": config["env"]["action_lag"],
                 "train_volatility": config["env"]["volatility"],
                 "train_seed": config["train"]["seed"]}
        print(f"Evaluating {label} on lag={args.eval_lag}, volatility={args.eval_volatility}", flush=True)
        result = evaluate_policy(config, checkpoint, args)
        traces.extend({**label, **point} for point in result.pop("trace"))
        rows.extend({**label, **episode} for episode in result["episodes"])
        results.append({**label, "checkpoint": str(checkpoint), **result})
        print(json.dumps(result["summary"]), flush=True)
        if result["capped_episodes"]:
            print(f"{result['capped_episodes']} episodes reached the step cap; increase --max-eval-steps", flush=True)
        write_json(evaluation_dir / "results.json", {
            "evaluation": {"lag": args.eval_lag, "volatility": args.eval_volatility,
                           "seed": args.eval_seed, "episodes_per_policy": args.eval_episodes,
                           "frameskip": args.frameskip, "reward_interval": args.eval_reward_interval,
                           "max_steps": args.max_eval_steps, "action_selection": "argmax"},
            "results": results,
        })
    for filename, records in (("episodes.csv", rows), ("trajectories.csv", traces)):
        if records:
            with (evaluation_dir / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    print(f"Saved comparison to {evaluation_dir}")


if __name__ == "__main__":
    main()
