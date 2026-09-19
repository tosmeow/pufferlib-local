"""Continue saved lag/seed policies with periodic validation and resumable training.

Select checkpoints using the original training lag and volatility. Evaluate the
selected checkpoint on separate seeds in the common, unlagged deterministic game.
"""

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_lagged_breakout import evaluate_policy


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def evaluation(config, checkpoint, episodes, seed, lag, volatility):
    options = argparse.Namespace(
        eval_episodes=episodes, eval_seed=seed, eval_lag=lag,
        eval_volatility=volatility, eval_reward_interval=config["env"]["reward_interval"],
        max_eval_steps=30000, trace_steps=0,
    )
    # Initializing an evaluation model must not perturb the training RNG stream.
    with torch.random.fork_rng():
        result = evaluate_policy(config, checkpoint, options)
    result.pop("trace", None)
    result["lag"] = lag
    result["volatility"] = volatility
    result["seed"] = seed
    result["max_steps"] = options.max_eval_steps
    result["win_rate"] = sum(row["completed"] and row["return"] == 864 for row in result["episodes"]) / episodes
    # Include capped episodes as their earned return for checkpoint ranking.
    # Keep the censoring counts visible; no incomplete episode is called solved.
    result["mean_return_including_capped"] = float(np.mean([row["return"] for row in result["episodes"]]))
    return result


def train_run(source, destination, options):
    from pufferlib import _C
    from pufferlib.torch_pufferl import PuffeRL, load_policy

    status_file = destination / "status.json"
    if status_file.exists():
        previous = json.loads(status_file.read_text())
        if (not options.get("continue_after_target")
                and previous.get("status") == "target_reached_on_validation"
                and previous.get("target_return", 0) >= options["target_return"]
                and previous.get("holdout_training_return", 0) >= options["target_return"]):
            print(f"ALREADY COMPLETE lag={previous['lag']} seed={previous['seed']}", flush=True)
            return previous
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if _C.env_name != "lagged_breakout" or _C.precision_bytes != 4:
        raise RuntimeError("Build lagged_breakout with --cpu or --float first")
    destination.mkdir(parents=True, exist_ok=True)
    validation_dir = destination / "validation"
    history_dir = destination / "history"
    validation_dir.mkdir(exist_ok=True)
    history_dir.mkdir(exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    source_encoder = config["torch"]["encoder"]
    previous_config = json.loads((destination / "config.json").read_text()) if (destination / "config.json").exists() else None
    source_steps = json.loads((source / "training_result.json").read_text())["steps"]
    device = options.get("device", "cpu")
    config["torch"]["device"] = device
    config["vec"].update(num_buffers=1, num_threads=1)
    if options.get("agents") is not None:
        config["vec"]["total_agents"] = options["agents"]
    if options.get("minibatch_size") is not None:
        config["train"]["minibatch_size"] = options["minibatch_size"]
    config["train"].update(
        total_timesteps=options["additional_steps"],
        learning_rate=options["learning_rate"],
        min_lr_ratio=0.1,
    )
    if options.get("gamma") is not None:
        config["train"]["gamma"] = options["gamma"]
    elif previous_config is not None:
        config["train"]["gamma"] = previous_config["train"]["gamma"]
    if options.get("anneal_steps") is not None:
        config["train"]["anneal_lr"] = 0  # The continuation uses a step-based, clamped schedule below.
    config["load_model_path"] = str(source / "policy.bin")
    config["load_id"] = None
    if options.get("normalize_velocity"):
        config["torch"]["encoder"] = "LaggedBreakoutEncoder"
    config["continuation"] = {
        "source": str(source), "source_steps": source_steps,
        "initial_optimizer": "fresh: source checkpoint contains weights only",
        "resume_environment": "new episodes; native environment state is not serialized",
        "options": options,
    }
    prior_continuation = previous_config.get("continuation", {}) if previous_config else {}
    config["continuation"]["warm_start_source"] = (
        options.get("warm_start") or prior_continuation.get("warm_start_source")
        or prior_continuation.get("options", {}).get("warm_start")
    )
    lag = config["env"]["action_lag"]
    seed = config["train"]["seed"]
    label = f"lag={lag} seed={seed}"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    checkpoint = destination / "training_state.pt"
    saved = None
    if checkpoint.exists():
        # This file is created locally by this script and contains RNG state.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if options.get("restart_from_best") or options.get("warm_start"):
            if options.get("restart_from_best") and not (destination / "policy.bin").is_file():
                raise FileNotFoundError("Restarting from best requires an existing policy.bin")
            atomic_save(history_dir / f"training_state_before_restart_{time.time_ns()}.pt", saved)
        config["load_model_path"] = None
        saved_encoder = saved.get("encoder", previous_config["torch"]["encoder"] if previous_config else source_encoder)
        if saved_encoder == "LaggedBreakoutEncoder":
            config["torch"]["encoder"] = saved_encoder
    if (destination / "config.json").exists():
        atomic_json(history_dir / f"config_before_resume_{time.time_ns()}.json",
                    json.loads((destination / "config.json").read_text()))
    atomic_json(destination / "config.json", config)
    vec = _C.create_vec(config, 0)
    try:
        policy = load_policy(config, vec, device=torch.device(device))
        trainer = PuffeRL(config, vec, policy, verbose=False, device=device)
    except BaseException:
        vec.close()
        raise
    best_score = -math.inf
    target_streak = 0
    history = []
    prior_seconds = 0.0
    if saved is not None:
        trainer.policy.load_state_dict(saved["policy"])
        trainer.optimizer.load_state_dict(saved["optimizer"])
        trainer.global_step = saved["global_step"]
        # Annealing follows sampled steps when resuming with a different batch.
        trainer.epoch = saved["global_step"] // trainer.batch_size
        trainer.last_log_step = trainer.global_step
        torch.set_rng_state(saved["torch_rng"])
        if device == "mps" and saved.get("mps_rng") is not None:
            torch.mps.set_rng_state(saved["mps_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        best_score = saved["best_score"]
        target_streak = saved["target_streak"]
        history = saved["history"]
        prior_seconds = saved["seconds"]
    loaded_encoder = (saved.get("encoder", previous_config["torch"]["encoder"] if previous_config else source_encoder)
                      if saved is not None else source_encoder)
    if config["torch"]["encoder"] == "LaggedBreakoutEncoder" and loaded_encoder == "DefaultEncoder":
        with torch.no_grad():
            trainer.policy.encoder.encoder.weight[:, 4:6].div_(60.0)
        # Momentum is expressed in the old coordinates. Restart it after the
        # invertible reparameterization, retaining the complete policy function.
        trainer.optimizer.state.clear()
        for filename in ("policy.bin", "latest.bin"):
            path = destination / filename
            if path.exists():
                weights = torch.load(path, map_location="cpu", weights_only=True)
                atomic_save(history_dir / path.with_suffix(".before_velocity_normalization.bin").name, weights)
                weights["encoder.encoder.weight"][:, 4:6] /= 60.0
                atomic_save(path, weights)
        print(f"NORMALIZED VELOCITY {label}: policy function preserved; optimizer momentum restarted", flush=True)
    if options.get("restart_from_best") and saved is not None:
        trainer.policy.load_state_dict(torch.load(destination / "policy.bin", map_location=device,
                                                  weights_only=True))
        trainer.optimizer.state.clear()
        target_streak = 0
        print(f"RESTART FROM BEST {label}: training counters retained; optimizer momentum restarted", flush=True)
    if options.get("warm_start"):
        warm_start = Path(options["warm_start"])
        warm_config = json.loads((warm_start.parent / "config.json").read_text())
        if any(warm_config["torch"][field] != config["torch"][field]
               for field in ("encoder", "network", "decoder")):
            raise ValueError("Warm-start checkpoint must use the same encoder, network, and decoder")
        trainer.policy.load_state_dict(torch.load(warm_start, map_location=device, weights_only=True))
        trainer.optimizer.state.clear()
        target_streak = 0
        print(f"WARM START {label} from {warm_start}: counters retained; optimizer momentum restarted", flush=True)
    start = last_report = time.monotonic()
    next_evaluation = ((trainer.global_step // options["eval_interval"]) + 1) * options["eval_interval"]
    status = "training"
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save_state():
        atomic_save(checkpoint, {
            "policy": trainer.policy.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "global_step": trainer.global_step, "epoch": trainer.epoch,
            "encoder": config["torch"]["encoder"],
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "mps_rng": torch.mps.get_rng_state() if device == "mps" else None,
            "python_rng": random.getstate(), "best_score": best_score,
            "target_streak": target_streak, "history": history,
            "seconds": prior_seconds + time.monotonic() - start,
        })

    def report_state():
        report = {
            "lag": lag, "seed": seed, "status": status,
            "worker_pid": os.getpid(),
            "source_steps": source_steps, "additional_steps": trainer.global_step,
            "total_steps": source_steps + trainer.global_step,
            "best_validation_return": best_score if math.isfinite(best_score) else None,
            "target_return": options["target_return"], "target_streak": target_streak,
            "seconds": prior_seconds + time.monotonic() - start,
            "history": history,
        }
        atomic_json(destination / "status.json", report)
        return report

    previous_eval_episodes = prior_continuation.get("options", {}).get("eval_episodes")
    if (saved is not None and previous_eval_episodes is not None
            and previous_eval_episodes != options["eval_episodes"]
            and (destination / "policy.bin").exists()):
        # Scores from differently sized selection cohorts are not interchangeable.
        # Re-rank the incumbent before comparing it with new candidate policies.
        validation = evaluation(config, destination / "policy.bin", options["eval_episodes"],
                                200000, lag, config["env"]["volatility"])
        best_score = validation["mean_return_including_capped"]
        target_streak = 0
        history.append({"additional_steps": trainer.global_step,
                        "validation_return": best_score, "win_rate": validation["win_rate"],
                        "capped_episodes": validation["capped_episodes"],
                        "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                        "evaluation_episodes": options["eval_episodes"],
                        "revalidated_selected_checkpoint": True})
        atomic_json(validation_dir / f"validation_recalibrated_{trainer.global_step:012d}.json", validation)
        save_state()
        print(f"REVALIDATED BEST {label} episodes={options['eval_episodes']} return={best_score:.2f}", flush=True)

    print(f"START {label} source_steps={source_steps:,} resumed_steps={trainer.global_step:,}", flush=True)
    report_state()
    try:
        with (destination / "training.jsonl").open("a") as stream:
            while trainer.global_step < options["additional_steps"]:
                trainer.rollouts()
                if options.get("anneal_steps") is not None:
                    fraction = min(1.0, trainer.global_step / options["anneal_steps"])
                    rate = options["learning_rate"] * (0.1 + 0.45 * (1 + math.cos(math.pi * fraction)))
                    for group in trainer.optimizer.param_groups:
                        group["lr"] = rate
                trainer.train()
                if stop_requested:
                    status = "paused"
                    save_state()
                    return report_state()
                if not all(torch.isfinite(parameter).all().item() for parameter in trainer.policy.parameters()):
                    raise FloatingPointError(f"Nonfinite parameters: {label}")
                if time.monotonic() - last_report >= 20:
                    metrics = trainer.log()
                    stream.write(json.dumps(metrics) + "\n")
                    stream.flush()
                    print(f"TRAIN {label} steps={trainer.global_step:,} "
                          f"score={metrics['env'].get('score', float('nan')):.2f} "
                          f"sps={metrics['SPS']:.0f}", flush=True)
                    report_state()
                    last_report = time.monotonic()
                if trainer.global_step >= next_evaluation or trainer.global_step >= options["additional_steps"]:
                    candidate = destination / "latest.bin"
                    atomic_save(candidate, trainer.policy.state_dict())
                    validation = evaluation(config, candidate, options["eval_episodes"],
                                            200000, lag, config["env"]["volatility"])
                    score = validation["mean_return_including_capped"]
                    if score >= best_score:
                        best_score = score
                        atomic_save(destination / "policy.bin", trainer.policy.state_dict())
                    target_streak = target_streak + 1 if (
                        score >= options["target_return"] and validation["capped_episodes"] == 0
                    ) else 0
                    record = {
                        "additional_steps": trainer.global_step,
                        "validation_return": score, "win_rate": validation["win_rate"],
                        "capped_episodes": validation["capped_episodes"],
                        "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                        "evaluation_episodes": options["eval_episodes"],
                    }
                    history.append(record)
                    atomic_json(validation_dir / f"validation_{trainer.global_step:012d}.json", validation)
                    print(f"EVAL {label} steps={trainer.global_step:,} return={score:.2f} "
                          f"win_rate={validation['win_rate']:.1%} best={best_score:.2f}", flush=True)
                    next_evaluation += options["eval_interval"]
                    save_state()
                    report_state()
                    if target_streak >= options["target_checks"] and not options.get("continue_after_target"):
                        status = "target_reached_on_validation"
                        break
            else:
                status = ("target_reached_on_validation" if target_streak >= options["target_checks"]
                          else "budget_reached_below_target")
            # Fixed holdout seeds differ from checkpoint-selection seeds.
            own_holdout = evaluation(config, destination / "policy.bin", options["holdout_episodes"],
                                     300000, lag, config["env"]["volatility"])
            original = evaluation(config, destination / "policy.bin", options["holdout_episodes"],
                                  100000, 0, 0.0)
            atomic_json(destination / "holdout_training_environment.json", own_holdout)
            atomic_json(destination / "holdout_original_environment.json", original)
            save_state()
            report = report_state()
            report["holdout_training_return"] = own_holdout["mean_return_including_capped"]
            report["holdout_original_return"] = original["mean_return_including_capped"]
            atomic_json(destination / "status.json", report)
            print(f"DONE {label} status={status} own={report['holdout_training_return']:.2f} "
                  f"original={report['holdout_original_return']:.2f}", flush=True)
            return report
    except BaseException:
        status = "interrupted_or_failed"
        save_state()
        report_state()
        raise
    finally:
        trainer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lags", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--manifest-name", default="experiment",
                        help="Name this worker group's manifest when separate seed groups share an output")
    parser.add_argument("--additional-steps", type=int, default=100_000_000)
    parser.add_argument("--anneal-steps", type=int,
                        help="Decay over this many additional steps, then keep the learning-rate floor")
    parser.add_argument("--eval-interval", type=int, default=5_000_000)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--holdout-episodes", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--gamma", type=float,
                        help="Override the reward discount; resuming otherwise preserves the saved value")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--agents", type=int)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--normalize-velocity", action="store_true")
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--restart-from-best", action="store_true",
                        help="Resume counters/history but restore the selected best policy with fresh momentum")
    restart.add_argument("--warm-start", type=Path,
                        help="Initialize from a compatible trained policy; record the source and retain counters")
    parser.add_argument("--target-return", type=float, default=850.0)
    parser.add_argument("--target-checks", type=int, default=3)
    parser.add_argument("--continue-after-target", action="store_true",
                        help="Keep training to the step budget even if the empirical target was already reached")
    args = parser.parse_args()
    if any(value <= 0 for value in (args.workers, args.additional_steps, args.eval_interval,
                                    args.eval_episodes, args.holdout_episodes, args.target_checks)):
        parser.error("workers, step counts, episode counts, and checks must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if args.gamma is not None and (not math.isfinite(args.gamma) or not 0 < args.gamma <= 1):
        parser.error("gamma must be finite and in (0, 1]")
    if not math.isfinite(args.target_return) or not 0 <= args.target_return <= 864:
        parser.error("target-return must be in [0, 864]")
    if args.device == "mps" and args.workers != 1:
        parser.error("Use --workers 1 when training on the Apple GPU")
    if any(value is not None and value <= 0 for value in (args.agents, args.minibatch_size)):
        parser.error("agents and minibatch-size must be positive")
    if args.anneal_steps is not None and args.anneal_steps <= 0:
        parser.error("anneal-steps must be positive")
    if not args.manifest_name or any(not (char.isalnum() or char == "_") for char in args.manifest_name):
        parser.error("manifest-name must contain only letters, digits, and underscores")
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents:
        parser.error("output must be a separate directory from the source experiment")
    jobs = [(source / f"lag_{lag}" / f"seed_{seed}", output / f"lag_{lag}" / f"seed_{seed}")
            for seed in args.seeds for lag in args.lags]
    if len(set(jobs)) != len(jobs):
        parser.error("lags and seeds must not contain duplicates")
    if args.warm_start is not None:
        if len(jobs) != 1:
            parser.error("warm-start requires exactly one lag and one seed")
        if not args.warm_start.is_file() or not (args.warm_start.parent / "config.json").is_file():
            parser.error("warm-start requires a checkpoint and its adjacent config.json")
    for folder, _ in jobs:
        for filename in ("config.json", "training_result.json", "policy.bin"):
            if not (folder / filename).is_file():
                parser.error(f"Missing source file: {folder / filename}")
    output.mkdir(parents=True, exist_ok=True)
    phase_dir = output / "history" / "phases"
    phase_dir.mkdir(parents=True, exist_ok=True)
    phase_name = f"{args.manifest_name}_{time.time_ns()}"
    options = {key: getattr(args, key) for key in (
        "additional_steps", "eval_interval", "eval_episodes", "holdout_episodes",
        "learning_rate", "gamma", "target_return", "target_checks", "continue_after_target", "device", "agents", "minibatch_size", "anneal_steps", "normalize_velocity", "restart_from_best")}
    options["warm_start"] = str(args.warm_start.resolve()) if args.warm_start else None
    atomic_json(phase_dir / f"{phase_name}.json", {"source": str(source), "options": options, "parent_pid": os.getpid(),
                                             "lags": args.lags, "seeds": args.seeds})
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    results = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(train_run, first, second, options) for first, second in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            atomic_json(phase_dir / f"results_{phase_name}.json", results)


if __name__ == "__main__":
    main()
