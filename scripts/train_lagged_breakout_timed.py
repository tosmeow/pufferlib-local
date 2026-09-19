"""Train one lag/seed for a wall-clock budget, optionally continuing weights.

Used by modal_lagged_breakout.py. Calibration uses disposable trainers; the
training clock starts after calibration and construction of the final trainer.
"""

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import random
import signal
import sys
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_for_seconds(trainer, seconds, learning_rate, report, stopped=lambda: False,
                    clock=time.monotonic, synchronize=lambda: None):
    """Stop between complete updates, keeping schedules tied to elapsed time."""
    synchronize()
    started = last_report = clock()
    iterations = 0
    while not stopped():
        elapsed = clock() - started
        if elapsed >= seconds:
            break
        progress = elapsed / seconds
        lr = learning_rate * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
        for group in trainer.optimizer.param_groups:
            group["lr"] = lr
        # train() also anneals replay prioritization using epoch/total_epochs.
        # Replace the parent's small timestep budget with wall-time progress.
        trainer.total_epochs = max(1, trainer.epoch / max(progress, 1e-12))
        trainer.rollouts()
        trainer.train()
        synchronize()
        iterations += 1
        now = clock()
        if now - last_report >= 10 or now - started >= seconds or stopped():
            report(now - started, lr)
            last_report = clock()
    return {"seconds": clock() - started, "iterations": iterations,
            "status": "interrupted" if stopped() else "completed"}


def create_trainer(config, threads):
    import numpy as np
    import torch
    from pufferlib import _C
    from pufferlib.torch_pufferl import PuffeRL, load_policy

    torch.set_num_threads(threads)
    seed = config["train"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(config["torch"]["device"])
    vec = _C.create_vec(config, int(device.type == "cuda"))
    try:
        policy = load_policy(config, vec, device=device)
        return PuffeRL(config, vec, policy, verbose=False, device=device)
    except BaseException:
        vec.close()
        raise


def calibrate(config, destination, threads):
    """Compare end-to-end steps/s, keeping horizon and replay ratio fixed."""
    import torch

    measurements = []
    synchronize = torch.cuda.synchronize if config["torch"]["device"] == "cuda" else lambda: None
    # Include the requested configuration, then test larger rollout/minibatches.
    sizes = list(dict.fromkeys([
        (config["vec"]["total_agents"], config["train"]["minibatch_size"]),
        (4096, 16384), (8192, 32768), (16384, 65536), (32768, 131072),
    ]))
    for candidate_threads in sorted({1, threads}):
        for agents, minibatch in sizes:
            if minibatch % config["train"]["horizon"] or minibatch > agents * config["train"]["horizon"]:
                continue
            candidate = copy.deepcopy(config)
            candidate["vec"].update(total_agents=agents, num_threads=candidate_threads)
            candidate["train"]["minibatch_size"] = minibatch
            trainer = None
            result = {"agents": agents, "minibatch_size": minibatch, "threads": candidate_threads}
            try:
                trainer = create_trainer(candidate, candidate_threads)
                trainer.rollouts()
                trainer.train()  # Warm CUDA kernels/allocator before timing.
                synchronize()
                started = time.monotonic()
                for _ in range(3):
                    trainer.rollouts()
                    trainer.train()
                synchronize()
                seconds = time.monotonic() - started
                result.update(seconds=seconds, steps_per_second=3 * trainer.batch_size / seconds,
                              metrics=trainer.log())
            except torch.cuda.OutOfMemoryError:
                result["error"] = "CUDA out of memory"
            finally:
                if trainer is not None:
                    trainer.close()
                del trainer
                gc.collect()
                if config["torch"]["device"] == "cuda":
                    torch.cuda.empty_cache()
            measurements.append(result)
            write_json(destination / "throughput.json", {"measurements": measurements})
            print("CALIBRATION " + json.dumps(result), flush=True)
    valid = [item for item in measurements if "steps_per_second" in item]
    if not valid:
        raise RuntimeError("No calibration configuration fit in GPU memory")
    best = max(valid, key=lambda item: item["steps_per_second"])
    write_json(destination / "throughput.json", {"measurements": measurements, "selected": best})
    config["vec"].update(total_agents=best["agents"], num_threads=best["threads"])
    config["train"]["minibatch_size"] = best["minibatch_size"]
    print("SELECTED " + json.dumps(best), flush=True)
    return best["threads"]


def train_one(args):
    import numpy as np
    import torch
    from pufferlib import _C
    from pufferlib.pufferl import load_config

    if _C.env_name != "lagged_breakout" or _C.precision_bytes != 4:
        raise RuntimeError("Expected the float32 lagged_breakout native extension")
    if args.device == "cuda" and (not torch.cuda.is_available() or not _C.gpu):
        raise RuntimeError("This run requires the CUDA native extension and a GPU")
    source_steps = 0
    source_state = None
    if args.source:
        config = json.loads((args.source / "config.json").read_text())
        if config["env"]["action_lag"] != args.lag or config["train"]["seed"] != args.seed:
            raise ValueError("Source checkpoint does not match the requested lag/seed")
        parent_result = args.source / "training_result.json"
        if parent_result.exists():
            parent = json.loads(parent_result.read_text())
            source_steps = parent.get("total_steps", parent["steps"])
        config["load_model_path"] = str(args.source / "policy.bin")
        if (args.source / "training_state.pt").exists():
            source_state = torch.load(args.source / "training_state.pt", map_location="cpu", weights_only=False)
    else:
        with mock.patch.object(sys, "argv", [sys.argv[0]]):
            config = load_config("lagged_breakout")
        config["load_model_path"] = None
        config["env"]["volatility"] = args.volatility
    config.update(slowly=True, load_id=None, seed=args.seed)
    config["env"].update(action_lag=args.lag, seed=args.seed)
    config["torch"]["device"] = args.device
    config["vec"].update(total_agents=args.agents, num_buffers=1, num_threads=args.threads)
    config["train"].update(seed=args.seed, horizon=args.horizon, minibatch_size=args.minibatch_size,
                           learning_rate=args.learning_rate, anneal_lr=0)
    config["timed_training"] = {"seconds": args.seconds, "source": str(args.source) if args.source else None,
                                "schedule": "wall-time cosine decay to 10% of initial learning rate"}
    destination = args.output
    destination.mkdir(parents=True, exist_ok=True)  # Modal wrapper already owns this unique directory.
    if (destination / "config.json").exists():
        raise FileExistsError(f"Refusing to overwrite training in {destination}")
    setup_started = time.monotonic()
    threads = calibrate(config, destination, args.threads) if args.autotune else args.threads
    write_json(destination / "config.json", config)
    trainer = create_trainer(config, threads)
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    previous_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    last_checkpoint = 0.0

    def save_checkpoint():
        weights = trainer.policy.state_dict()
        if not all(torch.isfinite(value).all().item() for value in weights.values()):
            raise RuntimeError("Nonfinite policy; preserving the previous checkpoint")
        state = {"policy": weights, "optimizer": trainer.optimizer.state_dict(),
                 "global_step": trainer.global_step, "total_steps": source_steps + trainer.global_step,
                 "epoch": trainer.epoch, "encoder": config["torch"]["encoder"],
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else None}
        for name, payload in (("training_state.pt", state), ("policy.bin", weights)):
            temporary = destination / (name + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(destination / name)
        print(f"CHECKPOINT steps={trainer.global_step}", flush=True)

    try:
        if source_state is not None:
            trainer.policy.load_state_dict(source_state["policy"])
            trainer.optimizer.load_state_dict(source_state["optimizer"])
            source_steps = source_state.get("total_steps", source_steps)
            random.setstate(source_state["python_rng"])
            np.random.set_state(source_state["numpy_rng"])
            torch.set_rng_state(source_state["torch_rng"])
            if args.device == "cuda" and source_state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(source_state["cuda_rng"])
        # Persist the starting state as well as periodic and final updates.
        save_checkpoint()
        setup_seconds = time.monotonic() - setup_started
        with (destination / "training.jsonl").open("w", buffering=1) as logfile:
            def report(elapsed, lr):
                nonlocal last_checkpoint
                metrics = trainer.log()
                metrics.update(training_seconds=elapsed, learning_rate=lr)
                logfile.write(json.dumps(metrics) + "\n")
                print(f"lag={args.lag} seed={args.seed} time={elapsed:.1f}/{args.seconds:g}s "
                      f"steps={trainer.global_step:,} SPS={metrics['SPS']:,.0f}", flush=True)
                if elapsed - last_checkpoint >= 30:
                    save_checkpoint()
                    last_checkpoint = elapsed

            result = run_for_seconds(
                trainer, args.seconds, args.learning_rate, report, lambda: interrupted,
                synchronize=torch.cuda.synchronize if args.device == "cuda" else lambda: None,
            )
        save_checkpoint()
        result.update(steps=trainer.global_step, source_steps=source_steps,
                      total_steps=source_steps + trainer.global_step,
                      requested_seconds=args.seconds, setup_seconds=setup_seconds,
                      steps_per_second=trainer.global_step / max(result["seconds"], 1e-9),
                      initial_optimizer="restored" if source_state is not None else "fresh",
                      agents=config["vec"]["total_agents"], minibatch_size=config["train"]["minibatch_size"],
                      threads=threads)
        write_json(destination / "training_result.json", result)
        print("RESULT " + json.dumps(result), flush=True)
        return result
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        trainer.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lag", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--agents", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=16384)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--volatility", type=float, default=6.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--autotune", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("seconds and learning-rate must be finite and positive")
    if args.horizon < 2 or min(args.agents, args.minibatch_size, args.threads) <= 0:
        parser.error("horizon must be >= 2; agents, minibatch-size and threads must be positive")
    if args.minibatch_size % args.horizon or args.minibatch_size > args.agents * args.horizon:
        parser.error("minibatch-size must be a multiple of horizon and <= agents * horizon")
    if not 0 <= args.lag <= 64 or not 0 <= args.seed < 2**31:
        parser.error("lag must be in [0,64] and seed in [0,2**31)")
    if not math.isfinite(args.volatility) or args.volatility < 0:
        parser.error("volatility must be finite and nonnegative")
    return args


if __name__ == "__main__":
    train_one(parse_args())
