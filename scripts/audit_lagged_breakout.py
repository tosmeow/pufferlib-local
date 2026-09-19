"""Evaluate qualified policies at their training lag on a larger unseen seed set."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.continue_lagged_breakout import atomic_json, evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=400000)
    args = parser.parse_args()
    if args.episodes <= 0 or not 0 <= args.seed <= 2**31 - args.episodes:
        parser.error("episodes must be positive and the seed range must fit in signed 32 bits")
    from pufferlib import _C
    if _C.env_name != "lagged_breakout" or _C.precision_bytes != 4:
        raise RuntimeError("Build lagged_breakout with --cpu or --float first")
    torch.set_num_threads(1)
    for path in sorted(args.output.resolve().glob("lag_*/seed_*/status.json")):
        status = json.loads(path.read_text())
        if (status["status"] != "target_reached_on_validation"
                or status.get("holdout_training_return", 0) < status["target_return"]):
            continue
        checkpoint = path.parent / "policy.bin"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        report = path.parent / "audit_training_environment.json"
        if report.exists():
            previous = json.loads(report.read_text())
            if (previous.get("checkpoint_sha256") == digest
                    and previous["seed"] == args.seed
                    and len(previous["episodes"]) == args.episodes):
                continue
        config = json.loads((path.parent / "config.json").read_text())
        result = evaluation(config, checkpoint, args.episodes, args.seed,
                            status["lag"], config["env"]["volatility"])
        result["checkpoint_sha256"] = digest
        atomic_json(report, result)
        print(f"AUDIT lag={status['lag']} seed={status['seed']} "
              f"mean={result['mean_return_including_capped']:.2f} "
              f"wins={result['win_rate']:.1%} capped={result['capped_episodes']}", flush=True)


if __name__ == "__main__":
    main()
