"""Slurm-only long RPS run with live checkpoint evaluation and final GPU parity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def run(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("share", type=float)
    parser.add_argument("--steps", type=int, default=2_000_000_000)
    parser.add_argument("--interval", type=int, default=3052)
    args = parser.parse_args()
    assert 0 <= args.share <= 1
    root = Path(os.environ["CLUSTER_RESULTS_DIR"])
    assert os.environ.get("SLURM_JOB_ID"), "Submit through cluster run"
    root.mkdir(parents=True, exist_ok=True)
    Path("build").mkdir(exist_ok=True)
    Path("build/rps_test_headers").mkdir(exist_ok=True)
    Path("build/rps_test_headers/raylib.h").write_text("// RPS has no graphics calls.\n")
    # Compilation and tests run within this managed allocation.
    run("g++", "-std=c++20", "-O1", "-g", "-fsanitize=address,undefined",
        "-Isrc", "-Ibuild/rps_test_headers",
        "ocean/rock_paper_scissors/test_opponents.cpp", "-o", "build/rps_opponent_tests")
    run("build/rps_opponent_tests")
    run("gcc", "-O2", "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-DRPS_BOT_EVAL",
        "-DRPS_HIDDEN_SIZE=64", "-DRPS_NUM_LAYERS=4", "-Isrc",
        "-Ibuild/rps_test_headers", "scripts/rps_policy_probe.c", "-lm",
        "-o", "build/rps_monitor_probe")
    run_id = "long_mix_seed_73"
    command = ["bash", "scripts/rps.sh", "train", "--env.bot_policy=5",
               f"--env.selfplay_share={args.share}", "--policy.hidden_size=64",
               "--policy.num_layers=4", "--base.seed=73", f"--base.run_id={run_id}",
               "--base.load_model_path=None", "--train.learning_rate=0.0015",
               f"--train.total_timesteps={args.steps}",
               f"--base.checkpoint_interval={args.interval}"]
    (root / "protocol.json").write_text(json.dumps(dict(command=command,
        share=args.share, seed=73, requested_steps=args.steps,
        warning_thresholds=dict(low_selfplay_entropy=0.5, entropy_drop=0.2,
                                scripted_reward_below=0.9)), indent=2))
    checkpoints = root / "checkpoints/rock_paper_scissors" / run_id
    seen = set()
    previous_entropy = None
    history = []
    with (root / "training.log").open("w") as log:
        training = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                finished = training.poll() is not None
                # Trainer publishes checkpoints by atomic rename.
                for checkpoint in sorted(checkpoints.glob("*.bin")):
                    if checkpoint.name in seen:
                        continue
                    output = root / "monitor" / checkpoint.stem
                    output.mkdir(parents=True, exist_ok=True)
                    with (output / "evaluation.log").open("w") as evaluation_log:
                        run("build/rps_monitor_probe", str(checkpoint),
                            str(output / "trajectories.csv"), str(output / "probes.csv"),
                            stdout=evaluation_log, stderr=subprocess.STDOUT)
                        run("python3", "scripts/rps_analysis.py", "summary", str(output),
                            stdout=evaluation_log, stderr=subprocess.STDOUT)
                    groups = json.loads((output / "summary.json").read_text())["groups"]
                    entropy = groups["selfplay:warm"]["mean_entropy_nats"]
                    row = dict(steps=int(checkpoint.stem), selfplay_entropy=entropy,
                        selfplay_tv=groups["selfplay:warm"]["mean_tv_from_uniform"],
                        rock_reward=groups["bot_rock"]["mean_reward"],
                        counter_reward=groups["bot_counter"]["mean_reward"],
                        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                    row["warnings"] = []
                    if entropy < 0.5:
                        row["warnings"].append("low self-play entropy")
                    if previous_entropy is not None and entropy < previous_entropy - 0.2:
                        row["warnings"].append("self-play entropy dropped >0.2")
                    if min(row["rock_reward"], row["counter_reward"]) < 0.9:
                        row["warnings"].append("scripted reward below 0.9")
                    history.append(row)
                    temporary = root / "monitor_history.tmp"
                    temporary.write_text(json.dumps(history, indent=2))
                    temporary.replace(root / "monitor_history.json")
                    live = Path(os.environ["CLUSTER_JOB_DIR"]) / "rps_monitor_history.tmp"
                    live.write_text(json.dumps(history, indent=2))
                    live.replace(live.with_name("rps_monitor_history.json"))
                    print(json.dumps(row), flush=True)
                    previous_entropy = entropy
                    seen.add(checkpoint.name)
                if finished:
                    break
                time.sleep(2)
            if training.returncode:
                raise RuntimeError(f"Training failed: {training.returncode}; see training.log")
        finally:
            if training.poll() is None:
                training.terminate()
                training.wait()
    assert seen, "No checkpoints produced"
    final = max(checkpoints.glob("*.bin"), key=lambda p: int(p.stem))
    run("bash", "scripts/rps.sh", "eval", str(final), "64", "4",
        hashlib.sha256(final.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
