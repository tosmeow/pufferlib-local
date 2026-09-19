"""Run the existing lagged-Breakout comparison on a Modal T4 GPU.

Setup and examples: docs/modal_lagged_breakout.md.
Only Modal and the standard library are imported locally; training imports and
native compilation happen in the remote Linux environment.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import uuid

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/opt/pufferlib"
RESULTS_ROOT = "/results"
VOLUME_NAME = "breakout-results"
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"
TORCH_VERSION = "2.9.1"

# Explicit inclusion keeps local binaries, credentials, benchmarks and virtual
# environments out of the upload. Uncommitted source edits are included.
SOURCE_PATTERNS = (
    "build.sh", "pyproject.toml", "README.md", "LICENSE",
    "pufferlib/**/*.py", "src/**/*.h", "src/**/*.cu",
    "vendor/**/*.h", "vendor/**/*.c",
    "ocean/breakout/*.h", "ocean/lagged_breakout/*.h",
    "ocean/lagged_breakout/binding.c",
    "config/default.ini", "config/breakout.ini", "config/lagged_breakout.ini",
    "scripts/build_modal_lagged_breakout.sh",
    "scripts/compare_lagged_breakout.py", "scripts/modal_lagged_breakout.py",
    "scripts/train_lagged_breakout_timed.py",
)

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .entrypoint([])
    .apt_install(
        "build-essential", "clang", "libomp-dev", "ccache", "curl",
        "ca-certificates", "libgl1-mesa-dev", "libx11-6", "libxrandr2",
        "libxi6", "libxcursor1", "libxinerama1",
    )
    .pip_install(f"torch=={TORCH_VERSION}", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install_from_pyproject(str(ROOT / "pyproject.toml"))
    .add_local_dir(
        ROOT, REMOTE_ROOT, copy=True,
        ignore=~modal.FilePatternMatcher(*SOURCE_PATTERNS),
    )
    .workdir(REMOTE_ROOT)
    .env({
        # `modal run scripts/file.py` imports the function by its bare module
        # name. Include the scripts directory because include_source=False uses
        # our uploaded copy instead of Modal's automatic source mount.
        "PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT}/scripts",
        "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    })
    .run_commands(
        "python -m pip install --no-deps -e .",
        "bash scripts/build_modal_lagged_breakout.sh",
    )
)
app = modal.App("pufferlib-lagged-breakout")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def integer_list(value, name, maximum):
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers") from exc
    if not values or any(item < 0 or item > maximum for item in values):
        raise ValueError(f"{name} must be between 0 and {maximum}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def experiment_command(run_name, lags, seeds, timesteps, agents, horizon,
                       minibatch_size, volatility, eval_episodes, max_eval_steps):
    """Validate before invoking a remote Function; pass arguments without a shell."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_name):
        raise ValueError("run-name must be 1–96 letters, digits, underscores or hyphens")
    lags = integer_list(lags, "lags", 64)
    seeds = integer_list(seeds, "seeds", 2**31 - 1)
    if any(value <= 0 for value in (
        timesteps, agents, horizon, minibatch_size, eval_episodes, max_eval_steps,
    )):
        raise ValueError("Training sizes and evaluation limits must be positive")
    if horizon < 2:
        raise ValueError("horizon must be at least 2 to compute advantages")
    if minibatch_size % horizon or minibatch_size > agents * horizon:
        raise ValueError("minibatch-size must divide into whole horizons and be <= agents * horizon")
    if not math.isfinite(volatility) or volatility < 0:
        raise ValueError("volatility must be finite and nonnegative")
    return [
        "scripts/compare_lagged_breakout.py",
        "--output", f"{RESULTS_ROOT}/{run_name}",
        "--lags", *map(str, lags), "--seeds", *map(str, seeds),
        "--timesteps", str(timesteps), "--agents", str(agents),
        "--horizon", str(horizon), "--minibatch-size", str(minibatch_size),
        "--volatility", str(volatility), "--device", "cuda",
        "--eval-episodes", str(eval_episodes), "--max-eval-steps", str(max_eval_steps),
    ]


def execute(run_name, command, smoke=False, relative_output=None):
    import subprocess
    import sys
    import torch
    from pufferlib import _C

    if not torch.cuda.is_available():
        raise RuntimeError("The remote PyTorch installation cannot use CUDA")
    if _C.env_name != "lagged_breakout" or _C.precision_bytes != 4 or not _C.gpu:
        raise RuntimeError("Expected the CUDA float32 lagged_breakout extension")

    # Refuse to overwrite a previous launch, including a failed partial run.
    volume.reload()
    destination = Path(RESULTS_ROOT) / (relative_output or run_name)
    destination.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for pattern in SOURCE_PATTERNS:
        for path in Path(REMOTE_ROOT).glob(pattern):
            if path.is_file():
                source_hashes[str(path.relative_to(REMOTE_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "run_name": run_name, "mode": "timed" if relative_output else "smoke" if smoke else "comparison",
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command, "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "cuda_image": CUDA_IMAGE, "source_sha256": source_hashes,
        "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
    }
    manifest_path = destination / "modal_run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    volume.commit()
    print(f"GPU: {manifest['gpu']}; PyTorch: {manifest['torch']}; CUDA: {manifest['cuda']}", flush=True)
    print(shlex.join(["python", *command]), flush=True)
    try:
        # Stream to both Modal's logs and the persistent run directory. Keep the
        # experiment in its own process so its imports/signals do not alter Modal.
        with (destination / "console.log").open("w", buffering=1) as log:
            with subprocess.Popen(
                [sys.executable, *command], cwd=REMOTE_ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            ) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    if relative_output and line.startswith("CHECKPOINT "):
                        volume.commit()
                if process.wait():
                    raise subprocess.CalledProcessError(process.returncode, command)
        if smoke:
            checkpoint = destination / "lag_4/seed_42/policy.bin"
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not weights or not all(torch.isfinite(value).all().item() for value in weights.values()):
                raise RuntimeError("Smoke checkpoint contains missing or nonfinite weights")
            print("Smoke check passed: CUDA training, evaluation and CPU checkpoint loading.", flush=True)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        volume.commit()
    return str(destination)


@app.function(
    image=image, gpu="T4", cpu=4, memory=16384, timeout=6 * 60 * 60,
    max_containers=1, retries=0, volumes={RESULTS_ROOT: volume}, include_source=False,
)
def train(run_name: str, command: list[str], smoke: bool):
    return execute(run_name, command, smoke)


@app.function(
    image=image, gpu="T4", cpu=4, memory=16384, timeout=900,
    max_containers=4, retries=0, scaledown_window=2,
    volumes={RESULTS_ROOT: volume}, include_source=False,
)
def train_timed(run_name: str, command: list[str], relative_output: str):
    # One input per container, one T4 per container. Each lag writes its own
    # directory so concurrent Volume commits never contend for the same files.
    return execute(run_name, command, relative_output=relative_output)


def timed_commands(run_name, source_run, lags, seeds, seconds, agents, horizon,
                   minibatch_size, volatility, learning_rate, threads, autotune):
    # Reuse the original validation before scheduling any billable work.
    experiment_command(run_name, lags, seeds, 1, agents, horizon,
                       minibatch_size, volatility, 1, 1)
    if source_run and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", source_run):
        raise ValueError("source-run must be a run name, not a path")
    if source_run == run_name:
        raise ValueError("Use a new run-name when continuing a source-run")
    if not math.isfinite(seconds) or not 0 < seconds <= 3600:
        raise ValueError("seconds must be in (0, 3600]")
    if not math.isfinite(learning_rate) or learning_rate <= 0 or not 1 <= threads <= 4:
        raise ValueError("learning-rate must be positive and threads must be between 1 and 4")
    jobs = []
    for lag in integer_list(lags, "lags", 64):
        for seed in integer_list(seeds, "seeds", 2**31 - 1):
            relative = f"{run_name}/lag_{lag}/seed_{seed}"
            command = [
                "scripts/train_lagged_breakout_timed.py", "--output", f"{RESULTS_ROOT}/{relative}",
                "--lag", str(lag), "--seed", str(seed), "--seconds", str(seconds),
                "--agents", str(agents), "--horizon", str(horizon),
                "--minibatch-size", str(minibatch_size), "--threads", str(threads),
                "--learning-rate", str(learning_rate), "--volatility", str(volatility),
                "--device", "cuda",
            ]
            if source_run:
                command.extend(["--source", f"{RESULTS_ROOT}/{source_run}/lag_{lag}/seed_{seed}"])
            if autotune:
                command.append("--autotune")
            jobs.append((run_name, command, relative))
    return jobs


@app.local_entrypoint()
def main(run_name: str = "", lags: str = "0,2,4,8", seeds: str = "42",
         timesteps: int = 1_000_000, agents: int = 4096, horizon: int = 64,
         minibatch_size: int = 16384, volatility: float = 6.0,
         eval_episodes: int = 32, max_eval_steps: int = 120000, smoke: bool = False,
         seconds: float = 0, source_run: str = "", autotune: bool = False,
         learning_rate: float = 0.005, threads: int = 4):
    """--seconds enables parallel timed training, one T4 per lag/seed (up to four)."""
    if smoke and seconds:
        raise ValueError("Use --smoke or --seconds, not both")
    if not seconds and (source_run or autotune):
        raise ValueError("source-run and autotune require --seconds")
    if not run_name:
        prefix = "smoke" if smoke else "timed" if seconds else "modal"
        run_name = f"{prefix}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    if smoke:
        lags, seeds = "4", "42"
        timesteps, agents, horizon, minibatch_size = 1024, 64, 16, 1024
        eval_episodes, max_eval_steps = 1, 64
    command = experiment_command(
        run_name, lags, seeds, timesteps, agents, horizon, minibatch_size,
        volatility, eval_episodes, max_eval_steps,
    )
    jobs = timed_commands(run_name, source_run, lags, seeds, seconds, agents, horizon,
                          minibatch_size, volatility, learning_rate, threads, autotune) if seconds else None
    print(f"Results: {VOLUME_NAME}/{run_name}", flush=True)
    # Modal 1.5's recursive download preserves the remote directory name and
    # requires an existing local parent directory.
    local_results = ROOT / "benchmarks" / "lagged_breakout"
    local_results.mkdir(parents=True, exist_ok=True)
    print("Download with: " + shlex.join([
        str(ROOT / ".venv-modal/bin/modal"), "volume", "get", VOLUME_NAME, f"/{run_name}",
        str(local_results),
    ]), flush=True)
    if jobs:
        print(f"Launching {len(jobs)} policies on up to 4 separate T4s, {seconds:g}s training each. "
              "Setup/calibration and final saving are additional time.", flush=True)
        function = train_timed.with_options(timeout=math.ceil(seconds) + 600)
        results = list(function.starmap(jobs, return_exceptions=True))
        errors = [result for result in results if isinstance(result, BaseException)]
        for result in results:
            print(result, flush=True)
        if errors:
            raise RuntimeError(f"{len(errors)} timed job(s) failed; inspect their Modal logs") from errors[0]
        print("All timed policies saved. Download, then evaluate locally with "
              "scripts/compare_lagged_breakout.py --evaluate-only.", flush=True)
        return
    # One tiny policy needs much less host compute than a full comparison.
    # Request T4 explicitly; do not fall back to a more expensive GPU.
    function = train.with_options(gpu="T4", cpu=1, memory=4096, timeout=600) if smoke else train
    function.remote(run_name, command, smoke)
