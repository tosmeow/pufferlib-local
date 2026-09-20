"""Run the native PufferLib 5.0 trainer on a remote Modal T4.

The local checkout is uploaded for each invocation, so uncommitted source
edits are included while local binaries, checkpoints, virtual environments,
and benchmark artifacts are excluded. Results are written to a persistent
Modal Volume.
"""

from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/opt/pufferlib"
RESULTS_ROOT = "/results"
VOLUME_NAME = "pufferlib-5-results"
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"

# Keep the upload source-only. In particular, do not send local benchmark
# artifacts or the Modal/Python virtual environments to the image builder.
SOURCE_PATTERNS = (
    "build.sh",
    "README.md",
    "LICENSE",
    "config/**/*.ini",
    "ocean/**/*.c",
    "ocean/**/*.cc",
    "ocean/**/*.cpp",
    "ocean/**/*.cu",
    "ocean/**/*.h",
    "src/**/*.c",
    "src/**/*.cc",
    "src/**/*.cpp",
    "src/**/*.cu",
    "src/**/*.h",
    "tests/**/*.c",
    "tests/**/*.cc",
    "tests/**/*.cpp",
    "tests/**/*.cu",
    "tests/**/*.h",
    "tests/**/*.py",
    "vendor/**/*.c",
    "vendor/**/*.cc",
    "vendor/**/*.cpp",
    "vendor/**/*.h",
    "scripts/build_modal_native.sh",
    "scripts/modal_native.py",
)


image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .entrypoint([])
    .apt_install(
        "build-essential",
        "clang",
        "cmake",
        "ccache",
        "ca-certificates",
        "curl",
        "git",
        "libgl1-mesa-dev",
        "libomp-dev",
        "libx11-6",
        "libxrandr2",
        "libxi6",
        "libxcursor1",
        "libxinerama1",
        "unzip",
    )
    .pip_install("nvidia-nccl-cu12")
    .add_local_dir(
        ROOT,
        REMOTE_ROOT,
        copy=True,
        ignore=~modal.FilePatternMatcher(*SOURCE_PATTERNS),
    )
    .workdir(REMOTE_ROOT)
    .env({
        # With include_source=False, Modal imports the function module from
        # the copy uploaded into the image rather than mounting local sources.
        "PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT}/scripts",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    })
)

app = modal.App("pufferlib-5-native")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _valid_name(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise ValueError(
            f"{label} must be 1-96 letters, digits, underscores or hyphens"
        )
    return value


def _source_hashes() -> dict[str, str]:
    hashes = {}
    for pattern in SOURCE_PATTERNS:
        for path in Path(REMOTE_ROOT).glob(pattern):
            if path.is_file():
                relative = str(path.relative_to(REMOTE_ROOT))
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _run_stream(command: list[str], log, env: dict[str, str] | None = None) -> None:
    print(shlex.join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=REMOTE_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _puffer_command(
    env_name: str,
    mode: str,
    run_name: str,
    model: str,
    overrides: str,
) -> list[str]:
    if mode not in {"train", "eval", "match", "sweep"}:
        raise ValueError("mode must be train, eval, match or sweep")
    command = ["./puffer", mode]
    if mode in {"eval", "match"}:
        command.append(model or "latest")
    command.extend([
        f"--base.env_name={env_name}",
        f"--base.run_id={run_name}",
        f"--base.checkpoint_dir={RESULTS_ROOT}/{run_name}/checkpoints",
        f"--base.log_dir={RESULTS_ROOT}/{run_name}/logs",
        "--headless",
    ])
    for item in shlex.split(overrides):
        if not item.startswith("--") or "=" not in item:
            raise ValueError(f"invalid override {item!r}; use --section.key=value")
        command.append(item)
    return command


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=6 * 60 * 60,
    retries=0,
    volumes={RESULTS_ROOT: volume},
    include_source=False,
)
def run_remote(
    env_name: str,
    mode: str,
    run_name: str,
    model: str,
    overrides: str,
) -> str:
    env_name = _valid_name(env_name, "env")
    run_name = _valid_name(run_name, "run-name")
    volume.reload()

    destination = Path(RESULTS_ROOT) / run_name
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_name": run_name,
        "env": env_name,
        "mode": mode,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cuda_image": CUDA_IMAGE,
        "source_sha256": _source_hashes(),
    }
    manifest_path = destination / "modal_run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    volume.commit()

    build_command = ["bash", "scripts/build_modal_native.sh", env_name]
    command = _puffer_command(env_name, mode, run_name, model, overrides)
    log_path = destination / "console.log"
    try:
        with log_path.open("w", buffering=1) as log:
            _run_stream(build_command, log)
            _run_stream(command, log)
        manifest["status"] = "completed"
        return str(destination)
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        volume.commit()


@app.local_entrypoint()
def main(
    env: str = "breakout",
    mode: str = "train",
    run_name: str = "",
    model: str = "",
    overrides: str = "",
    smoke: bool = False,
):
    if not run_name:
        run_name = f"{env}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    if smoke:
        smoke_overrides = (
            "--train.total_timesteps=1024 "
            "--vec.total_agents=64 "
            "--vec.num_buffers=1 "
            "--vec.num_threads=1 "
            "--train.horizon=16 "
            "--train.minibatch_size=1024 "
            "--base.checkpoint_interval=1 "
            "--base.cudagraphs=0"
        )
        overrides = f"{smoke_overrides} {overrides}".strip()
    if mode != "train" and smoke:
        raise ValueError("--smoke is only supported with --mode train")
    print(f"Results: {VOLUME_NAME}/{run_name}", flush=True)
    print("Download with:", flush=True)
    print(
        shlex.join([
            str(ROOT / ".venv-modal/bin/modal"),
            "volume",
            "get",
            VOLUME_NAME,
            f"/{run_name}",
            str(ROOT / "benchmarks" / "modal_native"),
        ]),
        flush=True,
    )
    result = run_remote.remote(env, mode, run_name, model, overrides)
    print(f"Remote result directory: {result}", flush=True)
