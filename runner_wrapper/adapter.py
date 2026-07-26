from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


def _job_log(log_path: Path, message: str) -> None:
    line = message.rstrip()
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _stream_command(command: list[str], log_path: Path) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        _job_log(log_path, line)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _job_timestamp(job_id: Any) -> str:
    match = re.search(r"(\d{8})T(\d{6})", str(job_id))
    if match:
        return f"{match.group(1)[2:]}-{match.group(2)}"
    return time.strftime("%y%m%d-%H%M%S", time.gmtime())


def _environment_group(value: Any) -> str:
    if isinstance(value, list):
        environments = [str(item).strip() for item in value if str(item).strip()]
    else:
        environments = [
            item.strip()
            for item in str(value or "").split(",")
            if item.strip()
        ]
    environments = sorted(dict.fromkeys(environments))
    if not environments:
        raise ValueError("job.parameters.env is required")

    safe_environments = [
        re.sub(r"[^A-Za-z0-9._-]+", "-", environment).strip(".-_")
        or "environment"
        for environment in environments
    ]
    combined = "+".join(safe_environments)
    if len(safe_environments) <= 3 and len(combined) <= 80:
        return combined

    digest = hashlib.sha256(
        json.dumps(environments, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:8]
    return f"multi-{digest}"


def _reserve_artifact_paths(output_dir: Path, job_id: Any) -> tuple[Path, Path]:
    timestamp = _job_timestamp(job_id)
    counter = 1
    while True:
        suffix = timestamp if counter == 1 else f"{timestamp}_{counter}"
        log_path = output_dir / f"runner_{suffix}.log"
        metrics_path = output_dir / f"metrics_{suffix}.json"
        if metrics_path.exists():
            counter += 1
            continue
        try:
            log_path.touch(exist_ok=False)
        except FileExistsError:
            counter += 1
            continue
        return log_path, metrics_path


def _build_command(
    *,
    params: dict[str, Any],
    dataset_name: str,
    dataset_dir: Path,
    temp_dir: Path,
    summary_path: Path,
) -> list[str]:
    script_path = Path("scripts/download_dataset.py")
    return [
        "python",
        str(script_path),
        "--dataset-name",
        dataset_name,
        "--dataset-dir",
        str(dataset_dir),
        "--temp-dir",
        str(temp_dir),
        "--params-json",
        json.dumps(params, sort_keys=True),
        "--summary-json",
        str(summary_path),
    ]


def run_job(job_request: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    job = job_request["job"]
    runtime = job_request["runtime"]
    params = dict(job.get("parameters") or {})

    output_dir = Path(runtime["output_dir"])
    dataset_name = str(params.get("dataset_name") or "").strip()
    if not dataset_name:
        raise ValueError("job.parameters.dataset_name is required")
    artifact_dir = output_dir / _environment_group(params.get("env"))

    datasets_root = Path(os.environ["PATH_DATASETS"])
    dataset_dir = datasets_root / dataset_name
    temp_dir = Path("/tmp") / str(job["job_id"]) / "tartanair"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    log_path, metrics_path = _reserve_artifact_paths(
        artifact_dir,
        job.get("job_id"),
    )
    summary_path = temp_dir / "pipeline_summary.json"

    _job_log(log_path, f"TartanAir dataset download started: {job['job_id']}")
    command = _build_command(
        params=params,
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        temp_dir=temp_dir,
        summary_path=summary_path,
    )
    _job_log(log_path, "Command: " + " ".join(shlex.quote(part) for part in command))
    _stream_command(command, log_path)

    manifest_path = dataset_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset manifest was not produced: {manifest_path}")
    pipeline_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    completed_at = time.time()
    wall_time_ms = round((completed_at - started_at) * 1000, 3)
    metrics = [
        {
            "namespace": "performance",
            "name": "wall_time_ms",
            "type": "float",
            "value": wall_time_ms,
            "unit": "ms",
            "source": "runner",
        }
    ]
    _write_json(
        metrics_path,
        {
            "parameters": params,
            "dataset": {
                "name": dataset_name,
                "path": str(dataset_dir),
                "manifest": str(manifest_path),
            },
            "pipeline": pipeline_summary,
            "resource_metrics": metrics,
        },
    )
    _job_log(log_path, f"TartanAir dataset download completed in {wall_time_ms} ms")

    return {
        "status": "completed",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(completed_at)),
        "metrics": metrics,
        "artifacts": [
            {
                "artifact_type": "job_log",
                "path": log_path.relative_to(output_dir).as_posix(),
            },
            {
                "artifact_type": "metric_summary",
                "path": metrics_path.relative_to(output_dir).as_posix(),
            },
        ],
        "failure": None,
    }
