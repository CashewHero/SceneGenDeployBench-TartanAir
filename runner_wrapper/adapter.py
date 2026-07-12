from __future__ import annotations

import json
import os
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


def _build_command(
    *,
    params: dict[str, Any],
    dataset_name: str,
    dataset_dir: Path,
    temp_dir: Path,
    output_dir: Path,
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
        str(output_dir / "pipeline_summary.json"),
    ]


def run_job(job_request: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    job = job_request["job"]
    runtime = job_request["runtime"]
    config = job_request.get("config") or {}
    params = dict(job.get("parameters") or {})

    output_dir = Path(runtime["output_dir"])
    dataset_dir = Path(runtime["dataset_dir"])
    temp_dir = Path(runtime["temp_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "runner.log"
    metrics_path = output_dir / "metrics.json"
    dataset_name = str(config.get("dataset_name") or dataset_dir.name)

    _job_log(log_path, f"TartanAir dataset download started: {job['job_id']}")
    command = _build_command(
        params=params,
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        temp_dir=temp_dir,
        output_dir=output_dir,
    )
    _job_log(log_path, "Command: " + " ".join(shlex.quote(part) for part in command))
    _stream_command(command, log_path)

    manifest_path = dataset_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset manifest was not produced: {manifest_path}")
    summary_path = output_dir / "pipeline_summary.json"
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
            "dataset_name": dataset_name,
            "dataset_dir": str(dataset_dir),
            "manifest": str(manifest_path),
            "parameters": params,
            "pipeline": pipeline_summary,
            "metrics": metrics,
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
                "role": "stdout",
                "path": "runner.log",
                "format": "text",
                "size_bytes": log_path.stat().st_size,
            },
            {
                "artifact_type": "metric_summary",
                "role": "summary",
                "path": "metrics.json",
                "format": "json",
                "size_bytes": metrics_path.stat().st_size,
            },
        ],
        "failure": None,
    }
