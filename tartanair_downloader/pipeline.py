from __future__ import annotations

import shutil
import os
import json
import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tartanair_downloader.config import DownloadConfig
from tartanair_downloader.manifest import (
    DATASET_OWNER,
    merge_stream_manifests,
    rewrite_parent_manifests,
    write_manifest,
)
from tartanair_downloader.tartanair_api import download


def run(
    *,
    dataset_name: str,
    dataset_dir: Path,
    temp_dir: Path,
    params: dict[str, Any],
) -> dict[str, Any]:
    config = DownloadConfig.from_params(params)
    config = config.without_known_unsupported_modalities()
    if not config.modality:
        raise ValueError(f"mode={config.mode} has no modalities after filtering")
    _validate_publish_target(dataset_dir)
    if config.mode == "raw":
        return _run_raw(dataset_name=dataset_name, dataset_dir=dataset_dir, temp_dir=temp_dir, config=config)
    return _run_pano_dataset(
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        temp_dir=temp_dir,
        config=config,
    )


def _run_raw(
    *,
    dataset_name: str,
    dataset_dir: Path,
    temp_dir: Path,
    config: DownloadConfig,
) -> dict[str, Any]:
    staging_dir = temp_dir / "raw_download"
    _reset_dir(staging_dir)
    print(f"Downloading raw TartanAir dataset into staging: {staging_dir}", flush=True)
    download(staging_dir, config)
    _remove_non_dataset_artifacts(staging_dir)
    _prune_staged_dataset(staging_dir, config)
    sample_count = write_manifest(staging_dir, dataset_name, config)
    if sample_count <= 0:
        raise RuntimeError(f"TartanAir download produced no raw samples under {staging_dir}")
    _publish_dataset(staging_dir, dataset_dir, dataset_name)
    return {
        "mode": "raw",
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "manifest": str(dataset_dir / "manifest.yaml"),
        "sample_count": sample_count,
    }


def _run_pano_dataset(
    *,
    dataset_name: str,
    dataset_dir: Path,
    temp_dir: Path,
    config: DownloadConfig,
) -> dict[str, Any]:
    if config.mode == "equirectangular":
        staging_dir = temp_dir / "equirect_download"
        _reset_dir(staging_dir)
        print(f"Downloading TartanAir equirectangular panoramas into staging: {staging_dir}", flush=True)
        download(staging_dir, config)
        _remove_non_dataset_artifacts(staging_dir)
        _prune_staged_dataset(staging_dir, config)
        sample_count = write_manifest(staging_dir, dataset_name, config)
        if sample_count <= 0:
            raise RuntimeError(f"TartanAir download produced no pano samples under {staging_dir}")
        _publish_dataset(staging_dir, dataset_dir, dataset_name)
        return {
            "mode": "equirectangular",
            "dataset_name": dataset_name,
            "dataset_dir": str(dataset_dir),
            "manifest": str(dataset_dir / "manifest.yaml"),
            "sample_count": sample_count,
        }

    raw_root = temp_dir / "raw_cube"
    converted_root = temp_dir / "converted_pano"
    _reset_dir(raw_root)
    _reset_dir(converted_root)
    print(f"Downloading TartanAir cube faces into {raw_root}", flush=True)
    download(raw_root, config)
    _prune_staged_dataset(raw_root, config)
    print(f"Converting cube faces into panoramas under staging: {converted_root}", flush=True)
    from tartanair_downloader.pano_conversion import convert_cube_dataset

    sequence_samples = convert_cube_dataset(raw_root=raw_root, dataset_dir=converted_root, config=config)
    sample_count = write_manifest(converted_root, dataset_name, config, sequence_samples)
    if sample_count <= 0:
        raise RuntimeError(f"TartanAir cube conversion produced no pano samples under {converted_root}")
    _publish_dataset(converted_root, dataset_dir, dataset_name)
    if raw_root.exists():
        shutil.rmtree(raw_root)
        print(f"Deleted temporary raw cube data: {raw_root}", flush=True)
    return {
        "mode": "pano_conversion",
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "manifest": str(dataset_dir / "manifest.yaml"),
        "sample_count": sample_count,
    }


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _remove_non_dataset_artifacts(path: Path) -> None:
    for cache_dir in path.rglob(".cache"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
    for zip_path in path.rglob("*.zip"):
        if zip_path.is_file():
            zip_path.unlink()


def _prune_staged_dataset(path: Path, config: DownloadConfig) -> None:
    allowed_envs = set(config.env)
    allowed_difficulties = {f"Data_{difficulty}" for difficulty in config.difficulty}
    allowed_trajectories = set(config.trajectory or [])

    for env_dir in sorted(item for item in path.iterdir() if item.is_dir()):
        if env_dir.name not in allowed_envs:
            shutil.rmtree(env_dir)
            continue
        for difficulty_dir in sorted(item for item in env_dir.iterdir() if item.is_dir() and item.name.startswith("Data_")):
            if difficulty_dir.name not in allowed_difficulties:
                shutil.rmtree(difficulty_dir)
                continue
            if not allowed_trajectories:
                continue
            for trajectory_dir in sorted(item for item in difficulty_dir.iterdir() if item.is_dir()):
                if trajectory_dir.name not in allowed_trajectories:
                    shutil.rmtree(trajectory_dir)


def _publish_dataset(staging_dir: Path, dataset_dir: Path, dataset_name: str) -> None:
    if not (staging_dir / "manifest.yaml").is_file():
        raise FileNotFoundError(f"staged dataset manifest was not produced: {staging_dir / 'manifest.yaml'}")

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    with _publish_lock(dataset_dir):
        _validate_publish_target(dataset_dir)
        publish_dir = dataset_dir.parent / f".{dataset_dir.name}.publishing-{os.getpid()}"
        if publish_dir.exists():
            shutil.rmtree(publish_dir)
        if dataset_dir.exists():
            merge_stream_manifests(dataset_dir, staging_dir)
            shutil.copytree(dataset_dir, publish_dir)
            shutil.copytree(staging_dir, publish_dir, dirs_exist_ok=True)
            rewrite_parent_manifests(publish_dir, dataset_name)
        else:
            shutil.copytree(staging_dir, publish_dir)
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        publish_dir.rename(dataset_dir)
    print(f"Published final dataset to {dataset_dir}", flush=True)


@contextmanager
def _publish_lock(dataset_dir: Path):
    lock_path = dataset_dir.parent / f".{dataset_dir.name}.publish.lock"
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _validate_publish_target(dataset_dir: Path) -> None:
    if not dataset_dir.exists() or _is_empty_dir(dataset_dir):
        return

    manifest_path = dataset_dir / "manifest.yaml"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"refusing to publish into non-empty dataset without manifest: {dataset_dir}"
        )

    owner = _read_dataset_owner(manifest_path)
    if owner != DATASET_OWNER:
        raise RuntimeError(
            f"refusing to publish into dataset owned by {owner or 'unknown'}; "
            f"expected metadata.dataset_owner: {DATASET_OWNER}"
        )


def _is_empty_dir(path: Path) -> bool:
    return not any(path.iterdir())


def _read_dataset_owner(manifest_path: Path) -> str | None:
    in_metadata = False
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            in_metadata = line.strip() == "metadata:"
            continue
        if in_metadata and line.startswith("  dataset_owner:"):
            raw_value = line.split(":", 1)[1].strip()
            if not raw_value:
                return None
            try:
                return str(json.loads(raw_value))
            except json.JSONDecodeError:
                return raw_value.strip("'\"")
    return None
