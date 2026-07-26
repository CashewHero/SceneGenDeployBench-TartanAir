from __future__ import annotations

from pathlib import Path

from tartanair_downloader.config import DownloadConfig

DOWNLOAD_SOURCES = ("huggingface", "airlab")


def download(root: Path, config: DownloadConfig) -> None:
    import tartanair as ta

    root.mkdir(parents=True, exist_ok=True)
    for modality in config.modality:
        config.warn_if_raw_camera_adjusted(modality)
        _download_modality(root, config, ta, modality)
    _verify_download(root, config)


def _download_modality(root: Path, config: DownloadConfig, ta: object, modality: str) -> None:
    errors: list[str] = []
    for data_source in DOWNLOAD_SOURCES:
        try:
            print(f"Downloading TartanAir {modality} from {data_source}", flush=True)
            _run_tartanair_download(root, config, ta, modality, data_source)
            _verify_modality(root, config, modality)
        except Exception as exc:
            errors.append(f"{data_source}: {exc}")
            if data_source != DOWNLOAD_SOURCES[-1]:
                next_source = DOWNLOAD_SOURCES[DOWNLOAD_SOURCES.index(data_source) + 1]
                print(
                    f"Warning: TartanAir {modality} download from {data_source} failed; "
                    f"retrying with {next_source}",
                    flush=True,
                )
            continue
        return

    raise RuntimeError(f"TartanAir {modality} download failed from all sources: {'; '.join(errors)}")


def _run_tartanair_download(root: Path, config: DownloadConfig, ta: object, modality: str, data_source: str) -> None:
    ta.init(str(root))
    ta.download(
        env=config.env,
        difficulty=config.difficulty,
        modality=[modality],
        camera_name=config.camera_names_for_download(modality),
        unzip=True,
        delete_zip=True,
        num_workers=config.download_workers,
        data_source=data_source,
    )


def _verify_modality(root: Path, config: DownloadConfig, modality: str) -> None:
    missing: list[str] = []
    for env_name in config.env:
        for difficulty in config.difficulty:
            difficulty_dir = root / env_name / f"Data_{difficulty}"
            trajectories = _trajectory_dirs(difficulty_dir, config)
            if not trajectories:
                missing.append(str(difficulty_dir.relative_to(root)))
                continue
            missing.extend(
                str(path.relative_to(root))
                for path in _missing_modality_paths(
                    trajectories,
                    modality,
                    config.camera_names_for_download(modality),
                )
            )
    if missing:
        preview = ", ".join(missing[:10])
        extra = "" if len(missing) <= 10 else f", ... ({len(missing)} missing total)"
        raise RuntimeError(f"missing data folders: {preview}{extra}")


def _verify_download(root: Path, config: DownloadConfig) -> None:
    for modality in config.modality:
        _verify_modality(root, config, modality)


def _trajectory_dirs(difficulty_dir: Path, config: DownloadConfig) -> list[Path]:
    if not difficulty_dir.is_dir():
        return []
    if config.trajectory is None:
        return sorted(path for path in difficulty_dir.iterdir() if path.is_dir())
    return [difficulty_dir / trajectory for trajectory in config.trajectory if (difficulty_dir / trajectory).is_dir()]


def _missing_modality_paths(trajectory_dirs: list[Path], modality: str, camera_names: list[str]) -> list[Path]:
    missing: list[Path] = []
    for trajectory_dir in trajectory_dirs:
        if not _has_any_modality_artifact(trajectory_dir, modality):
            missing.append(trajectory_dir / modality)
            continue

        camera_specific = {
            camera_name: _has_camera_modality_artifact(trajectory_dir, modality, camera_name)
            for camera_name in camera_names
        }
        if any(camera_specific.values()):
            missing.extend(
                trajectory_dir / f"{modality}_{camera_name}"
                for camera_name, exists in camera_specific.items()
                if not exists
            )
    return missing


def _has_any_modality_artifact(trajectory_dir: Path, modality: str) -> bool:
    return any(
        _has_data(path) and (path.name == modality or path.name.startswith(f"{modality}_"))
        for path in trajectory_dir.iterdir()
    )


def _has_camera_modality_artifact(trajectory_dir: Path, modality: str, camera_name: str) -> bool:
    prefixes = (f"{modality}_{camera_name}", f"{modality}-{camera_name}")
    return any(
        _has_data(path) and (path.name.startswith(prefixes) or path.stem.startswith(prefixes))
        for path in trajectory_dir.iterdir()
    )


def _has_data(path: Path) -> bool:
    if path.is_file():
        return not path.name.startswith(".")
    if path.is_dir():
        return any(child.is_file() and not child.name.startswith(".") for child in path.iterdir())
    return False
