from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tartanair_downloader.config import DownloadConfig
from tartanair_downloader.manifest_common import scene_tags, write_yaml


DEPTH_METADATA = {
    "format": "png",
    "dtype": "float32",
    "encoding": "tartanair_float32_rgba",
    "units": "meters",
}
DATASET_OWNER = "SceneGenDeployBench-TartanAir"


@dataclass(frozen=True)
class StreamSpec:
    env_name: str
    difficulty: str
    trajectory: str
    stream_name: str
    metadata: dict[str, Any]
    samples: dict[str, dict[str, Any]]
    data_types: set[str]


def write_manifest(
    dataset_dir: Path,
    dataset_name: str,
    config: DownloadConfig,
    sequence_samples: dict[tuple[str, str], dict[str, dict[str, Any]]] | None = None,
) -> int:
    if config.mode == "raw":
        streams = _raw_streams(dataset_dir, config)
        tags = ["raw", "multi-camera"]
    elif config.mode == "equirectangular":
        streams = _equirectangular_streams(dataset_dir, config)
        tags = ["panorama", "multi-viewpoint"]
    else:
        streams = _pano_conversion_streams(dataset_dir, config, sequence_samples or {})
        tags = ["panorama", "multi-viewpoint"]
    return _write_stream_tree(dataset_dir, dataset_name, streams, tags)


def rewrite_parent_manifests(dataset_dir: Path, dataset_name: str) -> int:
    env_sequences: dict[str, dict[str, str]] = {}
    data_types: set[str] = set()
    total_streams = 0
    has_raw = False
    has_pano = False

    for env_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        for difficulty_dir in sorted(env_dir.glob("Data_*")):
            difficulty = difficulty_dir.name.removeprefix("Data_")
            for sequence_dir in sorted(path for path in difficulty_dir.iterdir() if path.is_dir()):
                stream_subsets: dict[str, str] = {}
                for stream_manifest in sorted(sequence_dir.glob("*.yaml")):
                    if stream_manifest.name == "manifest.yaml":
                        continue
                    stream_name = stream_manifest.stem
                    stream_subsets[stream_name] = stream_manifest.name
                    has_pano = has_pano or stream_name.endswith(("_equirect", "_pano_conversion"))
                    has_raw = has_raw or not stream_name.endswith(("_equirect", "_pano_conversion"))
                if not stream_subsets:
                    continue
                data_types.update(_sequence_data_types(sequence_dir))
                write_yaml(
                    sequence_dir / "manifest.yaml",
                    {
                        "metadata": {
                            "difficulty": difficulty,
                            "trajectory": sequence_dir.name,
                        },
                        "tags": ["sequence", difficulty],
                        "subsets": dict(sorted(stream_subsets.items())),
                    },
                )
                env_sequences.setdefault(env_dir.name, {})[f"{difficulty}/{sequence_dir.name}"] = (
                    Path(difficulty_dir.name) / sequence_dir.name / "manifest.yaml"
                ).as_posix()
                total_streams += len(stream_subsets)

    env_subsets: dict[str, str] = {}
    for env_name, subsets in sorted(env_sequences.items()):
        write_yaml(
            dataset_dir / env_name / "manifest.yaml",
            {
                "metadata": {"scene": env_name},
                "tags": scene_tags(env_name),
                "subsets": dict(sorted(subsets.items())),
            },
        )
        env_subsets[env_name] = f"{env_name}/manifest.yaml"

    tags: list[str] = []
    if has_raw:
        tags.extend(["raw", "multi-camera"])
    if has_pano:
        tags.extend(["panorama", "multi-viewpoint"])
    if not tags:
        tags = ["tartanair"]

    write_yaml(
        dataset_dir / "manifest.yaml",
        {
            "dataset_name": dataset_name,
            "dataset_version": "0.1.0",
            "data_types": sorted(data_types),
            "metadata": {
                "dataset_owner": DATASET_OWNER,
                "pose_coordinate_system": "NED",
                "pose_convention": "camera_to_world",
                "pose_units": "meters",
            },
            "tags": list(dict.fromkeys(tags)),
            "subsets": env_subsets,
        },
    )
    return total_streams


def merge_stream_manifests(existing_dir: Path, staged_dir: Path) -> None:
    """Merge leaf stream manifests that would otherwise be overwritten on publish."""
    for staged_manifest in sorted(staged_dir.rglob("*.yaml")):
        if staged_manifest.name == "manifest.yaml":
            continue
        relative_path = staged_manifest.relative_to(staged_dir)
        existing_manifest = existing_dir / relative_path
        if not existing_manifest.is_file():
            continue
        existing = _read_yaml_mapping(existing_manifest)
        staged = _read_yaml_mapping(staged_manifest)
        write_yaml(staged_manifest, _merge_stream_payloads(existing, staged, relative_path))


def _write_stream_tree(
    dataset_dir: Path,
    dataset_name: str,
    streams: list[StreamSpec],
    tags: list[str],
) -> int:
    env_sequences: dict[str, dict[str, str]] = {}
    sequence_streams: dict[tuple[str, str, str], dict[str, str]] = {}
    total_samples = 0
    data_types: set[str] = set()

    for stream in sorted(streams, key=lambda item: (item.env_name, item.difficulty, item.trajectory, item.stream_name)):
        if not stream.samples:
            continue
        sequence_dir = dataset_dir / stream.env_name / f"Data_{stream.difficulty}" / stream.trajectory
        payload: dict[str, Any] = {
            "metadata": _ordered_metadata(stream.metadata),
            "tags": ["stream"],
            "samples": _ordered_samples(stream.samples),
        }
        write_yaml(sequence_dir / f"{stream.stream_name}.yaml", payload)
        sequence_key = (stream.env_name, stream.difficulty, stream.trajectory)
        sequence_streams.setdefault(sequence_key, {})[stream.stream_name] = f"{stream.stream_name}.yaml"
        total_samples += len(stream.samples)
        data_types.update(stream.data_types)
        if any("camera_pose" in sample for sample in stream.samples.values()):
            data_types.add("camera_pose")

    for (env_name, difficulty, trajectory), subsets in sorted(sequence_streams.items()):
        sequence_dir = dataset_dir / env_name / f"Data_{difficulty}" / trajectory
        write_yaml(
            sequence_dir / "manifest.yaml",
            {
                "metadata": {
                    "difficulty": difficulty,
                    "trajectory": trajectory,
                },
                "tags": ["sequence", difficulty],
                "subsets": dict(sorted(subsets.items())),
            },
        )
        env_sequences.setdefault(env_name, {})[f"{difficulty}/{trajectory}"] = (
            Path(f"Data_{difficulty}") / trajectory / "manifest.yaml"
        ).as_posix()

    env_subsets: dict[str, str] = {}
    for env_name, subsets in sorted(env_sequences.items()):
        write_yaml(
            dataset_dir / env_name / "manifest.yaml",
            {
                "metadata": {"scene": env_name},
                "tags": scene_tags(env_name),
                "subsets": dict(sorted(subsets.items())),
            },
        )
        env_subsets[env_name] = f"{env_name}/manifest.yaml"

    write_yaml(
        dataset_dir / "manifest.yaml",
        {
            "dataset_name": dataset_name,
            "dataset_version": "0.1.0",
            "data_types": sorted(data_types),
            "metadata": {
                "dataset_owner": DATASET_OWNER,
                "pose_coordinate_system": "NED",
                "pose_convention": "camera_to_world",
                "pose_units": "meters",
            },
            "tags": tags,
            "subsets": env_subsets,
        },
    )
    return total_samples


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected YAML mapping in stream manifest: {path}")
    return payload


def _merge_stream_payloads(
    existing: dict[str, Any],
    staged: dict[str, Any],
    relative_path: Path,
) -> dict[str, Any]:
    metadata = _merge_mappings(
        _mapping_value(existing, "metadata", relative_path),
        _mapping_value(staged, "metadata", relative_path),
        f"{relative_path}:metadata",
    )
    samples = _merge_mappings(
        _mapping_value(existing, "samples", relative_path),
        _mapping_value(staged, "samples", relative_path),
        f"{relative_path}:samples",
    )
    tags = sorted(
        {
            str(tag)
            for payload in (existing, staged)
            for tag in payload.get("tags", [])
        }
    )
    return {
        "metadata": _ordered_metadata(metadata),
        "tags": tags,
        "samples": _ordered_samples(samples),
    }


def _mapping_value(payload: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise RuntimeError(f"expected {key} mapping in stream manifest: {path}")
    return value


def _merge_mappings(existing: dict[str, Any], staged: dict[str, Any], location: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in sorted(existing.keys() | staged.keys()):
        if key not in existing:
            merged[key] = staged[key]
            continue
        if key not in staged:
            merged[key] = existing[key]
            continue
        old_value = existing[key]
        new_value = staged[key]
        if isinstance(old_value, dict) and isinstance(new_value, dict):
            merged[key] = _merge_mappings(old_value, new_value, f"{location}.{key}")
        elif old_value == new_value:
            merged[key] = old_value
        else:
            raise RuntimeError(
                f"conflicting stream manifest value at {location}.{key}: "
                f"existing={old_value!r}, staged={new_value!r}"
            )
    return merged


def _ordered_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    preferred = ("source", "camera_side", "projection", "cube_face", "resolution", "fov", "depth")
    return {
        key: metadata[key]
        for key in (*preferred, *sorted(set(metadata) - set(preferred)))
        if key in metadata
    }


def _ordered_samples(samples: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ordered: dict[str, dict[str, Any]] = {}
    for sample_id, sample in sorted(samples.items()):
        if not isinstance(sample, dict):
            raise RuntimeError(f"expected sample mapping for {sample_id}")
        keys = sorted(key for key in sample if key != "camera_pose")
        if "camera_pose" in sample:
            keys.append("camera_pose")
        ordered[sample_id] = {key: sample[key] for key in keys}
    return ordered


def _raw_streams(dataset_dir: Path, config: DownloadConfig) -> list[StreamSpec]:
    streams: list[StreamSpec] = []
    for env_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir() and path.name in config.env):
        for difficulty_dir in sorted(env_dir.glob("Data_*")):
            difficulty = difficulty_dir.name.removeprefix("Data_")
            if difficulty not in config.difficulty:
                continue
            for sequence_dir in sorted(path for path in difficulty_dir.iterdir() if path.is_dir()):
                if config.trajectory is not None and sequence_dir.name not in config.trajectory:
                    continue
                streams.extend(_raw_sequence_streams(sequence_dir, env_dir.name, difficulty))
    return streams


def _raw_sequence_streams(sequence_dir: Path, env_name: str, difficulty: str) -> list[StreamSpec]:
    pose_cache: dict[Path, list[list[float]]] = {}
    grouped_samples: dict[str, dict[str, dict[str, Any]]] = {}
    grouped_data_types: dict[str, set[str]] = {}
    grouped_folders: dict[str, list[Path]] = {}
    grouped_stream: dict[str, dict[str, str]] = {}

    for folder in sorted(path for path in sequence_dir.iterdir() if _is_stream_dir(path)):
        stream = _parse_raw_stream(folder.name)
        data_type = _data_type(stream["modality"])
        stream_name = stream["camera_name"] or data_type
        grouped_stream.setdefault(stream_name, stream)
        grouped_data_types.setdefault(stream_name, set()).add(data_type)
        grouped_folders.setdefault(stream_name, []).append(folder)
        samples = grouped_samples.setdefault(stream_name, {})
        for file_path in sorted(path for path in folder.iterdir() if _is_data_file(path)):
            sample_id = _sample_id(file_path.stem)
            sample = samples.setdefault(sample_id, {})
            sample[data_type] = (Path(folder.name) / file_path.name).as_posix()
            pose = _camera_pose(sequence_dir, stream["camera_name"], sample_id, pose_cache)
            if pose is not None:
                sample["camera_pose"] = pose

    streams: list[StreamSpec] = []
    for stream_name, samples in sorted(grouped_samples.items()):
        if not samples:
            continue
        stream = grouped_stream[stream_name]
        data_types = grouped_data_types[stream_name]
        streams.append(
            StreamSpec(
                env_name=env_name,
                difficulty=difficulty,
                trajectory=sequence_dir.name,
                stream_name=stream_name,
                metadata=_stream_metadata(
                    source="tartanair_raw",
                    data_types=data_types,
                    camera_side=_camera_side(stream["camera_side_raw"]),
                    projection=_projection(stream["projection_raw"]),
                    cube_face=_cube_face(stream["projection_raw"]),
                    resolution=_folders_resolution(grouped_folders[stream_name]),
                ),
                samples=samples,
                data_types=data_types,
            )
        )
    return streams


def _equirectangular_streams(dataset_dir: Path, config: DownloadConfig) -> list[StreamSpec]:
    streams: list[StreamSpec] = []
    pose_cache: dict[Path, list[list[float]]] = {}
    allowed_folders = set(config.camera_names_for_download())
    for env_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir() and path.name in config.env):
        for difficulty_dir in sorted(env_dir.glob("Data_*")):
            difficulty = difficulty_dir.name.removeprefix("Data_")
            if difficulty not in config.difficulty:
                continue
            for sequence_dir in sorted(path for path in difficulty_dir.iterdir() if path.is_dir()):
                if config.trajectory is not None and sequence_dir.name not in config.trajectory:
                    continue
                folders = [
                    path
                    for path in sorted(sequence_dir.iterdir())
                    if path.is_dir() and path.name.endswith("_equirect")
                ]
                for folder in folders:
                    modality, camera_name = _modality_camera_name(folder.name)
                    if modality not in config.modality or camera_name not in allowed_folders:
                        continue
                    camera_prefix = camera_name.removesuffix("_equirect")
                    stream_name = camera_name
                    data_type = _data_type(modality)
                    stream = _stream_by_name(streams, env_dir.name, difficulty, sequence_dir.name, stream_name)
                    if stream is None:
                        stream = StreamSpec(
                            env_name=env_dir.name,
                            difficulty=difficulty,
                            trajectory=sequence_dir.name,
                            stream_name=stream_name,
                            metadata=_stream_metadata(
                                source="tartanair_equirectangular",
                                data_types=set(),
                                camera_side=_camera_side(camera_prefix),
                                projection="equirectangular",
                                resolution=None,
                            ),
                            samples={},
                            data_types=set(),
                        )
                        streams.append(stream)
                    for file_path in sorted(path for path in folder.iterdir() if _is_data_file(path)):
                        sample_id = _sample_id(file_path.stem)
                        sample = stream.samples.setdefault(sample_id, {})
                        sample[data_type] = (Path(folder.name) / file_path.name).as_posix()
                        pose = _equirect_pose(sequence_dir, camera_prefix, sample_id, pose_cache)
                        if pose is not None:
                            sample["camera_pose"] = pose
                    stream.data_types.add(data_type)
                    stream.metadata.update(
                        _stream_metadata(
                            source="tartanair_equirectangular",
                            data_types=stream.data_types,
                            camera_side=_camera_side(camera_prefix),
                            projection="equirectangular",
                            resolution=stream.metadata.get("resolution") or _folder_resolution(folder),
                        )
                    )
    return streams


def _pano_conversion_streams(
    dataset_dir: Path,
    config: DownloadConfig,
    sequence_samples: dict[tuple[str, str], dict[str, dict[str, Any]]],
) -> list[StreamSpec]:
    stream_samples: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    stream_data_types: dict[tuple[str, str, str, str], set[str]] = {}
    stream_folders: dict[tuple[str, str, str, str], set[str]] = {}
    for (env_name, sequence_name), samples in sorted(sequence_samples.items()):
        difficulty, _, trajectory = sequence_name.partition("/")
        for sample_id, sample in sorted(samples.items()):
            for data_type, value in sample.items():
                if data_type == "camera_pose" or not isinstance(value, str):
                    continue
                data_folder = Path(value).parts[0]
                stream_name = _pano_conversion_stream_name(data_folder)
                key = (env_name, difficulty, trajectory, stream_name)
                stream_sample = stream_samples.setdefault(key, {}).setdefault(sample_id, {})
                stream_sample[data_type] = value
                if "camera_pose" in sample:
                    stream_sample["camera_pose"] = sample["camera_pose"]
                stream_data_types.setdefault(key, set()).add(data_type)
                stream_folders.setdefault(key, set()).add(data_folder)

    streams: list[StreamSpec] = []
    for (env_name, difficulty, trajectory, stream_name), samples in sorted(stream_samples.items()):
        sequence_dir = dataset_dir / env_name / f"Data_{difficulty}" / trajectory
        data_types = stream_data_types[(env_name, difficulty, trajectory, stream_name)]
        streams.append(
            StreamSpec(
                env_name=env_name,
                difficulty=difficulty,
                trajectory=trajectory,
                stream_name=stream_name,
                metadata=_stream_metadata(
                    source="tartanair_pano_conversion",
                    data_types=data_types,
                    camera_side=_camera_side_from_stream_name(stream_name),
                    projection="equirectangular",
                    resolution=_folders_resolution(
                        [sequence_dir / folder for folder in stream_folders[(env_name, difficulty, trajectory, stream_name)]]
                    )
                    or [config.pano_width, config.pano_height],
                ),
                samples=samples,
                data_types=data_types,
            )
        )
    return streams


def _parse_raw_stream(folder_name: str) -> dict[str, str]:
    parts = folder_name.split("_")
    if len(parts) < 3:
        return {
            "modality": parts[0],
            "camera_side_raw": "",
            "projection_raw": "",
            "camera_name": "",
        }
    return {
        "modality": parts[0],
        "camera_side_raw": parts[1],
        "projection_raw": parts[2],
        "camera_name": "_".join(parts[1:]),
    }


def _modality_camera_name(folder_name: str) -> tuple[str, str]:
    modality, _, camera_name = folder_name.partition("_")
    return modality, camera_name


def _data_type(modality: str) -> str:
    return modality


def _pano_conversion_stream_name(data_folder: str) -> str:
    parts = data_folder.split("_", 1)
    return parts[1] if len(parts) == 2 else data_folder


def _stream_by_name(
    streams: list[StreamSpec],
    env_name: str,
    difficulty: str,
    trajectory: str,
    stream_name: str,
) -> StreamSpec | None:
    for stream in streams:
        if (
            stream.env_name == env_name
            and stream.difficulty == difficulty
            and stream.trajectory == trajectory
            and stream.stream_name == stream_name
        ):
            return stream
    return None


def _sample_id(stem: str) -> str:
    frame = stem.split("_", 1)[0]
    if frame.isdigit():
        return f"frame_{frame}"
    return frame if frame else stem


def _stream_metadata(
    *,
    source: str,
    data_types: set[str],
    camera_side: str | None,
    projection: str | None,
    resolution: list[int] | None,
    cube_face: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"source": source}
    if camera_side:
        metadata["camera_side"] = camera_side
    if projection:
        metadata["projection"] = projection
    if cube_face:
        metadata["cube_face"] = cube_face
    if resolution:
        metadata["resolution"] = resolution
    fov = _projection_fov(projection)
    if fov:
        metadata["fov"] = fov
    if "depth" in data_types:
        metadata["depth"] = dict(DEPTH_METADATA)
    return metadata


def _camera_side(value: str) -> str | None:
    if value == "lcam":
        return "left"
    if value == "rcam":
        return "right"
    return None


def _camera_side_from_stream_name(stream_name: str) -> str | None:
    parts = stream_name.split("_")
    if len(parts) >= 2:
        return _camera_side(parts[1])
    return None


def _projection(value: str) -> str | None:
    if value in {"front", "left", "right", "back", "top", "bottom"}:
        return "pinhole"
    if value == "fish":
        return "fisheye"
    if value in {"equirect", "equirectangular", "pano_conversion"}:
        return "equirectangular"
    return None


def _cube_face(value: str) -> str | None:
    return value if value in {"front", "left", "right", "back", "top", "bottom"} else None


def _projection_fov(projection: str | None) -> list[int] | None:
    if projection == "pinhole":
        return [90, 90]
    if projection == "equirectangular":
        return [360, 180]
    return None


def _folder_resolution(folder: Path) -> list[int] | None:
    for file_path in sorted(path for path in folder.iterdir() if _is_data_file(path)):
        resolution = _png_resolution(file_path)
        if resolution:
            return resolution
    return None


def _folders_resolution(folders: list[Path]) -> list[int] | None:
    for folder in sorted(folders):
        if folder.is_dir():
            resolution = _folder_resolution(folder)
            if resolution:
                return resolution
    return None


def _sequence_data_types(sequence_dir: Path) -> set[str]:
    data_types: set[str] = set()
    for folder in sorted(path for path in sequence_dir.iterdir() if _is_stream_dir(path)):
        data_type, _, _ = folder.name.partition("_")
        if data_type:
            data_types.add(_data_type(data_type))
    for manifest_path in sorted(sequence_dir.glob("*.yaml")):
        if manifest_path.name == "manifest.yaml":
            continue
        payload = _read_yaml_mapping(manifest_path)
        samples = payload.get("samples")
        if isinstance(samples, dict) and any(
            isinstance(sample, dict) and "camera_pose" in sample for sample in samples.values()
        ):
            data_types.add("camera_pose")
    return data_types


def _png_resolution(path: Path) -> list[int] | None:
    if path.suffix.lower() != ".png":
        return None
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return [int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")]


def _is_stream_dir(path: Path) -> bool:
    return path.is_dir() and path.name != "streams" and not path.name.startswith(".")


def _is_data_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() not in {".yaml", ".yml"} and not path.name.startswith(".")


def _camera_pose(
    sequence_dir: Path,
    camera_name: str,
    sample_id: str,
    pose_cache: dict[Path, list[list[float]]],
) -> dict[str, list[float]] | None:
    if not camera_name:
        return None
    pose_path = sequence_dir / f"pose_{camera_name}.txt"
    if not pose_path.is_file():
        if camera_name.endswith("_equirect") or camera_name.endswith("_fish"):
            pose_path = sequence_dir / f"pose_{camera_name.rsplit('_', 1)[0]}_front.txt"
        if not pose_path.is_file():
            return None
    frame_idx = _frame_index(sample_id)
    if frame_idx is None:
        return None
    poses = pose_cache.setdefault(pose_path, _read_poses(pose_path))
    if frame_idx >= len(poses):
        return None
    pose = poses[frame_idx]
    return {
        "position": pose[:3],
        "rotation_quaternion_xyzw": pose[3:7],
    }


def _equirect_pose(
    sequence_dir: Path,
    camera_prefix: str,
    sample_id: str,
    pose_cache: dict[Path, list[list[float]]],
) -> dict[str, list[float]] | None:
    pose_path = sequence_dir / f"pose_{camera_prefix}_front.txt"
    if not pose_path.is_file():
        return None
    frame_idx = _frame_index(sample_id)
    if frame_idx is None:
        return None
    poses = pose_cache.setdefault(pose_path, _read_poses(pose_path))
    if frame_idx >= len(poses):
        return None
    pose = poses[frame_idx]
    return {
        "position": pose[:3],
        "rotation_quaternion_xyzw": pose[3:7],
    }


def _frame_index(sample_id: str) -> int | None:
    frame = sample_id.removeprefix("frame_").split("_", 1)[0]
    return int(frame) if frame.isdigit() else None


def _read_poses(path: Path) -> list[list[float]]:
    poses: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        pose = [float(value) for value in line.split()]
        if len(pose) != 7:
            raise RuntimeError(f"expected 7 pose values in {path} line {line_number}")
        poses.append(pose)
    return poses
