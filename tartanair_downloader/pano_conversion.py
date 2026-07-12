from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from tartanair_downloader.config import CUBE_FACES, DownloadConfig


_WORLD_VEC_CACHE: dict[tuple[str, int | None, int, int], torch.Tensor] = {}


def convert_cube_dataset(
    *,
    raw_root: Path,
    dataset_dir: Path,
    config: DownloadConfig,
) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    groups = _conversion_groups(raw_root=raw_root, dataset_dir=dataset_dir, config=config)
    total_tasks = sum(len(tasks) for tasks in groups)
    if total_tasks <= 0:
        raise RuntimeError("no cube panorama conversion tasks found")

    cuda_available = torch.cuda.is_available()
    device = "cuda" if config.pano_cuda and cuda_available else "cpu"
    print(
        "Panorama conversion: "
        f"groups={len(groups)}, frames={total_tasks}, "
        f"workers={config.pano_convert_workers}, device={device}, "
        f"cuda_requested={config.pano_cuda}, cuda_available={cuda_available}, "
        f"resolution={config.pano_width}x{config.pano_height}",
        flush=True,
    )
    results: list[tuple[Path, str, str, str, dict[str, list[float]]]]
    results = []
    for group_index, tasks in enumerate(groups, start=1):
        group = tasks[0]
        print(
            "Panorama conversion group "
            f"{group_index}/{len(groups)}: {group['env_name']}/Data_{group['difficulty']}/{group['trajectory']} "
            f"modality={group['modality']} camera={group['camera']} frames={len(tasks)}",
            flush=True,
        )
        results.extend(_run_tasks(tasks, config.pano_convert_workers))
        print(
            "Panorama conversion group done: "
            f"{group['env_name']}/Data_{group['difficulty']}/{group['trajectory']} "
            f"modality={group['modality']} camera={group['camera']}",
            flush=True,
        )

    samples: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for output_file, env_name, sequence_name, data_type, camera_pose in results:
        difficulty, _, trajectory = sequence_name.partition("/")
        sequence_dir = dataset_dir / env_name / f"Data_{difficulty}" / trajectory
        sample_id = f"frame_{output_file.stem.split('_', 1)[0]}"
        sample = samples.setdefault((env_name, sequence_name), {}).setdefault(
            sample_id,
            {"camera_pose": camera_pose},
        )
        sample[data_type] = output_file.relative_to(sequence_dir).as_posix()
    print(f"Panorama conversion completed: {len(groups)} groups, {len(results)} frames", flush=True)
    return samples


def _conversion_groups(*, raw_root: Path, dataset_dir: Path, config: DownloadConfig) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    direction_order = ("right", "left", "top", "bottom", "front", "back")
    for env_name in config.env:
        env_dir = raw_root / env_name
        for difficulty in config.difficulty:
            diff_dir = env_dir / f"Data_{difficulty}"
            if not diff_dir.exists():
                continue
            trajectories = (
                [path.name for path in sorted(diff_dir.iterdir()) if path.is_dir()]
                if config.trajectory is None
                else config.trajectory
            )
            for trajectory in trajectories:
                traj_dir = diff_dir / trajectory
                if not traj_dir.exists():
                    continue
                for modality in config.modality:
                    for camera in config.camera_sides_for_cube_conversion():
                        camera_prefix = _camera_prefix(camera)
                        face_files = _discover_cube_face_files(
                            traj_dir=traj_dir,
                            modality=modality,
                            camera_prefix=camera_prefix,
                            direction_order=direction_order,
                        )
                        if not face_files:
                            print(
                                "Warning: no complete cube-face frames found for "
                                f"{env_name}/Data_{difficulty}/{trajectory} "
                                f"modality={modality} camera={camera}",
                                flush=True,
                            )
                        tasks: list[dict[str, Any]] = []
                        for frame_id, files_by_direction in sorted(face_files.items()):
                            data_type = _data_type(modality)
                            stream_name = f"{data_type}_{camera_prefix}_pano_conversion"
                            suffix = "" if data_type == "image" else f"_{data_type}"
                            output_file = (
                                dataset_dir
                                / env_name
                                / f"Data_{difficulty}"
                                / trajectory
                                / stream_name
                                / f"{frame_id}_{camera_prefix}_pano_conversion{suffix}.png"
                            )
                            tasks.append(
                                {
                                    "output_file": output_file,
                                    "raw_root": raw_root,
                                    "env_name": env_name,
                                    "modality": modality,
                                    "data_type": data_type,
                                    "resolution": (config.pano_width, config.pano_height),
                                    "difficulty": difficulty,
                                    "camera": camera,
                                    "trajectory": trajectory,
                                    "frame_id": frame_id,
                                    "face_files": files_by_direction,
                                    "cuda": 1 if config.pano_cuda else 0,
                                    "png_compression": config.pano_png_compression,
                                }
                            )
                        if tasks:
                            groups.append(tasks)
    return groups


def _print_progress(done: int, total: int) -> None:
    milestones = _progress_milestones(total)
    if done in milestones:
        label = _format_percent(done, total)
        print(f"  progress: {done}/{total} frames ({label})", flush=True)


def _progress_milestones(total: int) -> set[int]:
    if total <= 0:
        return set()
    return {
        1,
        _ceil_percent(total, 20),
        _ceil_percent(total, 40),
        _ceil_percent(total, 60),
        _ceil_percent(total, 80),
        total,
    }


def _ceil_percent(total: int, percent: int) -> int:
    return max(1, (total * percent + 99) // 100)


def _format_percent(done: int, total: int) -> str:
    percent = done / total * 100
    return f"{percent:.0f}%"


def _run_tasks(tasks: list[dict[str, Any]], worker_count: int) -> list[tuple[Path, str, str, str, dict[str, list[float]]]]:
    results: list[tuple[Path, str, str, str, dict[str, list[float]]]] = []
    if worker_count > 1:
        with multiprocessing.get_context("spawn").Pool(worker_count) as pool:
            for index, result in enumerate(pool.imap_unordered(_run_task, tasks), start=1):
                results.append(result)
                _print_progress(index, len(tasks))
    else:
        for index, task in enumerate(tasks, start=1):
            results.append(_run_task(task))
            _print_progress(index, len(tasks))
    return results


def _run_task(task: dict[str, Any]) -> tuple[Path, str, str, str, dict[str, list[float]]]:
    camera_pose = convert_cube_to_pano(**task)
    sequence_name = f"{task['difficulty']}/{task['trajectory']}"
    return task["output_file"], task["env_name"], sequence_name, task["data_type"], camera_pose


def convert_cube_to_pano(
    *,
    output_file: Path,
    raw_root: Path,
    env_name: str,
    modality: str,
    data_type: str,
    resolution: tuple[int, int],
    difficulty: str,
    camera: str,
    trajectory: str,
    frame_id: str,
    face_files: dict[str, str],
    cuda: int,
    png_compression: int,
) -> dict[str, list[float]]:
    faces, pose = _load_cube_faces(
        raw_root=raw_root,
        env_name=env_name,
        modality=modality,
        trajectory=trajectory,
        frame_id=frame_id,
        face_files=face_files,
        difficulty=difficulty,
        camera=camera,
    )
    pano = _cube_to_pano(faces, resolution=resolution, quaternion=pose[3:], modality=modality, cuda=cuda)
    if modality == "depth":
        pano = _encode_tartanair_depth(pano)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_params = [cv2.IMWRITE_PNG_COMPRESSION, max(0, min(9, int(png_compression)))]
    if not cv2.imwrite(str(output_file), pano, write_params):
        raise RuntimeError(f"failed to write panorama: {output_file}")
    return {
        "position": [float(value) for value in pose[:3]],
    }


def _load_cube_faces(
    *,
    raw_root: Path,
    env_name: str,
    modality: str,
    trajectory: str,
    frame_id: str,
    face_files: dict[str, str],
    difficulty: str,
    camera: str,
) -> tuple[list[np.ndarray], np.ndarray]:
    camera_prefix = _camera_prefix(camera)
    trajectory_dir = raw_root / env_name / f"Data_{difficulty}" / trajectory
    direction_order = ("right", "left", "top", "bottom", "front", "back")
    faces = []
    for direction in direction_order:
        file_path = Path(face_files[direction])
        image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED if modality == "depth" else cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"missing cube face: {file_path}")
        if modality == "depth":
            image = _decode_tartanair_depth(image)
        faces.append(image)

    pose_path = trajectory_dir / f"pose_{camera_prefix}_front.txt"
    frame_idx = _frame_index(frame_id)
    if frame_idx is None:
        raise RuntimeError(f"could not parse frame index from {frame_id!r}")
    lines = pose_path.read_text(encoding="utf-8").splitlines()
    if frame_idx >= len(lines):
        raise RuntimeError(f"frame {frame_idx} is out of range in {pose_path}")
    pose = np.array([float(value) for value in lines[frame_idx].split()])
    if len(pose) != 7:
        raise RuntimeError(f"expected 7 pose values in {pose_path} line {frame_idx + 1}")
    return faces, pose


def _camera_prefix(camera: str) -> str:
    if camera == "left":
        return "lcam"
    if camera == "right":
        return "rcam"
    raise ValueError(f"unsupported cube camera: {camera}")


def _discover_cube_face_files(
    *,
    traj_dir: Path,
    modality: str,
    camera_prefix: str,
    direction_order: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    files_by_direction: dict[str, dict[str, Path]] = {}
    for direction in direction_order:
        folder = traj_dir / f"{modality}_{camera_prefix}_{direction}"
        if not folder.is_dir():
            print(f"Warning: missing cube-face folder: {folder}", flush=True)
            return {}
        frames = {
            frame_id: file_path
            for file_path in sorted(folder.iterdir())
            if file_path.is_file()
            for frame_id in [_frame_id(file_path)]
            if frame_id is not None
        }
        if not frames:
            print(f"Warning: no frame files found in cube-face folder: {folder}", flush=True)
            return {}
        files_by_direction[direction] = frames

    common_frames = set.intersection(*(set(frames) for frames in files_by_direction.values()))
    if not common_frames:
        print(
            "Warning: cube-face folders have no common frame ids for "
            f"{traj_dir} modality={modality} camera={camera_prefix}",
            flush=True,
        )
    else:
        for direction, frames in files_by_direction.items():
            missing_count = len(set(frames) - common_frames)
            if missing_count:
                print(
                    "Warning: skipped incomplete cube-face frames for "
                    f"{traj_dir} modality={modality} camera={camera_prefix} "
                    f"face={direction} skipped={missing_count}",
                    flush=True,
                )
    return {
        frame_id: {direction: str(files_by_direction[direction][frame_id]) for direction in direction_order}
        for frame_id in common_frames
    }


def _frame_id(path: Path) -> str | None:
    value = path.stem.split("_", 1)[0]
    return value if value.isdigit() else None


def _frame_index(frame_id: str) -> int | None:
    return int(frame_id) if frame_id.isdigit() else None


def _data_type(modality: str) -> str:
    return "video" if modality == "mp4" else modality


def _decode_tartanair_depth(depth_rgba: np.ndarray) -> np.ndarray:
    if depth_rgba.ndim == 3 and depth_rgba.shape[-1] == 4 and depth_rgba.dtype == np.uint8:
        return np.squeeze(np.ascontiguousarray(depth_rgba).view("<f4"), axis=-1)
    return depth_rgba


def _encode_tartanair_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.ascontiguousarray(depth.astype("<f4", copy=False))
    return depth.view(np.uint8).reshape(depth.shape + (4,))


def _world_vecs(resolution: tuple[int, int], device: torch.device) -> torch.Tensor:
    width, height = resolution
    device_index = torch.cuda.current_device() if device.type == "cuda" else None
    key = (device.type, device_index, int(width), int(height))
    cached = _WORLD_VEC_CACHE.get(key)
    if cached is not None:
        return cached

    with torch.no_grad():
        u = torch.linspace(0, 1, width, device=device, dtype=torch.float32)
        v = torch.linspace(0, 1, height, device=device, dtype=torch.float32)
        u_grid, v_grid = torch.meshgrid(u, v, indexing="xy")
        psi = (u_grid - 0.5) * 2 * np.pi
        alpha = (v_grid - 0.5) * np.pi
        vectors = torch.stack(
            [
                torch.cos(alpha) * torch.cos(psi),
                torch.cos(alpha) * torch.sin(psi),
                torch.sin(alpha),
            ],
            dim=-1,
        ).reshape(-1, 3)
    _WORLD_VEC_CACHE[key] = vectors
    return vectors


def _cube_to_pano(
    faces: list[np.ndarray],
    *,
    resolution: tuple[int, int],
    quaternion: np.ndarray,
    modality: str,
    cuda: int,
) -> np.ndarray:
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    width, height = resolution
    face_tensors = [torch.from_numpy(face).to(device).float() for face in faces]
    world_vecs = _world_vecs((width, height), device)
    if np.linalg.norm(quaternion) > 1e-6:
        rotation = R.from_quat(quaternion)
        inverse = torch.tensor(rotation.inv().as_matrix(), dtype=torch.float32, device=device)
        local = world_vecs @ inverse.T
    else:
        local = world_vecs

    lx, ly, lz = local[:, 0], local[:, 1], local[:, 2]
    rules = [
        ((ly > torch.abs(lx)) & (ly > torch.abs(lz)), 0, lambda x, y, z: -x / y, lambda x, y, z: z / y),
        ((ly < -torch.abs(lx)) & (ly < -torch.abs(lz)), 1, lambda x, y, z: x / -y, lambda x, y, z: z / -y),
        ((lz < -torch.abs(lx)) & (lz < -torch.abs(ly)), 2, lambda x, y, z: y / -z, lambda x, y, z: x / -z),
        ((lz > torch.abs(lx)) & (lz > torch.abs(ly)), 3, lambda x, y, z: y / z, lambda x, y, z: -x / z),
        ((lx > torch.abs(ly)) & (lx > torch.abs(lz)), 4, lambda x, y, z: y / x, lambda x, y, z: z / x),
        ((lx < -torch.abs(ly)) & (lx < -torch.abs(lz)), 5, lambda x, y, z: -y / -x, lambda x, y, z: z / -x),
    ]

    channels = 1 if face_tensors[0].ndim == 2 else face_tensors[0].shape[-1]
    pano = torch.zeros((height * width), dtype=torch.float32, device=device) if channels == 1 else torch.zeros(
        (height * width, channels),
        dtype=torch.float32,
        device=device,
    )
    for mask, face_index, get_u, get_v in rules:
        if not mask.any():
            continue
        face = face_tensors[face_index]
        face_height, face_width = face.shape[:2]
        u_raw = get_u(lx[mask], ly[mask], lz[mask])
        v_raw = get_v(lx[mask], ly[mask], lz[mask])
        px = (u_raw + 1) / 2 * (face_width - 1)
        py = (v_raw + 1) / 2 * (face_height - 1)
        x0 = torch.floor(px).long().clamp(0, face_width - 1)
        y0 = torch.floor(py).long().clamp(0, face_height - 1)
        x1 = (x0 + 1).clamp(0, face_width - 1)
        y1 = (y0 + 1).clamp(0, face_height - 1)
        wx = px - torch.floor(px)
        wy = py - torch.floor(py)
        wa = (1.0 - wx) * (1.0 - wy)
        wb = wx * (1.0 - wy)
        wc = (1.0 - wx) * wy
        wd = wx * wy
        if channels > 1:
            wa, wb, wc, wd = wa.unsqueeze(-1), wb.unsqueeze(-1), wc.unsqueeze(-1), wd.unsqueeze(-1)
        pano[mask] = face[y0, x0] * wa + face[y0, x1] * wb + face[y1, x0] * wc + face[y1, x1] * wd

    pano = pano.reshape(height, width) if channels == 1 else pano.reshape(height, width, channels)
    original_dtype = faces[0].dtype
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        pano = torch.clamp(pano, min=0, max=info.max)
    return pano.cpu().numpy().astype(original_dtype)
