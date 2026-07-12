from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


CUBE_FACES = ("front", "left", "right", "back", "top", "bottom")
SIDES = {"left": "lcam", "right": "rcam"}
RAW_CAMERA_FALLBACKS = {
    "flow": ["lcam_front"],
}
MODE_UNSUPPORTED_MODALITIES = {
    "raw": {"events", "mp4"},
    "equirectangular": {"flow", "events", "imu", "lidar", "mp4"},
    "pano_conversion": {"flow", "events", "imu", "lidar", "mp4"},
}


def _as_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _as_bool(value: Any, default: bool) -> bool:
    if value is None or str(value).strip() == "":
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _as_int(value: Any, default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _camera_values(value: Any) -> list[str]:
    values = _as_list(value, ["left"])
    if len(values) == 1 and values[0] == "both":
        return ["left", "right"]
    return values


@dataclass(frozen=True)
class DownloadConfig:
    mode: str
    env: list[str]
    difficulty: list[str]
    modality: list[str]
    camera: list[str]
    trajectory: list[str] | None
    download_workers: int
    pano_width: int
    pano_height: int
    pano_convert_workers: int
    pano_png_compression: int
    pano_cuda: bool

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "DownloadConfig":
        mode = str(params.get("mode") or "equirectangular").strip()
        if mode not in {"raw", "equirectangular", "pano_conversion"}:
            raise ValueError("mode must be 'raw', 'equirectangular', or 'pano_conversion'")

        env = _as_list(params.get("env"), [])
        if not env:
            raise ValueError("env is required")

        pano_png_compression = _as_int(params.get("pano_png_compression"), 9)
        if pano_png_compression < 0 or pano_png_compression > 9:
            raise ValueError("pano_png_compression must be between 0 and 9")

        trajectory_values = _as_list(params.get("trajectory"), ["all"])
        trajectory = None if trajectory_values == ["all"] else trajectory_values

        return cls(
            mode=mode,
            env=env,
            difficulty=_as_list(params.get("difficulty"), ["easy"]),
            modality=_as_list(params.get("modality"), ["image", "depth"]),
            camera=_camera_values(params.get("camera")),
            trajectory=trajectory,
            download_workers=max(_as_int(params.get("download_workers"), 2), 1),
            pano_width=max(_as_int(params.get("pano_width"), 2560), 1),
            pano_height=max(_as_int(params.get("pano_height"), 1280), 1),
            pano_convert_workers=max(_as_int(params.get("pano_convert_workers"), 4), 1),
            pano_png_compression=pano_png_compression,
            pano_cuda=_as_bool(params.get("pano_cuda"), True),
        )

    def without_known_unsupported_modalities(self) -> "DownloadConfig":
        unsupported = MODE_UNSUPPORTED_MODALITIES.get(self.mode)
        if unsupported is None:
            return self
        skipped = [modality for modality in self.modality if modality in unsupported]
        if not skipped:
            return self
        kept = [modality for modality in self.modality if modality not in unsupported]
        print(
            f"Warning: TartanAir {self.mode} skips unsupported modalities "
            f"{','.join(skipped)}; using {','.join(kept) if kept else 'none'}",
            flush=True,
        )
        return replace(self, modality=kept)

    def camera_names_for_download(self, modality: str | None = None) -> list[str]:
        requested = self._requested_camera_names()
        if self.mode == "raw" and modality in RAW_CAMERA_FALLBACKS:
            fallback = RAW_CAMERA_FALLBACKS[modality]
            selected = [camera_name for camera_name in requested if camera_name in fallback]
            return selected or list(fallback)
        return requested

    def _requested_camera_names(self) -> list[str]:
        explicit = [value for value in self.camera if value not in {"left", "right"}]
        if explicit:
            return explicit

        if self.mode == "equirectangular":
            return [f"{SIDES[side]}_equirect" for side in self.camera]

        names: list[str] = []
        for side in self.camera:
            prefix = SIDES[side]
            names.extend(f"{prefix}_{face}" for face in CUBE_FACES)
        return names

    def warn_if_raw_camera_adjusted(self, modality: str) -> None:
        if self.mode != "raw" or modality not in RAW_CAMERA_FALLBACKS:
            return
        requested = self._requested_camera_names()
        used = self.camera_names_for_download(modality)
        if requested == used:
            return
        fallback = RAW_CAMERA_FALLBACKS[modality]
        print(
            f"Warning: TartanAir {modality} only has camera_name={','.join(fallback)}; "
            f"requested camera_name={','.join(requested)}; using {','.join(used)}",
            flush=True,
        )

    def camera_sides_for_cube_conversion(self) -> list[str]:
        sides = [value for value in self.camera if value in {"left", "right"}]
        if len(sides) != len(self.camera):
            raise ValueError("pano_conversion requires camera to be left, right, or both")
        return sides
