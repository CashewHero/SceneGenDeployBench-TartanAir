from __future__ import annotations

"""Convert TartanAir depth maps into human-readable color images."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from tartanair_downloader.depth import decode_float32_le_bgra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize TartanAir depth maps with close pixels in red and far pixels in blue/purple."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Depth PNG/NPY files or directories to search recursively.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-depth", type=float, default=0.1, help="Near end of the color scale in meters.")
    parser.add_argument("--max-depth", type=float, default=50.0, help="Far end of the color scale in meters.")
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use a linear scale. The default logarithmic scale shows depth variation more clearly.",
    )
    return parser.parse_args()


def discover_inputs(inputs: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in inputs:
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(candidate for candidate in path.rglob("*.png") if "depth" in candidate.name.lower())
            files.update(candidate for candidate in path.rglob("*.npy") if "depth" in candidate.name.lower())
        else:
            raise FileNotFoundError(path)
    if not files:
        raise ValueError("no depth PNG or NPY files found")
    return sorted(files)


def load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path, allow_pickle=False)
    else:
        encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if encoded is None:
            raise ValueError(f"could not read image: {path}")
        if encoded.dtype == np.uint8 and encoded.ndim == 3 and encoded.shape[-1] == 4:
            depth = decode_float32_le_bgra(encoded)
        else:
            depth = encoded
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"depth must be a two-dimensional array: {path}")
    return np.asarray(depth, dtype=np.float32)


def colorize_depth(
    depth: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
    logarithmic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(depth) & (depth > 0)
    clipped = np.clip(depth, min_depth, max_depth)
    if logarithmic:
        normalized = (np.log(clipped) - np.log(min_depth)) / (np.log(max_depth) - np.log(min_depth))
    else:
        normalized = (clipped - min_depth) / (max_depth - min_depth)
    # OpenCV TURBO is purple/blue at zero and red at 255, so invert depth.
    normalized = np.nan_to_num(normalized, nan=1.0, posinf=1.0, neginf=0.0)
    color_index = np.clip((1.0 - normalized) * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(color_index, cv2.COLORMAP_TURBO)
    color[~valid] = (80, 80, 80)
    return color, valid


def add_legend(
    image: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
    logarithmic: bool,
) -> np.ndarray:
    height, width = image.shape[:2]
    footer_height = 54
    canvas = np.full((height + footer_height, width, 3), 255, dtype=np.uint8)
    canvas[:height] = image

    left = 10
    right = max(left + 1, width - 10)
    bar_top = height + 7
    bar_height = 12
    gradient = np.linspace(255, 0, right - left, dtype=np.uint8)[None, :]
    gradient = np.repeat(gradient, bar_height, axis=0)
    canvas[bar_top : bar_top + bar_height, left:right] = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)

    scale_name = "log" if logarithmic else "linear"
    cv2.putText(
        canvas,
        f"close {min_depth:g} m",
        (left, height + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    far_label = f"far {max_depth:g} m ({scale_name} scale)"
    text_width = cv2.getTextSize(far_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
    cv2.putText(
        canvas,
        far_label,
        (max(left, right - text_width), height + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return canvas


def output_path(output_dir: Path, source: Path, used_names: set[str]) -> Path:
    base = f"{source.stem}-vis.png"
    name = base
    index = 2
    while name in used_names:
        name = f"{source.stem}-vis-{index}.png"
        index += 1
    used_names.add(name)
    return output_dir / name


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.min_depth) or args.min_depth <= 0:
        raise ValueError("--min-depth must be a positive finite number")
    if not np.isfinite(args.max_depth) or args.max_depth <= args.min_depth:
        raise ValueError("--max-depth must be finite and greater than --min-depth")

    sources = discover_inputs(args.inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    for source in sources:
        depth = load_depth(source)
        color, valid = colorize_depth(
            depth,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            logarithmic=not args.linear,
        )
        visualization = add_legend(
            color,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            logarithmic=not args.linear,
        )
        destination = output_path(args.output_dir, source, used_names)
        if not cv2.imwrite(str(destination), visualization):
            raise RuntimeError(f"could not write image: {destination}")
        finite = depth[valid]
        if finite.size:
            summary = (
                f"valid={valid.mean():.1%}, min={finite.min():.4g} m, "
                f"median={np.median(finite):.4g} m, max={finite.max():.4g} m"
            )
        else:
            summary = "valid=0.0%"
        print(f"{source} -> {destination} ({summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
