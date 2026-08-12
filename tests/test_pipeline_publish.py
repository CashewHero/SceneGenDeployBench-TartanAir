from __future__ import annotations

from pathlib import Path
from unittest import mock

import yaml

from tartanair_downloader import pipeline
from tartanair_downloader.manifest import DATASET_OWNER
from tartanair_downloader.manifest_common import write_yaml


def _write_stream(
    dataset_dir: Path,
    sample_id: str,
    contents: bytes,
    *,
    env_name: str = "Env",
) -> Path:
    sequence_dir = dataset_dir / env_name / "Data_easy" / "P000"
    data_path = sequence_dir / "image_lcam_front" / f"{sample_id}.png"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(contents)
    write_yaml(
        sequence_dir / "lcam_front.yaml",
        {
            "metadata": {
                "source": "tartanair_raw",
                "camera_side": "left",
                "projection": "pinhole",
            },
            "tags": ["stream"],
            "samples": {
                sample_id: {
                    "image": f"image_lcam_front/{sample_id}.png",
                }
            },
        },
    )
    return data_path


def _write_parent_manifests(dataset_dir: Path) -> None:
    env_subsets: dict[str, str] = {}
    for env_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        sequence_dir = env_dir / "Data_easy" / "P000"
        stream_subsets = {
            path.stem: path.name
            for path in sorted(sequence_dir.glob("*.yaml"))
            if path.name != "manifest.yaml"
        }
        write_yaml(
            sequence_dir / "manifest.yaml",
            {
                "metadata": {"difficulty": "easy", "trajectory": "P000"},
                "tags": ["sequence", "easy"],
                "subsets": stream_subsets,
            },
        )
        write_yaml(
            env_dir / "manifest.yaml",
            {
                "metadata": {"scene": env_dir.name},
                "tags": ["scene"],
                "subsets": {"easy/P000": "Data_easy/P000/manifest.yaml"},
            },
        )
        env_subsets[env_dir.name] = f"{env_dir.name}/manifest.yaml"

    write_yaml(
        dataset_dir / "manifest.yaml",
        {
            "dataset_name": "tartanair",
            "dataset_version": "0.1.0",
            "data_types": ["image"],
            "metadata": {
                "dataset_owner": DATASET_OWNER,
                "pose_coordinate_system": "NED",
                "pose_convention": "camera_to_world",
                "pose_units": "meters",
            },
            "tags": ["raw", "multi-camera"],
            "subsets": env_subsets,
        },
    )


def test_existing_dataset_merge_only_publishes_staged_files(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets" / "tartanair"
    staging_dir = tmp_path / "staging"
    old_data = _write_stream(dataset_dir, "frame000000", b"old")
    new_data = _write_stream(staging_dir, "frame000001", b"new")
    other_env_data = _write_stream(
        staging_dir,
        "frame000000",
        b"other environment",
        env_name="NewEnv",
    )
    _write_parent_manifests(dataset_dir)
    _write_parent_manifests(staging_dir)
    old_inode = old_data.stat().st_ino

    with mock.patch.object(
        pipeline,
        "publish_directory",
        wraps=pipeline.publish_directory,
    ) as publish:
        pipeline._publish_dataset(staging_dir, dataset_dir)

    publish.assert_called_once_with(staging_dir, dataset_dir, dirs_exist_ok=True)
    assert old_data.read_bytes() == b"old"
    assert old_data.stat().st_ino == old_inode
    assert (dataset_dir / new_data.relative_to(staging_dir)).read_bytes() == b"new"
    assert (dataset_dir / other_env_data.relative_to(staging_dir)).read_bytes() == b"other environment"

    stream_manifest = yaml.safe_load(
        (dataset_dir / "Env" / "Data_easy" / "P000" / "lcam_front.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert list(stream_manifest["samples"]) == ["frame000000", "frame000001"]

    dataset_manifest = yaml.safe_load((dataset_dir / "manifest.yaml").read_text(encoding="utf-8"))
    assert dataset_manifest["data_types"] == ["image"]
    assert dataset_manifest["subsets"] == {
        "Env": "Env/manifest.yaml",
        "NewEnv": "NewEnv/manifest.yaml",
    }
