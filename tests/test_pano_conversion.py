from __future__ import annotations

from pathlib import Path
from typing import Any

from tartanair_downloader import pano_conversion, pipeline
from tartanair_downloader.config import DownloadConfig
from tartanair_downloader.manifest import _depth_representation, _stream_metadata


def test_run_tasks_uses_spawn_workers(monkeypatch: Any) -> None:
    start_methods: list[str] = []

    class FakePool:
        def __enter__(self) -> "FakePool":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def imap_unordered(self, function: Any, tasks: list[dict[str, Any]]) -> Any:
            return map(function, tasks)

    class FakeContext:
        def Pool(self, worker_count: int) -> FakePool:
            assert worker_count == 2
            return FakePool()

    def fake_get_context(start_method: str) -> FakeContext:
        start_methods.append(start_method)
        return FakeContext()

    monkeypatch.setattr(pano_conversion.multiprocessing, "get_context", fake_get_context)
    monkeypatch.setattr(pano_conversion, "_run_task", lambda task: task["result"])

    expected = [("output", "env", "easy/P000", "image", {"position": [0.0, 0.0, 0.0]})]
    assert pano_conversion._run_tasks([{"result": expected[0]}], worker_count=2) == expected
    assert start_methods == ["spawn"]


def test_depth_metadata_distinguishes_raw_z_from_panorama_distance() -> None:
    raw = _stream_metadata(
        source="tartanair_raw",
        data_types={"depth"},
        depth_representation="camera_z",
        camera_side="left",
        projection="pinhole",
        resolution=[640, 640],
    )
    panorama = _stream_metadata(
        source="tartanair_pano_conversion",
        data_types={"depth"},
        depth_representation="ray_distance",
        camera_side="left",
        projection="equirectangular",
        resolution=[1024, 512],
    )

    assert raw["depth"]["representation"] == "camera_z"
    assert panorama["depth"]["representation"] == "ray_distance"
    assert _depth_representation("fisheye") == "ray_distance"


def test_raw_pipeline_preserves_downloaded_depth_bytes(tmp_path: Path, monkeypatch: Any) -> None:
    original = b"downloaded depth bytes"

    def fake_download(staging_dir: Path, config: DownloadConfig) -> None:
        path = staging_dir / "Env" / "Data_easy" / "P000" / "depth_lcam_front" / "depth.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(original)

    def fake_publish(staging_dir: Path, *args: Any) -> None:
        path = staging_dir / "Env" / "Data_easy" / "P000" / "depth_lcam_front" / "depth.png"
        assert path.read_bytes() == original

    monkeypatch.setattr(pipeline, "download", fake_download)
    monkeypatch.setattr(pipeline, "_remove_non_dataset_artifacts", lambda *args: None)
    monkeypatch.setattr(pipeline, "_prune_staged_dataset", lambda *args: None)
    monkeypatch.setattr(pipeline, "write_manifest", lambda *args: 1)
    monkeypatch.setattr(pipeline, "_publish_dataset", fake_publish)
    config = DownloadConfig.from_params(
        {
            "mode": "raw",
            "env": "Env",
            "difficulty": "easy",
            "trajectory": "P000",
            "modality": "depth",
            "camera": "lcam_front",
        }
    )

    result = pipeline._run_raw(
        dataset_name="tartanair",
        dataset_dir=tmp_path / "published",
        temp_dir=tmp_path / "temporary",
        config=config,
    )

    assert result["mode"] == "raw"
