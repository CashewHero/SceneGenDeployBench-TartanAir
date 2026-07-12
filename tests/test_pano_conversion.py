from __future__ import annotations

from typing import Any

from tartanair_downloader import pano_conversion


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
