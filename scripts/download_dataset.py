from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tartanair_downloader.pipeline import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a TartanAir dataset for DeployBench.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--temp-dir", required=True)
    parser.add_argument("--params-json", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = json.loads(args.params_json)
    if not isinstance(params, dict):
        raise ValueError("--params-json must decode to an object")
    summary = run(
        dataset_name=args.dataset_name,
        dataset_dir=Path(args.dataset_dir),
        temp_dir=Path(args.temp_dir),
        params=params,
    )
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
