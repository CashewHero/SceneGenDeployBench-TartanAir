# TartanAir Dataset Downloader Wrapper

This folder is a minimal DeployBench runner wrapper for one role only:

```text
kind: dataset_downloader
```

It exposes the DeployBench runner HTTP endpoints and calls `runner_wrapper.adapter:run_job` for each dataset download job.

## Contract

The orchestrator provides:

- `runtime.output_dir`: write `runner.log` and `metrics.json`
- `runtime.dataset_dir`: final dataset tree and `manifest.yaml` only
- `runtime.temp_dir`: scratch space for downloads, unzip, and conversion staging
- `job.parameters`: catalog defaults merged with `deploybench dataset download --set key=value`

The catalog provides default job parameters under:

```yaml
job_parameters:
```

Command-line `--set` values override those parameters for one job. Supported parameters are flat: `mode`, `env`, `difficulty`, `modality`, `camera`, `trajectory`, `download_workers`, and pano_conversion `pano_*` keys.

Modes:

- `raw`: download official TartanAir files as-is after unzip.
- `equirectangular`: download official TartanAir equirectangular panoramas.
- `pano_conversion`: download cube faces and convert any six-face modality with this repo's converter.

All modes use the same manifest shape: dataset -> scene -> sequence -> stream. Sequence folders reference stream manifests such as `P000/rcam_back.yaml`; samples inside the stream use semantic data keys such as `image`, `depth`, or `seg`.

The actual TartanAir download/conversion implementation lives outside this wrapper. The adapter calls:

```text
scripts/download_dataset.py
```
