# SceneGenDeployBench-TartanAir

TartanAir dataset downloader and panorama converter for SceneGenDeployBench.

This repository wraps the TartanAir download and cube-to-panorama conversion pipeline as a `dataset_downloader` runner for [SceneGenDeployBench](https://github.com/CashewHero/SceneGenDeployBench).

## Runner

- Runner: `tartanair`
- Kind: `dataset_downloader`
- Image: `scenegendeploybench-tartanair`
- Dataset output: `/data/datasets/<dataset>`
- Job output: `/data/output/tartanair@0.1.0/<dataset>/<timestamp>_<id>`

Each job writes `runner.log` and `metrics.json` to the normal job output folder. The orchestrator supplies the dataset name in `job.parameters.dataset_name`; the runner publishes the validated dataset under `PATH_DATASETS/<dataset>` with a DeployBench-compatible `manifest.yaml`. Per-job staging stays under `/tmp`.

Manifests include `metadata.dataset_owner: SceneGenDeployBench-TartanAir`. The runner only appends to empty dataset folders or sets with this owner.

## Build

```bash
docker build -f runner_wrapper/Dockerfile -t scenegendeploybench-tartanair:local .
```

## DeployBench Config

Copy the runner catalog `runner_wrapper/config/runners/tartanair.yaml` into the main DeployBench config:

Then create a download job:

```bash
deploybench dataset download tartanair-small \
  --runner tartanair@0.1.0 \
  --set mode=equirectangular \
  --set env=AbandonedFactory2 \
  --set modality=image \
  --set camera=left
```

Useful parameters:

- `mode`: `equirectangular`, `pano_conversion`, or `raw`; defaults to `equirectangular`
- `env`: TartanAir environment name; repeat for multiple environments
- `difficulty`: `easy`, `hard`, or repeat for both
- `modality`: defaults to `image,depth`
- `camera`: `left`, `right`, `both`, or an explicit camera-name list
- `trajectory`: defaults to `all`
- `download_workers`: download worker count; defaults to `2` for stable large-file downloads
- `pano_width`, `pano_height`: panorama resolution; defaults to `2560x1280`
- `pano_convert_workers`: panorama conversion worker count
- `pano_png_compression`: PNG compression, `0` to `9`
- `pano_cuda`: `true` or `false`

`pano_conversion` converts requested modalities that TartanAir downloads as six image-readable cube-face folders.

## Source

The wrapper calls one runner-facing entry script:

```text
scripts/download_dataset.py
```

The implementation lives in `tartanair_downloader/`:

- `config.py`: normalize params and derive camera names
- `tartanair_api.py`: call `tartanairpy`
- `pipeline.py`: choose raw, official equirectangular, or pano_conversion flow
- `pano_conversion.py`: cube-face to panorama conversion
- `manifest.py`: produce manifests
