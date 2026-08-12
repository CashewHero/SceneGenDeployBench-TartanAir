FROM python:3.12-slim

WORKDIR /app/model_repo

ENV PYTHONPATH=/app/model_repo
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ARG INSTALL_REQUIREMENTS=1
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
ARG TORCH_CUDA=cu126
COPY requirements.txt /app/model_repo/requirements.txt
RUN if [ "$INSTALL_REQUIREMENTS" = "1" ] && [ -f requirements.txt ]; then \
      pip install --no-cache-dir \
        "torch==${TORCH_VERSION}+${TORCH_CUDA}" \
        "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
      && pip install --no-cache-dir -r requirements.txt \
      && python -c "import torch, torchvision; from torchvision.ops import nms; assert torch.__version__ == '${TORCH_VERSION}+${TORCH_CUDA}'; assert torchvision.__version__ == '${TORCHVISION_VERSION}+${TORCH_CUDA}'; assert torch.version.cuda == '12.6'; nms(torch.empty((0, 4)), torch.empty(0), 0.5)" \
      && pip check; \
    fi

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libxcb1 unzip \
    && rm -rf /var/lib/apt/lists/*

ENV USER=deploybench
ENV LOGNAME=deploybench
ENV HOME=/tmp
ENV XDG_CACHE_HOME=/tmp/.cache
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV CUPY_CACHE_DIR=/tmp/cupy

COPY . /app/model_repo

EXPOSE 58090

ENV RUNNER_PORT=58090
ENV RUNNER_NAME=tartanair
ENV RUNNER_TYPE=dataset_downloader
ENV RUNNER_VERSION=0.1.3
ENV RUNNER_CONTRACT_VERSION=1
ENV RUNNER_ADAPTER=runner_wrapper.adapter:run_job

CMD ["python", "-m", "runner_wrapper.server"]
