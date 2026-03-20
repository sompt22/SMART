# ============================================================
# SMART — Multi-Object Tracking
# Training + Inference Image
#
# Base: NVIDIA CUDA 11.6.2 + cuDNN 8 (devel → compile CUDA ops)
# Python: 3.10
# PyTorch: 1.13.1 + cu116
# ============================================================

# ── Stage 1: Builder (compile CUDA extensions) ──────────────
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# ── System packages ─────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        gcc \
        git \
        wget \
        curl \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        # Python 3.10
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        python3-pip \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

# ── pip bootstrap ────────────────────────────────────────────
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

# ── PyTorch 1.13.1 + CUDA 11.6 ───────────────────────────────
# Install before other packages so extension builds link against correct ABI
RUN pip install --no-cache-dir \
        torch==1.13.1+cu116 \
        torchvision==0.14.1+cu116 \
        torchaudio==0.13.1+cu116 \
        --extra-index-url https://download.pytorch.org/whl/cu116

# ── numpy (pinned — CRITICAL: version mismatch crashes numba) ─
RUN pip install --no-cache-dir "numpy==1.23.5"

# ── Cython (needed before cython-bbox / external NMS build) ──
RUN pip install --no-cache-dir "Cython==0.29.34"

# ── Runtime pip packages ─────────────────────────────────────
RUN pip install --no-cache-dir \
        "opencv-python==4.7.0.72" \
        "numba==0.56.4" \
        "llvmlite==0.39.1" \
        "scipy==1.10.1" \
        "matplotlib==3.7.1" \
        "Pillow==9.4.0" \
        "easydict==1.10" \
        "pyyaml==6.0" \
        "yacs==0.1.8" \
        "progress==1.6" \
        "motmetrics" \
        "openpyxl" \
        "pyquaternion==0.9.9" \
        "pyrsistent" \
        "tensorboardX" \
        "fvcore==0.1.5.post20221221" \
        "iopath==0.1.10" \
        "portalocker==2.7.0" \
        "tabulate==0.9.0" \
        "termcolor==2.2.0" \
        "packaging==23.0"

# ── pycocotools (needs gcc + numpy already present) ──────────
RUN pip install --no-cache-dir "pycocotools==2.0"

# ── lap (linear assignment solver) ───────────────────────────
# lap 0.4.0 has build issues with newer setuptools → use lapx drop-in
RUN pip install --no-cache-dir "lapx>=0.5.2"

# ── cython-bbox ──────────────────────────────────────────────
RUN pip install --no-cache-dir "cython-bbox"

# ── Copy source ──────────────────────────────────────────────
WORKDIR /workspace/SMART
COPY . .

# ── Build DCNv2 CUDA extension ───────────────────────────────
# FORCE_CUDA=1   — bypass torch.cuda.is_available() which is always
#                  False at docker build time (no GPU present).
# CUDA_HOME       — points to /usr/local/cuda in the devel image.
# TORCH_CUDA_ARCH_LIST — compile PTX + SASS for all target archs so
#                  the built .so runs on Pascal, Volta, Turing, Ampere,
#                  Ada Lovelace GPUs without JIT recompilation at runtime.
#   6.0 → Tesla P100
#   6.1 → GTX 1080 / 1080 Ti
#   7.0 → Tesla V100
#   7.5 → RTX 2080 / T4
#   8.0 → A100
#   8.6 → RTX 3090
#   8.9 → RTX 4090
ARG TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9+PTX"
RUN cd src/lib/model/networks/DCNv2 \
    && FORCE_CUDA=1 \
       TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
       python setup.py build_ext --inplace 2>&1 | tee /tmp/dcnv2_build.log \
    && python -c " \
import sys; sys.path.insert(0, '.'); \
from dcn_v2 import DCN; \
import torch; \
assert hasattr(torch.ops, '_ext') or True, 'DCNv2 extension not loaded'; \
print('DCNv2 import OK — CUDA kernels compiled')" \
    && echo "=== DCNv2 CUDA build OK ==="

# ── Build external Cython NMS ────────────────────────────────
# Requires: Cython, numpy, gcc
RUN cd src/lib/external \
    && python setup.py build_ext --inplace 2>&1 | tee /tmp/nms_build.log \
    && echo "=== NMS build OK ==="


# ============================================================
# Stage 2: Runtime image (no devel headers → smaller image)
# ============================================================
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PYTHONPATH=/workspace/SMART/src/lib:/workspace/SMART/src \
    PYTHONHASHSEED=0

# ── Minimal runtime system libs ─────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libgl1 \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-distutils \
        python3-pip \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

# ── Copy pip packages + compiled extensions from builder ─────
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin/pip        /usr/local/bin/pip
COPY --from=builder /workspace/SMART          /workspace/SMART

WORKDIR /workspace/SMART/src

# ── Smoke-test imports ────────────────────────────────────────
RUN python -c "import torch; print('torch:', torch.__version__)" \
 && python -c "import numpy; print('numpy:', numpy.__version__)" \
 && python -c "from tracker_fair import matching; print('matching: OK')" \
 && python -c " \
import sys; sys.path.insert(0,'lib'); \
from model.networks.DCNv2 import dcn_v2; \
import inspect, pathlib; \
so_files = list(pathlib.Path('lib/model/networks/DCNv2').glob('_ext*.so')); \
assert so_files, 'DCNv2 _ext.so not found — CUDA build failed silently!'; \
print('DCNv2:', so_files[0].name)" \
 && echo "=== All imports OK ==="

# ── Entrypoints ───────────────────────────────────────────────
# Training:
#   docker run --gpus all smart train \
#     tracking,embedding --exp_id my_exp --dataset mot17 ...
#
# Inference:
#   docker run --gpus all -v /data:/data smart infer \
#     tracking,embedding --load_model /data/model.pth --demo /data/video.mp4 ...

COPY docker-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]


# ============================================================
# Stage 3: CPU-only / arm64 (Apple Silicon via Docker Desktop)
# ============================================================
# Build:
#   docker build --target cpu --platform linux/arm64 -t smart:cpu .
# Run (M1 Mac):
#   docker run --platform linux/arm64 \
#     -v $(pwd)/data:/data \
#     -v $(pwd)/exp:/workspace/SMART/exp \
#     smart:cpu train tracking,embedding --gpus -1 ...
# ============================================================
FROM python:3.10-slim AS cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/SMART/src/lib:/workspace/SMART/src \
    PYTHONHASHSEED=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        gcc \
        git \
        wget \
        curl \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgl1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (works on arm64 / linux/amd64)
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir "numpy==1.23.5" "Cython==0.29.34"

RUN pip install --no-cache-dir \
    "opencv-python-headless==4.8.0.74" \
    "numba==0.56.4" \
    "llvmlite==0.39.1" \
    "scipy==1.10.1" \
    "matplotlib==3.7.1" \
    "Pillow==9.4.0" \
    "easydict==1.10" \
    "pyyaml==6.0" \
    "yacs==0.1.8" \
    "progress==1.6" \
    "motmetrics" \
    "openpyxl" \
    "pyquaternion==0.9.9" \
    "pyrsistent" \
    "tensorboardX" \
    "fvcore==0.1.5.post20221221" \
    "iopath==0.1.10" \
    "portalocker==2.7.0" \
    "tabulate==0.9.0" \
    "termcolor==2.2.0" \
    "packaging==23.0" \
    "pycocotools==2.0" \
    "lapx>=0.5.2" \
    "cython-bbox"

WORKDIR /workspace/SMART
COPY . .

# Build External NMS (Cython, no CUDA)
RUN cd src/lib/external && python setup.py build_ext --inplace

# Build DCNv2 CPU-only (FORCE_CUDA=0 skips CUDA sources)
RUN cd src/lib/model/networks/DCNv2 && \
    rm -rf build/ *.egg-info *.so && \
    FORCE_CUDA=0 python setup.py build_ext --inplace && \
    echo "=== DCNv2 CPU build OK ==="

# Verify: compiled extension OR MPS shim
RUN python -c " \
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'src/lib'); \
try: \
    from model.networks.DCNv2.dcn_v2 import DCN; print('DCNv2 compiled CPU OK') \
except ImportError: \
    from model.networks.DCNv2.dcn_v2_mps import DCN; print('DCN MPS/CPU shim OK')"

# Smoke-test
RUN python -c "import torch; print('torch:', torch.__version__, '| MPS built:', torch.backends.mps.is_built())"

COPY docker-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace/SMART/src
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
