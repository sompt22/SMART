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
# Requires: torch, nvcc, g++
RUN cd src/lib/model/networks/DCNv2 \
    && python setup.py build_ext --inplace 2>&1 | tee /tmp/dcnv2_build.log \
    && echo "=== DCNv2 build OK ==="

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
 && python -c "import sys; sys.path.insert(0,'lib'); \
               from model.networks.DCNv2 import dcn_v2; print('DCNv2: OK')" \
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
