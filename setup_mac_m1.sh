#!/usr/bin/env bash
# =============================================================================
# SMART — Apple Silicon (M1/M2/M3) Setup Script
#
# Platform : macOS 13+ with Apple Silicon (arm64)
# Backend  : PyTorch MPS (Metal Performance Shaders)
#            DCNv2 → MPS/CPU shim via torchvision.ops.deform_conv2d
#
# Usage:
#   chmod +x setup_mac_m1.sh
#   ./setup_mac_m1.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="smart_m1"

echo "============================================================"
echo " SMART — Apple Silicon Setup"
echo " Repo root : $REPO_ROOT"
echo " Conda env : $ENV_NAME"
echo "============================================================"

# ── 0. Sanity checks ─────────────────────────────────────────────────────────
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "[ERROR] This script targets arm64 (Apple Silicon). Detected: $(uname -m)"
  exit 1
fi

if ! command -v conda &>/dev/null; then
  echo "[ERROR] conda not found. Install Miniconda for Apple Silicon:"
  echo "  https://docs.anaconda.com/free/miniconda/"
  exit 1
fi

# ── 1. Conda environment ──────────────────────────────────────────────────────
echo ""
echo "[1/6] Creating conda environment '$ENV_NAME'..."
conda env create -f "$REPO_ROOT/environment_mac_m1.yml" --force
echo "      Done."

# Activate in this script (source conda.sh first)
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── 2. Verify PyTorch + MPS ───────────────────────────────────────────────────
echo ""
echo "[2/6] Verifying PyTorch + MPS..."
python - <<'EOF'
import torch
print(f"  PyTorch    : {torch.__version__}")
print(f"  MPS avail  : {torch.backends.mps.is_available()}")
print(f"  MPS built  : {torch.backends.mps.is_built()}")
if not torch.backends.mps.is_available():
    print("  [WARN] MPS unavailable — will fall back to CPU.")
    print("         Ensure macOS >= 12.3 and PyTorch >= 1.12")
EOF

# ── 3. Build External NMS (Cython, no CUDA needed) ───────────────────────────
echo ""
echo "[3/6] Building External NMS (Cython)..."
cd "$REPO_ROOT/src/lib/external"
python setup.py build_ext --inplace
echo "      Done."

# ── 4. Build DCNv2 (CPU-only, no CUDA) ───────────────────────────────────────
echo ""
echo "[4/6] Building DCNv2 (CPU-only — CUDA skipped on Apple Silicon)..."
cd "$REPO_ROOT/src/lib/model/networks/DCNv2"
rm -rf build/ *.egg-info *.so

# FORCE_CUDA=0 ensures CUDA sources are skipped even if somehow available
FORCE_CUDA=0 python setup.py build develop --user

echo "      Done."
echo ""
echo "      NOTE: DCNv2 is compiled for CPU. However, the MPS shim"
echo "      (dcn_v2_mps.py) will be used automatically when MPS is"
echo "      the active device, providing full MPS acceleration."

# ── 5. Verify DCNv2 shim ─────────────────────────────────────────────────────
echo ""
echo "[5/6] Verifying DCN import fallback chain..."
cd "$REPO_ROOT/src"
python - <<'EOF'
import sys
sys.path.insert(0, 'lib')
try:
    from model.networks.DCNv2.dcn_v2 import DCN
    print("  [OK] Compiled DCNv2 loaded (CPU-only build)")
except ImportError:
    from model.networks.DCNv2.dcn_v2_mps import DCN
    print("  [OK] MPS/CPU torchvision shim loaded")

import torch
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"  [OK] Target device: {device}")
# Quick smoke test
net = DCN(32, 64, kernel_size=(3,3), stride=1, padding=1)
net = net.to(device)
x = torch.randn(1, 32, 16, 16).to(device)
y = net(x)
print(f"  [OK] DCN forward pass on {device}: {x.shape} → {y.shape}")
EOF

# ── 6. Restore working dir ────────────────────────────────────────────────────
cd "$REPO_ROOT"
echo ""
echo "[6/6] Setup complete!"
echo ""
echo "============================================================"
echo " Usage"
echo "============================================================"
echo ""
echo " conda activate $ENV_NAME"
echo " cd $REPO_ROOT/src"
echo ""
echo " # Training (uses MPS automatically on M1)"
echo " python main.py tracking,embedding \\"
echo "   --exp_id my_exp \\"
echo "   --dataset mot17 \\"
echo "   --load_model ../models/ctdet_coco_dla_2x.pth \\"
echo "   --gpus -1 \\"
echo "   --batch_size 4 \\"
echo "   --num_workers 4 \\"
echo "   --num_epochs 30 \\"
echo "   --lr 1.25e-4 \\"
echo "   --lr_step 20,25 \\"
echo "   --num_classes 1 \\"
echo "   --ltrb_amodal \\"
echo "   --pre_hm \\"
echo "   --same_aug_pre \\"
echo "   --hm_disturb 0.05 \\"
echo "   --lost_disturb 0.4 \\"
echo "   --fp_disturb 0.1"
echo ""
echo " # Inference on video"
echo " python demo.py tracking,embedding \\"
echo "   --load_model ../exp/tracking,embedding/my_exp/model_last.pth \\"
echo "   --demo /path/to/video.mp4 \\"
echo "   --gpus -1 \\"
echo "   --num_classes 1 \\"
echo "   --ltrb_amodal \\"
echo "   --save_video \\"
echo "   --no_pause"
echo ""
echo " NOTE: --gpus -1 triggers MPS/CPU auto-selection."
echo "       On M1 Max with MPS the model runs on GPU cores"
echo "       for all native PyTorch ops."
echo "============================================================"
