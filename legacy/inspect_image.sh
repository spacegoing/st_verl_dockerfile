#!/bin/bash
# Usage: ./inspect_image.sh <image_name> <output_dir>
# Inspects a docker image and dumps package/version info to output_dir

IMAGE="$1"
OUTDIR="$2"

if [ -z "$IMAGE" ] || [ -z "$OUTDIR" ]; then
    echo "Usage: $0 <image_name> <output_dir>"
    exit 1
fi

mkdir -p "$OUTDIR"

echo "=== Inspecting $IMAGE ==="

# 1. System info
docker run --rm "$IMAGE" bash -c '
echo "--- python ---"
python3 --version 2>&1

echo "--- nvcc ---"
nvcc --version 2>&1 || echo "nvcc not found"

echo "--- cudnn ---"
cat /usr/include/cudnn_version.h 2>/dev/null | grep -E "CUDNN_(MAJOR|MINOR|PATCHLEVEL) " || echo "cudnn header not found"

echo "--- nccl ---"
dpkg -l 2>/dev/null | grep -E "libnccl[2 ]" | awk "{print \$2, \$3}" || echo "nccl not found"

echo "--- gcc ---"
gcc --version 2>&1 | head -1

echo "--- cmake ---"
cmake --version 2>&1 | head -1

echo "--- os ---"
cat /etc/os-release | grep -E "^(NAME|VERSION)="

echo "--- pytorch ---"
python3 -c "
import torch
print(\"torch:\", torch.__version__)
print(\"cuda:\", torch.version.cuda)
print(\"cudnn:\", torch.backends.cudnn.version())
" 2>&1

echo "--- triton ---"
python3 -c "import triton; print(\"triton:\", triton.__version__)" 2>&1 || echo "triton not importable"

echo "--- tensorrt ---"
python3 -c "import tensorrt; print(\"tensorrt:\", tensorrt.__version__)" 2>&1 || echo "tensorrt not importable"
' > "$OUTDIR/system_info.txt" 2>&1

echo "  [1/6] system_info.txt done"

# 2. Full pip list
docker run --rm "$IMAGE" bash -c 'pip list --format=freeze 2>/dev/null | sort' > "$OUTDIR/pip_freeze.txt" 2>&1

echo "  [2/6] pip_freeze.txt done"

# 3. Key package details (location, version, editable)
docker run --rm "$IMAGE" bash -c '
for pkg in torch torchvision vllm megatron-core nemo-toolkit NeMo-FW megatron-bridge transformers flash-attn flashinfer-python tensorrt nvidia-modelopt wandb numpy pydantic setuptools pip accelerate deepspeed xformers triton pytorch-triton ray scipy compressed-tensors nvidia-resiliency-ext; do
    echo "=== $pkg ==="
    pip show "$pkg" 2>/dev/null | grep -E "^(Name|Version|Location|Editable)" || echo "NOT INSTALLED"
    echo ""
done
' > "$OUTDIR/key_packages.txt" 2>&1

echo "  [3/6] key_packages.txt done"

# 4. /opt directory structure
docker run --rm "$IMAGE" bash -c '
echo "--- /opt top level ---"
ls -la /opt/ 2>/dev/null

echo ""
echo "--- /opt/vllm ---"
ls -la /opt/vllm/ 2>/dev/null | head -5 || echo "NO /opt/vllm"

echo ""
echo "--- /opt/megatron-lm ---"
ls -la /opt/megatron-lm/ 2>/dev/null | head -5 || echo "NO /opt/megatron-lm"

echo ""
echo "--- /opt/NeMo ---"
ls -la /opt/NeMo/ 2>/dev/null | head -5 || echo "NO /opt/NeMo"

echo ""
echo "--- /opt/NeMo-FW ---"
ls -la /opt/NeMo-FW/ 2>/dev/null | head -5 || echo "NO /opt/NeMo-FW"

echo ""
echo "--- /opt/Megatron-Bridge ---"
ls -la /opt/Megatron-Bridge/ 2>/dev/null | head -5 || echo "NO /opt/Megatron-Bridge"

echo ""
echo "--- /opt/xformers ---"
ls -la /opt/xformers/ 2>/dev/null | head -5 || echo "NO /opt/xformers"

echo ""
echo "--- /opt/venv ---"
ls -la /opt/venv/ 2>/dev/null | head -5 || echo "NO /opt/venv"

echo ""
echo "--- /opt/DeepEP ---"
ls -la /opt/DeepEP/ 2>/dev/null | head -5 || echo "NO /opt/DeepEP"
' > "$OUTDIR/opt_structure.txt" 2>&1

echo "  [4/6] opt_structure.txt done"

# 5. Python site-packages paths and importability checks
docker run --rm "$IMAGE" bash -c '
echo "--- site-packages paths ---"
python3 -c "import site; print(site.getsitepackages())"

echo ""
echo "--- vllm import ---"
python3 -c "import vllm; print(\"version:\", vllm.__version__); print(\"file:\", vllm.__file__)" 2>&1 || echo "vllm not importable"

echo ""
echo "--- megatron.core import ---"
python3 -c "import megatron.core; print(\"version:\", getattr(megatron.core, \"__version__\", \"no attr\")); print(\"file:\", megatron.core.__file__)" 2>&1 || echo "megatron.core not importable"

echo ""
echo "--- nemo import ---"
python3 -c "import nemo; print(\"file:\", nemo.__file__); print(\"version:\", getattr(nemo, \"__version__\", \"no attr\"))" 2>&1 || echo "nemo not importable"

echo ""
echo "--- transformer_engine import ---"
python3 -c "import transformer_engine; print(\"version:\", transformer_engine.__version__); print(\"file:\", transformer_engine.__file__)" 2>&1 || echo "transformer_engine not importable"

echo ""
echo "--- dist-packages listing (first 20) ---"
ls /usr/local/lib/python3.12/dist-packages/ 2>/dev/null | head -20
' > "$OUTDIR/python_paths.txt" 2>&1

echo "  [5/6] python_paths.txt done"

# 6. apt packages (key ones)
docker run --rm "$IMAGE" bash -c '
dpkg -l 2>/dev/null | grep -E "^ii" | awk "{print \$2, \$3}" | sort
' > "$OUTDIR/apt_packages.txt" 2>&1

echo "  [6/6] apt_packages.txt done"
echo "=== Done: $IMAGE -> $OUTDIR ==="
