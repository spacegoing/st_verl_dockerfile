#!/bin/bash

# =============================================================================
# COMPLETE FAST PROBE (Metadata & File System Only)
# - Checks 100% of your requested list
# - Zero Imports = Zero Hangs = Instant Results
# - Captures DeepEP Git Hashes & System Libs (NVSHMEM/cuDNN)
# =============================================================================

IMAGES=(
    "nvcr.io/nvidia/nemo:25.11"
    "nvcr.io/nvidia/nemo:25.11.01"
    "nvcr.io/nvidia/nemo:25.09"
    "nvcr.io/nvidia/nemo:25.09.02"
    "nvcr.io/nvidia/nemo:25.09.01"
    "nvcr.io/nvidia/nemo:25.07.gpt_oss"
)

PROBE_SCRIPT='
import json, subprocess, re, sys, os
import importlib.metadata

info = {}

# --- 1. PRE-LOAD PIP METADATA (The Fast Way) ---
installed = {}
try:
    for dist in importlib.metadata.distributions():
        # normalize: "Flash_Attn" -> "flash-attn"
        name = dist.metadata["Name"].lower().replace("_", "-")
        ver = dist.version
        
        # DeepEP: Hunt for Git Commit Hash in direct_url.json
        if name == "deep-ep":
            try:
                direct_url = dist.read_text("direct_url.json")
                if direct_url:
                    data = json.loads(direct_url)
                    commit = data.get("vcs_info", {}).get("commit_id", "")
                    if commit: ver = f"{ver}+{commit[:7]}"
            except: pass
            
        installed[name] = ver
except: pass

def get_ver(search_names):
    if isinstance(search_names, str): search_names = [search_names]
    for n in search_names:
        key = n.lower().replace("_", "-")
        if key in installed: return installed[key]
    return "N/A"

# --- 2. SYSTEM CHECKS (CUDA, cuDNN, NVSHMEM) ---

# CUDA
cuda_ver = "N/A"
try:
    r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    m = re.search(r"release (\d+\.\d+)", r.stdout)
    if m: cuda_ver = m.group(1)
except: pass
info["cuda"] = cuda_ver

# NVSHMEM (Pip -> Env -> File)
nvshmem_ver = get_ver(["nvidia-nvshmem-cu12", "nvshmem"])
if nvshmem_ver == "N/A":
    if os.environ.get("NVSHMEM_VERSION"):
        nvshmem_ver = os.environ.get("NVSHMEM_VERSION")
    elif os.path.exists("/usr/local/lib/libnvshmem.so") or os.path.exists("/usr/lib/x86_64-linux-gnu/libnvshmem.so"):
        nvshmem_ver = "Lib Found"
info["nvshmem"] = nvshmem_ver

# cuDNN (Pip -> File)
cudnn_ver = get_ver(["nvidia-cudnn-cu12", "cudnn"])
if cudnn_ver == "N/A":
    # Heuristic: Check for header or library symlink versions
    # This is rough but fast. 
    paths = ["/usr/lib/x86_64-linux-gnu/libcudnn.so", "/usr/local/cuda/lib64/libcudnn.so"]
    for p in paths:
        if os.path.exists(p):
            # Try to resolve symlink to get version (e.g. libcudnn.so.8.9.2)
            try:
                real = os.path.realpath(p)
                m = re.search(r"so\.(\d+\.\d+\.\d+)", real)
                if m: cudnn_ver = m.group(1)
                else: cudnn_ver = "Lib Found"
            except: cudnn_ver = "Lib Found"
            break
info["cudnn"] = cudnn_ver


# --- 3. PACKAGE MAPPING ---
# Format: "Key": ["PipName1", "PipName2"]
pkg_map = {
    "torch": "torch",
    "te": ["transformer-engine", "transformer_engine"],
    "vllm": "vllm",
    "flash_attn": ["flash-attn", "flash_attn"],
    "triton": "triton",
    "deep_ep": ["deep-ep", "deep_ep"],
    "apex": "apex",
    "megatron": ["megatron-core", "megatron"],
    "mbridge": "mbridge",
    "transformers": "transformers",
    "tokenizers": "tokenizers",
    
    # Utilities
    "cbor2": "cbor2",
    "setproctitle": "setproctitle",
    "blake3": "blake3",
    "openai_harmony": ["openai-harmony", "openai_harmony"],
    "pybase64": "pybase64",
    "msgspec": "msgspec",
    "partial_json": ["partial-json-parser", "partial_json_parser"],
    "cpuinfo": ["py-cpuinfo", "cpuinfo"],
    "diskcache": "diskcache",
    "gguf": "gguf",
    "codetiming": "codetiming",
    "tensordict": "tensordict",
    "mathruler": "mathruler",
    "pylatexenc": "pylatexenc"
}

for key, lookup in pkg_map.items():
    info[key] = get_ver(lookup)

print(json.dumps(info))
'

echo "Fast-Probing ${#IMAGES[@]} images..." >&2
TMPFILE=$(mktemp)
trap "rm -f $TMPFILE" EXIT

for i in "${!IMAGES[@]}"; do
    img="${IMAGES[$i]}"
    # Runs instantly (no --gpus needed for metadata check)
    res=$(docker run --rm --entrypoint "" "$img" python3 -c "$PROBE_SCRIPT" || echo '{}')
    echo "$img|$res" >> "$TMPFILE"
done

# =============================================================================
# REPORT GENERATOR
# =============================================================================
python3 << EOF
import json

# The Display Order
# (Tuple: "Display Header", "json_key")
rows = [
    ("CUDA", "cuda"),
    ("cuDNN", "cudnn"),
    ("PyTorch", "torch"),
    ("Trf-Engine", "te"),
    ("Flash-Attn", "flash_attn"),
    ("Triton", "triton"),
    ("DeepEP", "deep_ep"),
    ("vLLM", "vllm"),
    ("NVSHMEM", "nvshmem"),
    ("Megatron", "megatron"),
    ("Apex", "apex"),
    ("MBridge", "mbridge"),
    ("Transformers", "transformers"),
    ("Tokenizers", "tokenizers"),
    ("GGUF", "gguf"),
    ("DiskCache", "diskcache"),
    ("Msgspec", "msgspec"),
    ("Tensordict", "tensordict"),
    ("CPU Info", "cpuinfo"),
    ("Codetiming", "codetiming"),
    ("Mathruler", "mathruler"),
    ("PyLatexEnc", "pylatexenc"),
    ("OpenAI Harmony", "openai_harmony"),
    ("Blake3", "blake3"),
    ("CBOR2", "cbor2"),
    ("SetProcTitle", "setproctitle"),
    ("PyBase64", "pybase64"),
    ("Partial JSON", "partial_json"),
]

data = {}
images = []
with open("$TMPFILE") as f:
    for line in f:
        if "|" in line:
            img, js = line.split("|", 1)
            images.append(img)
            try: data[img] = json.loads(js)
            except: data[img] = {}

# Format Header
short_names = []
for img in images:
    name = img.split("/")[-1].replace("nemo:", "")
    # Keep it readable: "25.11.01" -> "25.11.01"
    if len(name) > 10: name = name[:10]
    short_names.append(name)

col_w = 14
lbl_w = 16

def print_sep(): print("-" * (lbl_w + 3 + (col_w + 3) * len(images)))

print(f"\n### FULL STACK REPORT ###")
print_sep()
print("Package".ljust(lbl_w) + " | " + " | ".join(n.ljust(col_w) for n in short_names))
print_sep()

for label, key in rows:
    row = label.ljust(lbl_w) + " | "
    cols = []
    for img in images:
        val = str(data.get(img, {}).get(key, "-"))
        
        # Formatting Tricks for readability
        if "+" in val and len(val) > col_w: val = val.split("+")[0] + "+"
        if len(val) > col_w: val = val[:col_w-1] + "."
        
        cols.append(val.ljust(col_w))
    print(row + " | ".join(cols))
print_sep()
EOF
