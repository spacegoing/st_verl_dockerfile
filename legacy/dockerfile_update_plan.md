# Dockerfile Update Plan: 25.11.01 → 26.02

Please review each item and mark with `[x]` to confirm, or add comments.

---

## Action 1: Change base image
```
OLD: FROM nvcr.io/nvidia/nemo:25.11.01
NEW: FROM nvcr.io/nvidia/nemo:26.02
```
- [x] Confirmed

---

## Action 2: Install wandb BEFORE `wandb login`

wandb is NOT pre-installed in 26.02. Need to add `pip install wandb` before the login step.

```dockerfile
# NEW: install wandb first (not in 26.02 base)
RUN pip install --no-cache-dir wandb
RUN wandb login df3cecbfc0874c8a352c40820becf4a15575614e
```
- [x] Confirmed
- [ ] Or: skip wandb login in Dockerfile entirely (login at runtime instead)?

---

## Action 3: vLLM install — change removal strategy

In 25.11.01, vLLM source was at `/opt/vllm/`, so `rm -rf /opt/vllm` worked.
In 26.02, vLLM 0.14.2 is installed as a pip package (no `/opt/vllm/` dir).

```dockerfile
# OLD:
RUN rm -rf /opt/vllm && pip install setuptools_scm

# NEW:
RUN pip uninstall -y vllm && pip install setuptools_scm
```

Your custom vLLM (0.12.0) will be built against **PyTorch 2.10** (was 2.9). This means CUDA extensions get recompiled during `pip install -e .` which should handle it.

- [x] Confirmed: vLLM 0.12.0 source code is compatible with PyTorch 2.10

u need to compile vllm exactly as we did, otherwise verl won't work

---

## Action 4: `model_hosting_container_standards` and `anthropic`

In 26.02, both `anthropic` (0.71.0) and `model-hosting-container-standards` (0.1.13) are **already pre-installed** as vLLM dependencies.

```dockerfile
# OLD:
RUN pip install --no-cache-dir --upgrade-strategy "only-if-needed" model_hosting_container_standards anthropic

# Decision needed:
```
- [x] Keep this line (your custom vLLM 0.12.0 will uninstall the base vLLM, so these deps may get removed too if installed as vllm deps)
- [ ] Remove this line (after your vLLM install, these should still be present since `--no-deps` is used)

---

## Action 5: dist-packages symlink path

```dockerfile
# OLD (line 65):
ln -s /usr/local/lib/python3.12/dist-packages /root/myCodeLab/dist-packages
```

In 26.02, most packages moved to `/usr/local/lib/python3.12/dist-packages/` (same path). This symlink is still correct.

- [x] Confirmed: keep as-is

---

## Action 6: triton package rename

`triton` package is renamed to `pytorch-triton` in 26.02. Your custom vLLM build may have `triton` as a dependency.

- [x] No action needed (vLLM build uses `--no-deps`, triton functionality is provided by `pytorch-triton`)
- [ ] Need to `pip install triton` explicitly
- [ ] Need to check if vLLM 0.12.0 imports `triton` by name (it should work since `pytorch-triton` exposes the same `triton` Python module)

---

## Action 7: Packages your Dockerfile explicitly installs

These are installed via `pip install` in the Dockerfile. Confirm they're still needed:

```
gpustat codetiming tensordict mathruler pylatexenc torchdata
```
- [x] Keep all
- [ ] Remove: _______________
- [ ] Add new packages: _______________

---

## Action 8: mbridge install

```dockerfile
COPY mbridge /root/myCodeLab/host/mbridge/
RUN cd /root/myCodeLab/host/mbridge/ && pip3 install --no-cache-dir --no-deps . && cd .. && rm -rf mbridge
```

26.02 does NOT have `megatron-bridge` in pip (though source is at `/opt/Megatron-Bridge/`). Your custom mbridge installs with `--no-deps`.

- [x] Confirmed: keep as-is (your mbridge is a separate package from NVIDIA's megatron-bridge)
- [ ] Or: does your mbridge depend on megatron-bridge? If so, may need to install it first

---

## Action 9: verl install

```dockerfile
COPY verl /root/myCodeLab/host/verl/
RUN cd /root/myCodeLab/host/verl && pip3 install --no-deps -e .
```

Uses `--no-deps` so should be fine. But note megatron-core is not in pip metadata in 26.02 (importable though).

- [x] Confirmed: keep as-is

---

## Action 10: MJ_NEMO_GYM install

```dockerfile
COPY MJ_NEMO_GYM /root/myCodeLab/host/mjnemogym/
RUN cd /root/myCodeLab/host/mjnemogym && pip3 install --no-cache-dir --upgrade-strategy "only-if-needed" -e .
```

This uses `--upgrade-strategy "only-if-needed"` (NOT `--no-deps`), so pip will try to resolve dependencies. With protobuf jumping from 4.x to 6.x and other changes, this could cause conflicts.

- [ ] Confirmed: keep as-is
- [x] Change to `--no-deps` to avoid dependency resolution surprises
- [ ] Need to check mjnemogym's dependencies first

---

## Action 11: Packages removed from 26.02 that your code may need

These were in 25.11.01 but are GONE in 26.02. Check if your code (verl, mbridge, mjnemogym) imports any:

| Package | Was in A | In B | Action needed? |
|---------|----------|------|---------------|
| `xformers` | 0.0.32+nv25.9 | REMOVED | [ ] Not needed / [x] Need to install |
| `hydra-core` | installed | REMOVED | [ ] Not needed / [x] Need to install |
| `wandb` | 0.23.1 | REMOVED | [x] Will install (Action 2) |
| `grouped_gemm` | installed | REMOVED | [ ] Not needed / [x] Need to install |
| `bitsandbytes` | 0.48.2 | REMOVED | [ ] Not needed / [x] Need to install |
| `mamba_ssm` | installed | REMOVED | [ ] Not needed / [x] Need to install |
| `causal_conv1d` | installed | REMOVED | [ ] Not needed / [x] Need to install |

---

## Action 12: Notable version changes to be aware of

These changed between your myverl and 26.02. No Dockerfile action needed, but may affect runtime:

| Package | myverl | 26.02 | Risk |
|---------|--------|-------|------|
| protobuf | 4.25.8 | **6.33.5** | HIGH: major version, may break serialization |
| transformer_engine | 2.9.0 | **2.12.0** | MEDIUM: may affect training behavior |
| ray | 2.52.1 | **2.54.0** | LOW |
| peft | 0.13.2 | **0.18.1** | MEDIUM: API changes possible |
| transformers | 4.57.3 | **4.56.0** | LOW: minor downgrade |

- [x] Acknowledged

for transformers, update it to 4.57.3

---

## Summary: Minimal required changes

1. Change `FROM` to `nvcr.io/nvidia/nemo:26.02`
2. Add `pip install wandb` before `wandb login`
3. Change `rm -rf /opt/vllm` to `pip uninstall -y vllm`
4. Everything else can stay the same (pending your confirmations above)


---

## Special Attention ##

### SA-1: Proxy for docker build
`dvon` already executed in current session. Docker build uses `--network host` so
the proxy configured via `dvon` applies. If build stalls on network, check proxy.

### SA-2: Missing local folders

**ALL 5 COPY sources are MISSING** from `./`:

| Folder | What it is | Git commit in myverl image | Repo |
|--------|-----------|---------------------------|------|
| `vllm/` | Custom vLLM 0.12.0 | `4fd9d6a` (tag `v0.12.0`) | `https://github.com/vllm-project/vllm.git` |
| `verl/` | verl 0.7.0.dev0 | `e69998c7` (branch `main`) | `git@github.com:spacegoing/verl.git` |
| `MJ_NEMO_GYM/` | mjnemogym 0.1.0 | `05ddabf3` (branch `master`) | `git@github.com:HlllMan/NEMO_GYM.git` |
| `mbridge/` | mbridge 0.15.1 | source deleted after install, pip only | `https://github.com/ISEEKYAN/mbridge` |
| `nltk_data/` | NLTK tokenizers data | just `tokenizers/` subdir | — |

**Files that ARE present**: `o200k_base.tiktoken`, `to_append.sh`, `.tmux.conf`

**YOU MUST clone/copy these repos before building.**

### SA-3: Git commits from myverl image (recovered)

See SA-2 table above.

Note: `verl` in myverl was actually installed as editable pointing to
`/public/lichang93/stCodeLab/verl` (host mount), but the COPY in Dockerfile
puts it at `/root/myCodeLab/host/verl/` inside the container.

### SA-4: "DO NOT TOUCH" Protected Package List

These packages come from `nvcr.io/nvidia/nemo:26.02` and **MUST NOT be modified
by any pip install step**. Every `pip install` in the Dockerfile must use
`--no-deps` or `--upgrade-strategy "only-if-needed"` + `--no-cache-dir` to
prevent accidentally upgrading/downgrading these.

| Package | 26.02 Version | Why protected |
|---------|--------------|---------------|
| `torch` | 2.10.0a0+b558c986e8.nv25.11 | NVIDIA custom build, CUDA kernels |
| `torchvision` | 0.25.0a0+7a13ad0f | Must match torch |
| `pytorch-triton` | 3.5.0+gitde3506d2 | Must match torch |
| `flash_attn` | 2.7.4.post1+25.11 | NVIDIA custom build |
| `flashinfer-python` | 0.5.3 | Compiled for this CUDA/torch |
| `transformer_engine` | 2.12.0+5671fd36 | NVIDIA custom, megatron-core depends on it |
| `deep_ep` | 1.2.1+eb9cee7 | Compiled for this CUDA |
| `nvidia-modelopt` | 0.37.0 | NVIDIA custom |
| `nvidia-resiliency-ext` | 0.4.1+cuda13 | NVIDIA custom |
| `nvfuser` | 0.2.34+gitfce1aec | Must match torch |
| `torchao` | 0.14.0+git | Must match torch |
| `torch_tensorrt` | 2.10.0a0 | Must match torch |
| `tensorrt` | 10.14.1.48 | System-level |
| `numpy` | 1.26.4 | Many things depend on exact ABI |
| `cuda-bindings` | 13.1.1 | System-level |
| `cuda-python` | 13.1.1 | System-level |
| `cupy-cuda12x` | 14.0.1 | Compiled for this CUDA |
| `apex` | 0.1 | NVIDIA custom build |
| `compressed-tensors` | 0.13.0 | vLLM dependency (but we uninstall vLLM) |

**Build monitoring strategy**: After each `pip install` layer, we should verify
none of the above changed. In the Dockerfile, every `pip install` that installs
deps (not `--no-deps`) MUST be followed by a sanity check, OR we pin protected
packages using `--no-deps` / constraints.
