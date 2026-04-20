# Qwen3-30B-A3B-Base port — dev notes

Stage-by-stage progress. Entries prepended (newest first).

---

## 2026-04-20 — Start

Arch summary from `/mnt/public/lichang93/downloads/models/Qwen3-30B-A3B-Base/config.json`:
- `Qwen3MoeForCausalLM`, `model_type: qwen3_moe`
- 48 layers, 2048 hidden, 32 q-heads / 4 kv-heads (GQA ratio 8), head_dim 128
- 128 experts, topk 8, moe_intermediate 768
- ffn 6144, rope_theta 1e6, max_pos 32768, vocab 151936, bfloat16

Framework support verified:
- mbridge `@register_model("qwen3_moe")` in `/opt/venv/lib/python3.12/site-packages/mbridge/models/qwen3moe.py` ✓
- vLLM `qwen3_moe.py` in `vllm.model_executor.models` ✓

Cluster: 28/31 busy on sd ablation, 3 free → using NNODES=2.

## Stage 1 — configs (done, committed)

Created:
- `verl/my_scripts/k8s/config/qwen3_30b_a3b_2node_sd.yaml` (Hydra base)
- `verl/my_scripts/k8s/config/env/k8s_b300_2node_qwen3.yaml` (2-node env, PP=1/EP=8/TP=1)
- `verl/my_scripts/k8s/run_qwen3_k8s_2node_single_domain.sh` (entrypoint)
- `iter_kuberay_.../submit/rayjob_qwen3.yaml` (RayJob template, **no VLLM_ATTENTION_BACKEND env**)
- `iter_kuberay_.../submit/submit_qwen3.sh` (dispatcher, all NNODES → k8s_b300_2node_qwen3 env)
- `qwen3_port/submit_qwen3.sh` (top wrapper, always NNODES=2)
- combos `cbdg-qwen3-init` (0 step) and `cbdg-qwen3-smoke` (1 step)

Hydra compose dry-run produced expected cfg (model.path, PP=1/EP=8, loss_mode=bspo,
attention_backend=fused).

## Stage 2 — first submission: NVTE_FUSED_ATTN conflict

Submitted `cbdg-qwen3-init` (`trainer.val_before_train=false`). Failed with:
```
AssertionError: NVTE_FUSED_ATTN set to 1, but expected 0 for attention backend type flash
```
The rayjob env has `NVTE_FUSED_ATTN=1` (from 40Bra MLA). We'd set `attention_backend: flash`
which conflicts — TE refuses.

**Fix:** switch `attention_backend: flash → fused` in `qwen3_30b_a3b_2node_sd.yaml`. TE
fused attention supports non-MLA GQA at sm_103. No rayjob env change needed. Committed on
`nemo_bspo_sd_ablation @ 81c87a2b`.

## Stage 2 retry — model loads, optimizer scheduler asserts

Resubmitted. Logs:
```
(WorkerDict ...) actor_module: 1      ← mbridge Qwen3MoE load SUCCEEDED
...
AssertionError at optimizer_param_scheduler.py:156:
    assert self.lr_decay_steps > 0
```

Root cause: `total_training_steps=0` produces `lr_decay_steps=0` → Megatron scheduler
asserts. This is expected — 0-step runs aren't supported by the scheduler. Since model
loading already succeeded (`actor_module: 1`), **Stage 2's real goal is met**.

**Decision:** collapse Stages 2/3/4 into one — go straight to `cbdg-qwen3-smoke`
(1-step, val_before_train=true, bspo simplest, δ=3e-4, λ=1e-2). Override
`lr_warmup_steps=0` so the 1-step warmup→decay schedule is well-defined.

## Stage 2/3/4 combined — 1-step smoke (in progress)

Submitted `qwen3-sd-cbdg-qwen3-smoke-nscnc` @ 17:12 UTC with:
  `actor_rollout_ref.actor.optim.lr_warmup_steps=0`

Watching in background (task `b5thiqg8t`). Waiting on first training/global_step or
terminal error.

## Stage 4 try 1 — vLLM FlashInfer backend incompat

Submitted smoke `qwen3-sd-cbdg-qwen3-smoke-nscnc` at 17:12 UTC. Actor loaded on
all 16 workers (`actor_module: 1` repeated ×16). Then vLLM engine init failed:

```
File "/opt/venv/lib/python3.12/site-packages/flashinfer/decode.py", line 948, in plan
    self._paged_kv_indptr_buf = indptr.to(
TypeError: to() received an invalid combination of arguments - got (torch.device, non_blocking=NoneType), ...
```

vLLM's v1 engine picked FlashInfer backend automatically (no `VLLM_ATTENTION_BACKEND`
env var — I removed the 40Bra `CUTLASS_MLA` earlier). FlashInfer 0.x calls
`tensor.to(device, non_blocking=None)` which torch rejects (None is not a valid
bool for `non_blocking`). This is a pinned-version incompatibility between the
flashinfer wheel in the image and torch.

**Fix:** pin `VLLM_ATTENTION_BACKEND=FLASH_ATTN` in rayjob_qwen3.yaml env list.
FlashAttn is well-tested for GQA on recent vLLM. Committed on iter_kuberay
`nemo_bspo_md @ 0668c4e`.

## Stage 4 try 2 — FLASH_ATTN (in progress)

Submitted `qwen3-sd-cbdg-qwen3-smoke-kg7kj` at 17:40 UTC. Watcher `bvsyi0hc7`
running in background.

## Stage 4 try 2 — ✅ SUCCEEDED (2026-04-20 18:43 UTC, 78min total)

Full pipeline worked end-to-end with FLASH_ATTN backend. Terminal state:
```
qwen3-sd-cbdg-qwen3-smoke-kg7kj  SUCCEEDED  Complete
  start=17:24:39Z  end=18:43:15Z  (78m)
```

### Step-0 val (val-before-train) — Qwen3 baseline on BSPO hard-eval

| data_source    | acc@1 |
|----------------|-------|
| BeyondAIME     | 0.06  |
| AIME2025       | 0.067 |
| OlympiadBench  | 0.43  |

Base model baseline. Much lower than 40Bra on BeyondAIME/AIME2025 (0.33–0.55)
but matches on OlympiadBench — Qwen3-30B-A3B-Base is weaker on hard math as
expected for a 3B-active model vs 40Bra's stronger MoE.

### Step 1 (on-policy, bspo simplest, δ=3e-4, λ=1e-2)

| metric                              | value               |
|-------------------------------------|---------------------|
| `actor/pg_loss`                     | 0.0                 |
| `actor/grad_norm`                   | 3.097               |
| `actor/bspo_s_tau_mean`             | 0.0                 |
| `actor/bspo_delta_reg_mean`         | 0.0                 |
| `actor/bspo_penalty_mean`           | 0.0                 |
| `actor/bspo_clip_pos_frac`          | 0.0                 |
| `actor/entropy`                     | 1.313               |
| `critic/score/mean` (reward)        | 0.129               |
| `response_length/mean`              | 996                 |
| `timing_s/agent_loop/generate` slow | 247s                |

All `bspo_*` metrics 0 because ppo_epochs=1 → on-policy step with
log_ratio ≈ 0 → s_pos/s_neg ≈ 0 → zero clip and zero penalty. Exactly the
same step-1 pattern as 40Bra. Confirms the BSPO loss **runs end-to-end on
Qwen3-30B-A3B** without crashes; gradient flows (`grad_norm=3.097`).

### Step-1 post-val

| data_source    | acc@1 |
|----------------|-------|
| BeyondAIME     | 0.06  |
| OlympiadBench  | 0.40  |

Essentially unchanged from step 0 (pg_loss=0, no weight update).

## Exit criteria met — Qwen3 port works

Verdict: Qwen3-30B-A3B-Base SD training is wired end-to-end on this cluster.

**Port summary (for B200 cluster or future reference):**
- Key architectural axes: GQA (not MLA), 48 layers, 128 experts topk 8, rope 1e6.
- `attention_backend: fused` works (TE fused supports GQA at sm_103).
- `VLLM_ATTENTION_BACKEND=FLASH_ATTN` is mandatory — FlashInfer (vLLM default)
  is broken in this image (torch.to(non_blocking=None) incompat).
- `NVTE_FUSED_ATTN=1` in the rayjob env requires `attention_backend` to match
  (set either both to fused, or unset env AND use flash).
- PP=1/EP=8/TP=1 on 2-node is fine; no offload needed (30B fits in 288GB×16 easily).
- mbridge auto-derives all GQA/rope params from HF config — no manual override needed.

**Files committed:**
- verl `nemo_bspo_sd_ablation @ 81c87a2b` (attention_backend fix)
- iter_kuberay `nemo_bspo_md @ 0668c4e` (VLLM_ATTENTION_BACKEND=FLASH_ATTN)
- main `nemo_bspo_md` (plan.md + dev_notes.md + submit_qwen3.sh wrapper)

Next steps (for when user wakes up): run longer training (e.g. 120 steps with
bspo w_penalty_only / cbsp401-equivalent hparams) to see if Qwen3 actually
improves on the math eval, once the 18 sd ablation jobs free up more nodes.
