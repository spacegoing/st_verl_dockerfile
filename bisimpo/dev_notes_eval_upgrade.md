# BSPO Dev Notes — Stage: Eval Set Upgrade

**Motivation:** The current k8s eval parquet (`0320_split/eval_nemogym_math.parquet`) is 125 rows, of which only the AIME-2024-adjacent slice is meaningfully hard. Model already scores > 0.7 on AIME 2024 → poor discrimination between BSPO variants. Upgrade to a harder, multi-source eval.

## Eval-set selection

Constraints (user):
- Total rows < 256 **unless BeyondAIME alone exceeds 256** (it does not — 100 rows).
- Must include BeyondAIME.
- Prefer **harder** benchmarks (to discriminate between algos).
- Per-source acc must be logged separately in wandb (no mixing across benchmarks).

Survey of the 12 HF eval sets downloaded under `/mnt/public/lichang93/downloads/datasets/` (see `papers_summary.md`):

| Dataset | Size | Difficulty | Contamination risk | Keep? | Why |
|---|---|---|---|---|---|
| **BeyondAIME** | 100 | Hardest (≥ AIME P11-15) | Very low (manually rewritten) | ✅ 100 | Mandatory; contamination-safe; designed for LLM math discrimination |
| **AIME 2025** (I+II) | 30 | Very hard | Low (post Moonlight-16B cutoff) | ✅ 30 | Post-2024 → minimal training-set overlap |
| **OlympiadBench / OE_TO_maths_en_COMP** | 674 | Olympiad-level, international | Moderate | ✅ 100 | Subsample to first 100 `(answer_type="Numerical", is_multiple_answer=False)`. Numerical answers simplify grading |
| AIME 2024 | 30 | Very hard | **High** (widely used in training) | ❌ | Already in current eval as the 30-problem slice; not useful as delta |
| MATH-500 | 500 | Mixed (levels 1-5) | High | ❌ | Model saturates (~0.8-0.9 on avg); too easy for variant discrimination |
| Minerva Math | 272 | Moderate | Moderate | ❌ | Size and difficulty not justified given we already have 230 from 3 harder sets |
| AMC 2023 | 40 | Easy/medium | High | ❌ | Saturation expected (> 0.85 on this class of model) |
| gsm8k | 1.3k | Easy | Very high | ❌ | Solved |
| CodeForces / CodeContests / TACO / LiveCodeBench | — | — | — | ❌ | Code, not math; training domain is math-only |
| DeepMath-103K | 103k | training corpus | — | ❌ | Training set, not eval |
| Reddit TL;DR | 12k | summarization | — | ❌ | Wrong task |

**Selected:** BeyondAIME (100) + AIME 2025 (30) + OlympiadBench-EN-Math-Comp subset (100) = **230 rows**, three distinct `data_source` tags.

## Parquet schema (verl-compatible)

One row per problem. Matches `0320_split/eval_nemogym_math.parquet`:

| Column | Type | Value |
|---|---|---|
| `data_source` | str | `"beyondaime"` / `"aime2025"` / `"olympiadbench"` |
| `prompt` | list of dict | `[{role: "system", content: JoyAI system prompt}, {role: "user", content: problem + "\n\nRemember to put your final answer inside \\boxed{}."}]` |
| `reward_model` | dict | `{"ground_truth": "", "style": "rule"}` |
| `ability` | str | `""` |
| `extra_info` | str (JSON) | `{"index": i, "agent_ref": {"type": "responses_api_agents", "name": "math_with_judge_simple_agent"}, "question": <problem>, "expected_answer": <answer>, "source_dataset": <name>}` |

The math judge extracts `expected_answer` from `extra_info` (see `verl/workers/reward_manager/nemogym_server.py:_build_verify_body`). Using distinct `data_source` tags produces distinct wandb keys:
- `val-core/beyondaime/acc/mean@1`
- `val-core/aime2025/acc/mean@1`
- `val-core/olympiadbench/acc/mean@1`

Aggregate math accuracy is NOT a simple mean of the three (size-weighted would be misleading) — we report per-source explicitly.

## Reward manager patch

Both `verl/workers/reward_manager/nemogym_server.py` and `verl/experimental/reward_loop/reward_manager/nemogym_server.py` share `_build_verify_body`. The math branch is gated by `data_source == "nemogym_math"`. Patch:

```python
_MATH_SOURCES = {"nemogym_math", "beyondaime", "aime2025", "olympiadbench"}
...
if data_source in _MATH_SOURCES:
    body["question"] = extra_info.get("question", "")
    body["expected_answer"] = extra_info.get("expected_answer", "")
```

No change to routing — `server_urls` is configured via Hydra at launch time.

## Hydra config wiring

In `verl/my_scripts/k8s/config/40bra_16node_sd.yaml`:

```yaml
data:
  val_files: /root/myCodeLab/host/downloads/datasets/eval_bspo_hard/eval.parquet
reward_model:
  reward_kwargs:
    server_urls:
      nemogym_math: http://localhost:20006
      beyondaime: http://localhost:20006
      aime2025: http://localhost:20006
      olympiadbench: http://localhost:20006
```

The 4 new URL entries all point at the same math server — the verifier dispatches via `expected_answer` field in `extra_info`, not via port.

## Artifacts

| Path | Purpose |
|---|---|
| `bisimpo/etl_eval_hard.py` | Builds `eval_bspo_hard/eval.parquet` from the 3 source datasets |
| `bisimpo/dev_notes_eval_upgrade.md` | This doc |
| `/mnt/public/lichang93/downloads/datasets/eval_bspo_hard/eval.parquet` | Output (230 rows) |

## Debug plan

1. Run ETL → 230-row parquet.
2. Apply reward-manager patch.
3. Update Hydra config.
4. Smoke via `submit_bspo.sh cdbgbspo` (1-step, `val_before_train=true`) → confirm 3 val-core keys appear in `run.log`.
5. Sanity-check val accuracy distribution (should be lower than 0.7 since these are harder; ~0.2-0.5 range expected for the base policy).
6. **Only after smoke passes:** delete Phase-1 rayjobs (cbsp101/103/104), resubmit via `launch_phase1.sh`. New runs inherit the new eval automatically.
7. Resume BSPO Phase-1 → analysis → Phase-2 as planned.

## Smoke submission record

| Time UTC | RayJob | What |
|---|---|---|
| 2026-04-18 18:05 | `bra40-sd-cdbgbspo-4n-gfc7n` | cdbgbspo with new 230-row eval; val_before_train=true; 1 step |

**Result: PASSED** at 18:19 UTC (val_before_train step 0, 230-sample eval).

All 3 new data_sources produced separate wandb keys and non-zero acc:

| data_source | val-core/<>/acc/mean@1 | val-aux/<>/reward/mean@1 |
|---|---|---|
| beyondaime   | 0.51  | 0.40 |
| aime2025     | 0.667 | 0.61 |
| olympiadbench | 0.68 | 0.62 |

Interpretation:
- 40Bra-16B is stronger than expected on these benchmarks (0.51–0.68 at step 0). BeyondAIME at 0.51 is actually a good operating point — large headroom in either direction (0→1) to detect BSPO effects.
- The gap between `val-aux/reward` (soft judge score, 0-1) and `val-core/acc` (binary match) reflects the math judge's tolerance. A correct answer gets ~0.9-1.0 reward; a close-but-wrong gets ~0.3-0.5.
- Per-source logging works: 3 distinct keys, no mixing. wandb will plot them as independent curves.

Monitor: `bdfsb9e9c` (watches for `val-core/(beyondaime|aime2025|olympiadbench)/acc/*` keys and terminal state).

Expected wall clock: ~35 min (5 min init + 10 min val-before + 6 min train + 10 min val-after + 4 min ckpt save).
Expected val distribution (base policy on hard math):
- `beyondaime`: 0.05-0.20 (very hard)
- `aime2025`: 0.15-0.40
- `olympiadbench`: 0.15-0.35

If any source returns 0.0 for all samples → indicates server routing / extra_info key-name mismatch, investigate.

## Post-upgrade Phase-1 launch procedure

```bash
# Delete in-flight Phase-1 with old eval
HTTPS_PROXY= HTTP_PROXY= NO_PROXY=180.184.249.201 kubectl delete rayjob \
  bra40-sd-cbsp101-4n-q5xxv bra40-sd-cbsp103-4n-tjxg7 bra40-sd-cbsp104-4n-x567v
# Relaunch (now with upgraded eval baked into the config)
bash bisimpo/launch_phase1.sh
```

## Phase-1 v2 relaunch (hard eval)

Executed at 18:36 UTC after smoke passed. Old jobs deleted, 3 fresh jobs submitted:

| Combo | RayJob v2 | Variant |
|---|---|---|
| cbsp101 | `bra40-sd-cbsp101-4n-w42xk` | simplest |
| cbsp103 | `bra40-sd-cbsp103-4n-8jgzx` | w_penalty |
| cbsp104 | `bra40-sd-cbsp104-4n-w92mz` | w_penalty_only |

Monitors: `b9nukq81n` (terminal), `bbh4pxuh3` (progress + val per source + errors).
ETA per run: ~20h. Expected completion ~14:30 UTC 2026-04-19.
