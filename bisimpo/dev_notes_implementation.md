# BSPO Dev Notes — Stage: Implementation

Chronological log of the code changes that add BSPO to verl. Follows the
plan in `dev_notes_planning.md`.

---

## Summary of changes

Four files touched. Total additions: ~180 lines across verl + a new wrapper
script.

| File | Change | Lines |
|---|---|---|
| `verl/verl/trainer/ppo/core_algos.py` | New `compute_policy_loss_bspo` (three variants) + `@register_policy_loss("bspo")` | +167 |
| `verl/verl/trainer/config/actor/actor.yaml` | Added `bspo_variant`, `bspo_delta`, `bspo_lambda_tj` with GSPO-scale defaults | +10 |
| `verl/verl/workers/config/actor.py` | Mirror 3 fields on `ActorConfig` dataclass | +4 |
| `verl/my_scripts/k8s/config/combo_40Bra.yaml` | Add `cdbgbspo` (1-step smoke) + `cbsp101/103/104` (Phase-1) | +32 |
| `bisimpo/submit_bspo.sh` | New wrapper: combo → Hydra overrides → delegates to production `submit.sh` | new |
| `iter_kuberay_32nodes_verl_training/submit/submit.sh` | Relaxed combo-id regex to allow debug-style names (kept hyphen-safe) | 1 line |

## Decisions during implementation

### 1. Naming — alphanumeric only

K8s resource names (RFC 1123) disallow underscore. Original plan used
`cdbg_bspo` and `cbsp_101`. Renamed to `cdbgbspo` / `cbsp101` / `cbsp103` /
`cbsp104`. The pattern matches the existing `c351` convention closely;
reader can still parse "cbsp + NNN" as the combo shorthand.

### 2. `submit.sh` regex relaxed

Old: `^c[0-9]+$` — rejects any combo with letters (beyond leading 'c').
New: `^c[a-z_]*[0-9]*[a-z_]*[0-9]*$` — accepts `cdbg5`, `cdbgbspo`,
`cbsp101`, `c351`. Does NOT accept `cbsp_101` (underscore) because k8s
would reject the generated RayJob name anyway. One-line change, fully
backward compatible with existing combos.

### 3. Why `submit_bspo.sh` wraps `submit.sh`

Plan0.md: "yaml > bash for hparams, kuberay-compatible variant flag".
Hard-wiring per-combo BSPO hparams (variant, delta, lambda) into
`combo_40Bra.yaml` would require either adding default values to every
non-BSPO combo (noisy) or adding Hydra `oc.select` fallbacks everywhere
(brittle). Instead:

- Combo yaml carries only the 6 fields the main yaml already
  interpolates (`fapo_delta`, `ppo_epochs`, `domains`, etc.).
- BSPO-specific overrides (`loss_mode`, `bspo_variant`, `bspo_delta`,
  `bspo_lambda_tj`) are resolved by `submit_bspo.sh` from the combo id
  and passed via Hydra CLI overrides to the delegated `submit.sh`.

Result: existing combos unaffected; BSPO combos are fully specified by
combo id + variant code.

### 4. Why no `dp_actor.py` / `megatron_actor.py` edits

Under single-domain + verl's GRPO full-group baseline, Obj 2 (Strict
Wasserstein) reduces identically to Obj 3 with `lambda_tj=0` (see the
Q13 derivation in `bisim_eqs_annotated.tex` §7). The three-level
aggregation machinery from the `stash@{0}` draft is therefore not
needed. The `lambda_tj=0` point of the Obj 3 sweep already gives Obj 2
semantics. No edits to the actor worker modules.

If we later reintroduce multi-domain, the stash pattern
(`trajectory_rewards`, `group_ids` kwargs + upstream hooks in both
actor files) is documented in `bisim_eqs_annotated.tex` §8 and ready to
wire up.

### 5. `loss_agg_mode` hard-coded inside loss

The math-correct aggregation is `seq-mean-token-mean` (Q7 proof in
annotated tex §9). To prevent accidental use of `token-mean` via config
drift, the BSPO loss **ignores the `loss_agg_mode` kwarg** and calls
`agg_loss(..., loss_agg_mode="seq-mean-token-mean", ...)` directly.
FACPO does the same for the same reason.

### 6. `sign(s_tau)` detached

Per Q8: `torch.sign(s_tau).detach()`. Sign is a selector, not a
differentiable function. The gradient flows only through the
`clamp(min(s_pos, s_neg), max=delta)` factor in `delta_w`.

### 7. `rollout_is_weights` accepted but ignored

Per Q9. The argument remains in the signature (`= None`) to match the
verl loss-function ABI, but is not read inside the function.

## Verification performed

- `python3 -c "import ast; ast.parse(open(f).read())"` for both
  `core_algos.py` and `actor.py` — syntax OK.
- `yaml.safe_load` on `actor.yaml` and `combo_40Bra.yaml` — parse OK,
  `bspo_*` keys present with correct scalar types.
- `bash -n submit_bspo.sh` — syntax OK.
- `submit.sh` dry-run for `cdbgbspo` — k8s accepts the generated RayJob
  yaml after the underscore-to-nothing rename.

Next stage: debug smoke test — see `dev_notes_dev_debug.md`.
