# BSPO — open questions before implementation

Questions the LaTeX + fiberpo code did not fully answer. Please reply in line
(e.g. `Q1. arithmetic`) and I will update `bisim_eqs_annotated.tex` plus draft
the implementation plan.

Questions are ordered so that **Q1, Q6, Q7** are the blockers — they change
the code structure. The rest can be defaulted if you do not care.

---

## Status (updated 2026-04-18)

- **Q1** ✅ answered: **arithmetic** `max(r − 1, 0)` / `max(1 − r, 0)`.
  `bisim_eqs_annotated.tex` §2 updated accordingly; the log-form block was
  removed.
- **Q2** ✅ answered: the GSPO-style derivation at the top of
  `bisim_eqs.tex` is **motivation only** (hparams should be comparable in
  magnitude to GSPO's, but the loss is unrelated). $\hat d(z)$ is the
  definition of reward MAD, used nowhere in the 4 losses. **Single-domain
  only for now** — drop every term that is constant over a single-domain
  batch. `bisim_eqs_annotated.tex` §6 rewritten, §7 added to flag a
  consequence for Obj 2.
- **Q3–Q12** still open.
- **Q13** new, added below — blocker for Objective 2 in single-domain.
- **Q14** new, added below — depends on Q13.

---

## Q1 — `r - 1` vs `log r` for the $s^+$, $s^-$ aggregates

The LaTeX `bisim_eqs.tex` defines:

```
s_τ^+ = (1/|τ|) Σ_t max(r_t − 1, 0)
s_τ^- = (1/|τ|) Σ_t max(1 − r_t, 0)
```

But the FiberPO code in `fiberpo_in_core_algos.py` uses log-ratio:

```
log_s_pos = (clamp(log_r, 0) · mask).sum(-1) / T_τ
log_s_neg = (clamp(-log_r, 0) · mask).sum(-1) / T_τ
```

The two are close for `r ≈ 1` but diverge at higher deviations. The verl
config `fiberpo_use_abs_deviation` flag already documents this choice. Which
form is canonical for BSPO?

- (a) Arithmetic `r − 1` (matches bisim_eqs.tex exactly)

**Impact:** one numpy line. Pick one and I will thread it through.

---

## Q2 — Is $\hat d(z) = \mathbb{E}[|R − \mathbb{E}[R]|]$ part of the loss, or just motivation?

The reward deviation $\hat d$ appears in the derivation of
${}^{(\text{GSPO})}\tilde\delta_\tau$ at the top of the file, but **does not**
appear in any of the 4 objectives. It is used purely to argue why the
$s^\pm$ quantities are proxies for trajectory-importance.


sorry for the confusion, but !!!

closely pay attention here: the
${}^{(\text{GSPO})}\tilde\delta_\tau$ has nothing to do with gspo
loss, I only make it significant visually to remind us, hparams
here should be relatively comparable to gspo, but the loss has
nothing to do with gspo

\hat d(z) is the exact implementation / definition of all \hat
d(dom) and \hat d(g) etc.


---

## Q3 — $\delta_\tau$, $\delta_g$, $\delta_\text{dom}$: three independent hparams, or one?

Objective 2 (Strict Wasserstein) uses three $\delta$ parameters. Options:

- (b) Independent: `bspo_delta_tau`, `bspo_delta_g`, `bspo_delta_dom`
  (more flexible; could ablate the ratio).

we are only implementing single domain for now. so if some
term is constant for single domain case, drop them. and for the
sake of debugging / simplicity, when never possible, use constant
over code vars, do not consider compatability to multi-domain for
now. because we are still at dev stage, my theory is not formal
yet, and need to keep implementation simple enough to debug /
ablate etc.

---

## Q4 — $\lambda_\text{Tj}$: fixed value, tunable, or ablation sweep?

LaTeX says `λ_Tj = 0.001 or 1`. Which is it?

- (c) Sweep `[0, 0.001, 0.005]`.

---

## Q5 — `Â_dom = (16/128) Σ_{g:dom} Â_g` — what if a domain has a different number of groups this step?

Batch size 128 and group size 16 → 8 groups per step if the batch is
single-domain. With curriculum sampling the batch is usually multi-domain,
so different domains have different group counts per step (e.g. 5 math
groups and 3 code groups).

The `(16/128)` factor literally gives 1/8, which is correct only when there
are exactly 8 groups in that domain. For the general case I'd use
`(1/num_groups_in_dom)` i.e. the usual mean. Please confirm:

- (a) Always use `mean_{g:dom}(Â_g)` — my default assumption.
- (b) Literal `(16/128) · Σ Â_g` — penalizes small-domain contributions.
- (c) Weight by group size.

good question, keep it here, remind us later, but we are only
implementing single domain for now, so pls don't add this
complexity.

we are only implementing single domain for now. so if some
term is constant for single domain case, drop them. and for the
sake of debugging / simplicity, when never possible, use constant
over code vars, do not consider compatability to multi-domain for
now. because we are still at dev stage, my theory is not formal
yet, and need to keep implementation simple enough to debug /
ablate etc.

---

## Q6 — How should `group_ids` and `domain_ids` reach the loss function?

Objective 2 (and any reward-weighted variant from Q2) needs group/domain
membership at loss-compute time. The current verl loss fn signature is:

```
def compute_policy_loss_X(
    old_log_prob, log_prob, advantages, response_mask,
    loss_agg_mode, config, rollout_is_weights,
) -> (loss, metrics)
```

No group/domain. Options to fix:

- (a) Extend signature with `meta: Optional[dict] = None` carrying `{"group_ids":
  Tensor, "domain_ids": Tensor}`. Minimal invasion; only the loss dispatcher
  needs updating.
- (b) Pre-compute `s_pos_g`, `s_neg_g`, `A_g`, etc. upstream and pass as
  pre-aggregated tensors. Less flexible but avoids the signature change.
- (c) Follow your `stash@{0} group loss` draft — please point me at it so
  I can see the pattern you prefer.

Your preference?: as instructed in plan0.md u read @../verl's
stash first, learn the real implementation, then think hard on
this and come up with a best practice design.

**Note:** verl's `uid` (used by some codepaths) gives per-sample group id.
Domain id needs to come from the curriculum sampler's `data_source` column.
I can wire both as `(B,)` int tensors.

---

## Q7 — Per-trajectory scalar objective, broadcast to per-token for `agg_loss`?

In all four objectives the inner term is per-trajectory (scalar × scalar per
traj). Verl's `agg_loss` expects `(B, T)` with a mask. Two ways:

- (a) Broadcast the per-traj scalar to all valid tokens, then `agg_loss` with
  `"token-mean"` gives `Σ(per_traj * T_τ) / Σ T_τ` — length-weighted.
- (b) Broadcast, then use `"seq-mean-token-mean"` → each traj contributes
  equally regardless of length (matches the standalone NumPy reference
  aggregation `loss_agg_mode="token-mean"` if we divide by `T_τ` first).
- (c) Compute the scalar directly and return it, skipping `agg_loss` entirely.

FACPO forces `"seq-mean-token-mean"` for this reason; I'd do the same for
BSPO unless you prefer otherwise.

which one is mathematically correct? or is this a var worths to
sweep?

choose math correctness over sweep, but do sweep if both math correct.

---

## Q8 — Does `sgn(s_z)` flow gradient, or treat as detached?

$\Delta_z^{(W)} = \operatorname{sgn}(s_z) \cdot \operatorname{clip}(\min(s_z^+, s_z^-), \delta_z)$.

`sgn` is non-differentiable. Two options:

- (a) `torch.sign(s_z).detach()` — sign is a selector, gradient flows only
  through the `clip(min(...))` term (my default).

---

## Q9 — Interaction with `rollout_is_weights`

Verl's loss functions accept an optional `rollout_is_weights` that rescales
off-policy corrections (used e.g. for LoRA-style rollout distillation). For
BSPO:

- (b) Ignore (BSPO's $\Delta$ already captures the correction).

---

## Q10 — Scope of ablation matrix

The plan says "4 variants to run to see which implementation wins". Is the
ablation just **{Obj 1, Obj 2, Obj 3, Obj 4} × default hparams**, or do you
want each objective swept on its hparams (`δ`, `λ_Tj`, `s` form)?

- (c) Full grid (you specify the axes).

!!! this is very important !!!
we have identified some hparams to sweep along this doc, all them
in the vision. but do priority sweeps, our first priority is to
get knowledge which Obj 1, Obj 2, Obj 3, Obj 4, is the best, not
hparams.

so do design our sweep with the shortest time we can get a
conclusion which Obj variant is the best / differentiate them in
shortest run in mind

---

## Q11 — Naming: `bspo_` vs `bisimpo_` vs something else

Implementation will add ~10 config fields. Prefix?

- (a) `bspo_` (shortest; matches the filename `plan0.md` abbreviation).

---

## Q12 — Anything not captured by the 4 objectives?

Skimming `bisim_eqs.tex` I see a few symbols with ambiguous definitions
(e.g. `δ_D`, `R_{max res}(dom)`, the quadratic sum $\sum \hat d_{16}(g')^2$).
Are these required for any implemented variant, or are they part of the
derivation only?

If required, please give an implementation hint for each (or confirm "skip").

Per Q2's single-domain rule, all of these are constant per single-domain
batch, so they drop from the gradient. I am treating them as "skip for now"
unless you say otherwise.

D is domain
g is group
\tau Tj is trajectory

---

## Q13 — Objective 2 under single-domain + GRPO: does it reduce to Objective 3 without penalty?

**Blocker for Objective 2 implementation.**

From `bisim_eqs.tex`:

$$\hat A_\tau = (R_\tau - \bar R_g) / \sigma_g, \qquad
  \hat A_g = \tfrac{1}{n} \sum_{\tau \in g} \hat A_\tau, \qquad
  \hat A_\text{dom} = \tfrac{1}{G_d} \sum_{g \in \text{dom}} \hat A_g$$

GRPO normalization forces $\sum_{\tau \in g} \hat A_\tau \equiv 0$
exactly, so $\hat A_g \equiv 0$ and therefore $\hat A_\text{dom} \equiv 0$.
The two upper-level terms of Objective 2

$$\hat J = \mathbb{E}[\Delta_\text{dom}^{(W)} \hat A_\text{dom}
                    + \Delta_g^{(W)} \hat A_g
                    + \Delta_\tau^{(W)} \hat A_\tau]$$

vanish in single-domain mode. What remains is
$\mathbb{E}[\Delta_\tau^{(W)} \hat A_\tau]$ — which is Objective 3 with
$\lambda_{\text{Tj}} = 0$.


Options:
- **(a)** Confirm reduction; drop Obj 2 from the single-domain ablation
  and keep the 3-level machinery behind a `bspo_enable_multidomain` flag
  for future work.
- **(b)** Redefine $\hat A_g$ / $\hat A_\text{dom}$ without GRPO
  normalization — e.g. $\hat A_g = \bar R_g - R_{\text{global}}$,
  $\hat A_\text{dom} = \bar R_\text{dom} - R_{\text{global}}$ (raw-reward
  deviations). Please specify the intended formula.
- **(c)** Use $\hat d(g)$ (MAD of $R$ within group) in place of
  $\hat A_g$ at the group/domain level — consistent with Q2's $\hat d$
  definition and non-zero under GRPO.
- **(d)** Your intended semantics (please describe).

u need to double check on this: even for single domain, I think
group still here, and delta is not IS ratio in GRPO, I'm not sure
about the equiv u derive above, but pls give it a rigorous
derivation in math notation. and the thumb rule is: math always
win, as long as the math is correct, we only use theoretically
rigorous implementation.

---

## Q14 — Ablate only the 4 objectives in single-domain, or smaller set?

If Q13 resolves as option (a), Objective 2 is redundant in single-domain
and the effective ablation matrix is {Obj 1, Obj 3, Obj 4} — three runs.

Please confirm which list for the first ablation run:

Ok if obj 2 does redundant then only 3 objs, but as stated above,
pls double check in math.
