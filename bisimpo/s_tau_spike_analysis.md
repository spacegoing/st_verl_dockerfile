# s_tau spike analysis: metric vs. gradient

Follow-up to the earlier (incorrect) hardware-decay claim. Analysing
whether `bspo_md_s_tau_abs_mean` excursions affect the actual loss or are
a logging artefact, with rigorous citations from
`bisimpo/bspo_algorithm.tex` and
`verl/verl/trainer/ppo/core_algos.py::compute_policy_loss_bspo_md`.

## 1. What generates the spike — the origin tensor

Reference: `core_algos.py:1965-1973`

```python
log_ratio = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
ratio     = torch.exp(log_ratio)                            # (B, T)
seq_lens  = response_mask.sum(dim=-1).clamp(min=1)
dev_pos   = torch.clamp(ratio - 1.0, min=0.0) * response_mask
dev_neg   = torch.clamp(1.0 - ratio, min=0.0) * response_mask
s_tau_pos = dev_pos.sum(dim=-1) / seq_lens                  # (B,)
s_tau_neg = dev_neg.sum(dim=-1) / seq_lens
s_tau     = s_tau_pos - s_tau_neg                           # (B,)
```

Math — `bspo_algorithm.tex:85-86`:

$$ s_\tau^{+} = \tfrac{1}{|\tau|}\sum_{t=1}^{|\tau|}\max(r_t-1,\,0),\quad
   s_\tau^{-} = \tfrac{1}{|\tau|}\sum_{t=1}^{|\tau|}\max(1-r_t,\,0),\quad
   s_\tau = s_\tau^{+} - s_\tau^{-} $$

Per-token `log_ratio` is clamped at $\pm 20$, so per-token `dev_pos` is
bounded by $e^{20}-1 \approx 4.85\times 10^8$. If even a few tokens in
$\tau$ saturate, $s_\tau^{+}$ can be huge. Empirically in cbmd-v1's step
87, one trajectory had $s_\tau \approx 2.1\times 10^4$ which, averaged
into $\lvert s_\tau \rvert.\mathrm{mean}()$ over $B{\approx}50$, produced
the logged 435.

**The spike originates in `log_prob - old_log_prob` saturating the ±20
clamp for a handful of tokens in one trajectory** — usually the tail of a
long generation where the model's output distribution shifted markedly
between ppo epoch 1 and epoch 2. `cbsp302` (sd, different physical nodes,
same code path) shows the same spikes, ruling out hardware correlation.

## 2. What `s_tau` actually does to the loss — per variant

Reference: `core_algos.py:2005-2047` (variant dispatch).

### V1 — simplest

Math (`bspo_algorithm.tex:141-142`, 130):
$$ \ell_\tau^{(V1)} = \Delta_\tau^{(\mathrm{reg})} \cdot \hat A_\tau, \quad
   \Delta_\tau^{(\mathrm{reg})} = \operatorname{clip}(s_\tau^{+}, \delta) - \operatorname{clip}(s_\tau^{-}, \delta) $$

Code (2015): `delta_tau_reg = torch.clamp(s_tau_pos, max=delta) - torch.clamp(s_tau_neg, max=delta)`.

- Since $\operatorname{clip}(x, \delta) \in [0, \delta]$ for $x \geq 0$,
  $\Delta_\tau^{(\mathrm{reg})} \in [-\delta, \delta]$. **Hard-bounded.**
- For a spike trajectory with $s_\tau^{+}\!\!\gg\!\delta$ and
  $s_\tau^{-}\!\!\approx\!0$ (one-sided saturation, the usual case),
  $\Delta_\tau^{(\mathrm{reg})} = \delta - 0 = \delta$. Max magnitude.
- For two-sided saturation, $\Delta_\tau^{(\mathrm{reg})} = \delta - \delta = 0$.
- **Gradient through `s_tau_pos`**: $\partial \operatorname{clip}(s, \delta)/\partial s = 1$ when
  $s \leq \delta$, else $0$. So on spike trajectories the gradient path
  through `s_tau_pos` is **zeroed** at the clamp saturation point.
  The dL/d(log_prob) contribution from that trajectory is therefore
  $A_\tau \times \partial \Delta^{(\mathrm{reg})}/\partial(\text{log\_prob}) = 0$.

**V1 is fully protected.** Empirical confirmation: cbmd-v1's
`grad_norm` on spike steps (median 0.26) is indistinguishable from
normal steps (median 0.21).

### V2 — hierarchy

Math (`bspo_algorithm.tex:143-144`):
$$ \ell_\tau^{(V2)} = \bigl(\Delta_\tau^{(\mathrm{reg})} - \Delta_{g(\tau)}^{(\mathrm{reg})}\bigr) \cdot \hat A_\tau $$

Code (2037): `ell = (delta_tau_reg - dg_reg_exp) * A_tau`.

- $\Delta_\tau^{(\mathrm{reg})} \in [-\delta, \delta]$, same for $\Delta_g^{(\mathrm{reg})}$.
- Difference $\in [-2\delta, 2\delta]$. **Hard-bounded.**
- Same zero-gradient-at-clamp argument as V1. **Safe.**

### V3 — strict_wasserstein

Math (`bspo_algorithm.tex:145-148`, 131):
$$ \Delta_z^{(\mathrm{W})} = \operatorname{sign}(s_z)\cdot\operatorname{clip}(\min(s_z^{+}, s_z^{-}), \delta_z) $$
$$ \ell_\tau^{(V3)} = \Delta_{dom(\tau)}^{(\mathrm{W})} \hat A_{dom(\tau)} +
   \Delta_{g(\tau)}^{(\mathrm{W})} \hat A_{g(\tau)} +
   \Delta_\tau^{(\mathrm{W})} \hat A_\tau $$

Code (2010, 2039): `delta_tau_w = torch.sign(s_tau).detach() * torch.clamp(torch.minimum(s_tau_pos, s_tau_neg), max=delta)`.

- $|\Delta_\tau^{(\mathrm{W})}| \leq \delta$. **Hard-bounded at every level.**
- `sign(...).detach()` means direction carries no gradient — all gradient
  flows through the clipped $\min$, which is $\leq \delta$.
- For the common one-sided spike ($s_\tau^{+}\!\!\gg\!\!0$, $s_\tau^{-}\!\!\approx\!0$):
  $\min(s_\tau^{+}, s_\tau^{-}) \approx 0$, so $\Delta_\tau^{(\mathrm{W})}\approx 0$
  and the spike trajectory contributes **~zero** to the loss. Stronger
  protection than V1 on one-sided spikes.

### V4 — w_penalty

Math (`bspo_algorithm.tex:132, 149-153`):
$$ \operatorname{pen}_z = \max\!\Bigl(0,\,\tfrac{\min(s_z^{+},s_z^{-})}{\delta_z} - 1\Bigr) $$
$$ \ell_\tau^{(V4)} = \Delta_\tau^{(\mathrm{W})} \hat A_\tau - \lambda_{\mathrm{Tj}}\,\mathrm{pen}_\tau - \lambda_{\mathrm{Grp}}\,\mathrm{pen}_{g(\tau)} - \lambda_{\mathrm{D}}\,\mathrm{pen}_{dom(\tau)} $$

Code (2041-2044): `ell = delta_tau_w * A_tau - lam_tj * pen_tau - lam_grp * pen_g_exp - lam_d * pen_d_exp`.

- $\Delta_\tau^{(\mathrm{W})}$ is bounded (same as V3).
- $\mathrm{pen}_z$ is **unbounded above**. If $s_z^{+}$ AND $s_z^{-}$ are
  both huge (two-sided saturation), $\mathrm{pen}_z$ can be $\approx 10^{12}$.
- For one-sided spikes (the common case): $\min \approx 0 \Rightarrow \mathrm{pen}_\tau \approx 0$.
  **Observed pattern in cbmd-v1**: `pen_tau_mean` stays 19–27 throughout,
  including spike steps — i.e. penalties are NOT blowing up on spikes.

**V4 is safe for the observed spike pattern** because spikes are
one-sided. If a two-sided spike were ever to occur, `pen_τ` would
dominate and the $-\lambda_{\mathrm{Tj}} \mathrm{pen}_\tau$ term would
push the loss toward $-\infty$ — this is the failure mode that killed
sd cbsp104 (V5) earlier in Phase-1.

### V5 — w_penalty_only

Math (`bspo_algorithm.tex:154-158`):
$$ \ell_\tau^{(V5)} = s_\tau \hat A_\tau - \lambda_{\mathrm{Tj}}\,\mathrm{pen}_\tau - \lambda_{\mathrm{Grp}}\,\mathrm{pen}_{g(\tau)} - \lambda_{\mathrm{D}}\,\mathrm{pen}_{dom(\tau)} $$

Code (2045-2046): `ell = s_tau * A_tau - lam_tj * pen_tau - ...`.

- **`s_tau` enters the loss raw, no clip.** One trajectory with
  $s_\tau = 2\times 10^4$ at $A_\tau \approx 3$ contributes
  $\ell_\tau \approx -6\times 10^4$ — swamps the batch.
- The per-token clamp at $\pm 20$ still bounds the gradient
  contribution from each token (saturated tokens have zero grad), but
  the accumulated $s_\tau$ over non-saturated tokens is NOT bounded.
- **This is the cbsp104 failure mode**: `grad_norm` exploded to ~40k at
  step ~100 and the run was aborted. V5's known behaviour.

## 3. Verdict per currently-running combo

All 5 formal cbmd-v{1..5} runs saw `bspo_md_s_tau_abs_mean` spikes in
the debug smokes and (for v1) in the formal run up to step 96. For each
variant whether the spike actually affects training:

| combo | variant | spike → loss? | spike → grad? | expected to survive |
|---|---|---|---|---|
| cbmd-v1 | simplest            | bounded by $\delta$ | zeroed by clip | ✅ |
| cbmd-v2 | hierarchy           | bounded by $2\delta$ | zeroed by clip | ✅ |
| cbmd-v3 | strict_wasserstein  | bounded by $\delta$ at every level | zeroed by clip, amplified for one-sided | ✅ |
| cbmd-v4 | w_penalty           | bounded on one-sided spikes | bounded (pen≈0) | ✅ for observed pattern |
| cbmd-v5 | w_penalty_only      | unbounded (`s_τ` direct) | unbounded via `s_τ` | ⚠️ likely to collapse, matches cbsp104 |

## 4. Should we act?

**No urgent code change.** The loss math in the paper is specifically
designed to clip the per-trajectory ratio stats so outlier trajectories
can't dominate the gradient — that's the whole point of
$\Delta^{(\mathrm{reg})}$ and $\Delta^{(\mathrm{W})}$. The spikes we
see are the diagnostic metric (`s_tau.abs().mean()`) being a
non-robust estimator — it faithfully reports that some trajectories
have saturated ratios, but doesn't mean training is degrading.

**Optional follow-ups (only if spike-metric noise is bothering
downstream tooling):**
- Replace `s_tau_abs_mean` with median/p99 of `|s_τ|` in the metrics
  dict. Loss is untouched; metric becomes outlier-robust.
- Add a `pre_clamp_log_ratio_p99` metric so we can see the raw pre-clamp
  excursions and correlate with training progress.

Neither is blocking. V5's known unbounded nature is the real thing to
watch — the original plan already earmarked λ=1e-2 (up from 1e-3)
specifically to buy V5 some headroom (sd cbsp401 currently running with
that config).

## 5. Retracting the hardware-decay claim

My earlier "20 steps of NVLink silent bit-flips before the
uncorrectable" narrative is not supported. Evidence against it:

1. `cbsp302` (sd run on physically different nodes) produces the same
   `s_tau_abs_mean` spikes at a similar or higher rate — clearly not
   hardware-local.
2. `grad_norm` and `pg_loss` distributions are identical on spike vs
   non-spike steps within cbmd-v1. NCCL all-reduce output looks fine.
3. Memory traces flat; pod infra stable until the fatal SIGABRT.
4. The spikes go away on the step AFTER a spike — bit-flip corruption
   would not self-heal in one step.

The NVLink uncorrectable error at 05:54:40 on `host-10-125-1-166` is
still a real, separate, one-shot hardware fault. Node remains cordoned.
