"""
FiberPO: Fiber-corrected Proximal Optimization Loss  (NumPy reference impl)
============================================================================

A variant of PPO-style clipped surrogate objective with trajectory-level
fiber corrections for importance sampling ratios.

Mathematical notation (unified):
---------------------------------
  g   : group index (GRPO group)
  j   : trajectory index within group g  (= row in batch)
  i   : token index within trajectory j  (= column)
  T_j : sequence length of trajectory j  (= response_mask.sum per row)

  log r_{j,i}     = log π_θ(a_{j,i}) − log π_{θ_old}(a_{j,i})
                   = log_prob[j,i] − old_log_prob[j,i]

  l_{j,i}         = sign(log r_{j,i})

  log s⁺_j        = (1/T_j) Σ_i max(log r_{j,i}, 0)
  log s⁻_j        = (1/T_j) Σ_i max(−log r_{j,i}, 0)

  g^agg(x, C, k)  = ⎧ x                              if |x| ≤ C
                     ⎪ sign(x)·(k+1)·C − k·x          if C < |x| < (1+1/k)·C
                     ⎩ 0                               otherwise

  logclip(r, ε)   = exp(clip(log r, −ε, +ε))

  fiber_weight_j  = exp( g^agg(log s⁺_j, δ/2, T_j)
                       − g^agg(log s⁻_j, δ/2, T_j) )

  corrected_log_r_{j,i} = log r_{j,i}
                         − sign(log r_{j,i}) · log s^{sign(log r_{j,i})}_j

  ε(Â)            = ε_pos if Â > 0   else ε_neg

  L_{j,i}         = fiber_w_j · logclip(corrected_r_{j,i}, ε(Â)) · Â_{j,i}

  Quadrant bypass overlays (applied after L_{j,i}):
    clip_ratio_c (c):      A<0, log_r>0 → raw r·A floored at c·A
    unclip_pos_floor:      A>0, log_r<0 → raw r·A (full recovery gradient)
  Both bypass logclip AND fiber weight for their respective quadrants.

  J^FiberPO(θ)    = E_g[ (1/|J_g|) Σ_j (1/T_j) Σ_i L_{j,i} ]

Note
----
Pure NumPy implementation.  All array shapes follow (B, T) convention
where B = batch_size (trajectories), T = response_length (padded).
"""

from __future__ import annotations
from typing import Any, Optional, Tuple
import numpy as np

# ============================================================================
# Hyperparameters
# ============================================================================
EPSILON_POS: float = 0.2       # ε_pos : clip bound when Â > 0
EPSILON_NEG: float = 0.2       # ε_neg : clip bound when Â < 0
DELTA: float = 0.4             # δ     : fiber aggregation threshold (C = δ/2)
CLIP_RATIO_C: float | None = None  # c : dual-clip lower bound when Â < 0 (None = disabled)
UNCLIP_POS_FLOOR: bool = False     # When True, bypass logclip+fiber for A>0, log_r<0

_SENTINEL = object()  # used for keyword-only override detection


# ============================================================================
# Core mathematical primitives
# ============================================================================

def compute_log_ratio(
    log_prob: np.ndarray,
    old_log_prob: np.ndarray,
) -> np.ndarray:
    """
    Per-token log importance sampling ratio.

    log r_{j,i} = log π_θ(a_{j,i}) − log π_{θ_old}(a_{j,i})

    Args:
        log_prob:     (B, T)  log-probs under current policy π_θ
        old_log_prob: (B, T)  log-probs under old policy π_{θ_old}

    Returns:
        log_ratio:    (B, T)  per-token log importance ratio
    """
    return log_prob - old_log_prob  # (B, T)


def compute_trajectory_log_s(
    log_ratio: np.ndarray,
    response_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trajectory-level positive / negative log-ratio statistics.

    log s⁺_j = (1/T_j) Σ_{i} max(log r_{j,i}, 0)
    log s⁻_j = (1/T_j) Σ_{i} max(−log r_{j,i}, 0)

    Args:
        log_ratio:     (B, T)  per-token log importance ratio
        response_mask: (B, T)  binary mask, 1 for valid tokens

    Returns:
        log_s_pos:     (B,)   trajectory-mean of positive log-ratio parts
        log_s_neg:     (B,)   trajectory-mean of negative log-ratio parts
    """
    seq_lens = np.maximum(response_mask.sum(axis=-1), 1)  # (B,)

    log_s_pos = (
        (np.maximum(log_ratio, 0.0) * response_mask).sum(axis=-1) / seq_lens
    )  # (B,)
    log_s_neg = (
        (np.maximum(-log_ratio, 0.0) * response_mask).sum(axis=-1) / seq_lens
    )  # (B,)

    return log_s_pos, log_s_neg


def g_agg(
    log_s: np.ndarray,
    C: float,
    k: np.ndarray | float,
) -> np.ndarray:
    """
    Piecewise linear aggregation with soft clamping.

    g^agg(x, C, k) =
        x                                 if |x| ≤ C           (identity)
        sign(x)·(k+1)·C − k·x            if C < |x| < (1+1/k)C  (decay)
        0                                 otherwise            (clamped)

    Continuity:
        At |x| = C          → g^agg = x                    (matches identity)
        At |x| = (1+1/k)·C  → g^agg = 0                   (matches clamped)

    Args:
        log_s: (B,) or (N,)  input values
        C:     scalar         threshold (= δ/2)
        k:     (B,) or scalar decay rate (typically T_j)

    Returns:
        out:   same shape as log_s
    """
    log_s = np.asarray(log_s, dtype=float)
    abs_log_s = np.abs(log_s)
    sign_log_s = np.sign(log_s)

    k_safe = np.maximum(np.asarray(k, dtype=float), 1.0)
    upper_bound = (1.0 + 1.0 / k_safe) * C  # (1 + 1/k) · C

    identity_mask = abs_log_s <= C
    decay_mask = (abs_log_s > C) & (abs_log_s < upper_bound)

    identity_val = log_s
    decay_val = sign_log_s * (k_safe + 1) * C - k_safe * log_s

    out = np.zeros_like(log_s)
    out = np.where(identity_mask, identity_val, out)
    out = np.where(decay_mask, decay_val, out)

    return out


def logclip(
    log_r: np.ndarray,
    epsilon: np.ndarray | float,
) -> np.ndarray:
    """
    Symmetric log-space clipping of importance ratio.

    logclip(r, ε) = exp(clip(log r, −ε, +ε))

    Args:
        log_r:   (B, T) or (N,)  log of the ratio to clip
        epsilon: same shape       per-element clip bound

    Returns:
        clipped_ratio: same shape  exp(clip(log_r, −ε, +ε))
    """
    return np.exp(np.clip(log_r, -epsilon, epsilon))


# ============================================================================
# Composite building blocks
# ============================================================================

def compute_fiber_weight(
    log_s_pos: np.ndarray,
    log_s_neg: np.ndarray,
    delta: float,
    seq_lens: np.ndarray,
) -> np.ndarray:
    """
    Trajectory-level fiber correction weight.

    fiber_w_j = exp( g^agg(log s⁺_j, C, T_j)
                   − g^agg(log s⁻_j, C, T_j) )     where C = δ/2

    Args:
        log_s_pos: (B,)  trajectory-mean positive log-ratio
        log_s_neg: (B,)  trajectory-mean negative log-ratio
        delta:     scalar
        seq_lens:  (B,)  sequence lengths T_j

    Returns:
        fiber_w:   (B,)  per-trajectory weight ≥ 0
    """
    C = delta / 2.0
    g_pos = g_agg(log_s_pos, C, seq_lens)  # (B,)
    g_neg = g_agg(log_s_neg, C, seq_lens)  # (B,)
    return np.exp(g_pos - g_neg)            # (B,)


def compute_corrected_log_ratio(
    log_ratio: np.ndarray,
    log_s_pos: np.ndarray,
    log_s_neg: np.ndarray,
    response_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fiber-corrected per-token log importance ratio (numerator).

    corrected_i = log r_i - sign(log r_i) · log s^{sign(log r_i)}

    l = +1:  log r - log s⁺    (deflate positive tokens)
    l = −1:  log r + log s⁻    (inflate negative tokens toward 0)
    l =  0:  log r             (no correction)

    Args:
        log_ratio:     (B, T)
        log_s_pos:     (B,)
        log_s_neg:     (B,)
        response_mask: (B, T)  optional mask (unused, API compat)

    Returns:
        corrected: (B, T)
    """
    sign_lr = np.sign(log_ratio)                         # (B, T)
    lsp = log_s_pos[:, None]                             # (B, 1)
    lsn = log_s_neg[:, None]                             # (B, 1)
    # select same-sign statistic: s⁺ when l=+1, s⁻ when l=−1
    log_s_same = np.where(sign_lr > 0, lsp,
                 np.where(sign_lr < 0, lsn,
                          np.zeros_like(log_ratio)))     # (B, T)
    return log_ratio - sign_lr * log_s_same              # (B, T)


def compute_cross_sign_log(
    log_ratio: np.ndarray,
    log_s_pos: np.ndarray,
    log_s_neg: np.ndarray,
) -> np.ndarray:
    """
    Cross-sign denominator log term (v2).

    denom_log_{j,i} = −l_{j,i} · log s^{−l_{j,i}}_j

    l = +1:  −log s⁻ ≤ 0
    l = −1:  +log s⁺ ≥ 0
    l =  0:  0

    Same-signed as numer → clips cancel at log r = 0.

    Args:
        log_ratio: (B, T)
        log_s_pos: (B,)
        log_s_neg: (B,)

    Returns:
        denom_log: (B, T)
    """
    sign_lr = np.sign(log_ratio)                         # (B, T)
    lsp = log_s_pos[:, None]                             # (B, 1)
    lsn = log_s_neg[:, None]                             # (B, 1)
    # select opposite-sign statistic: s⁻ when l=+1, s⁺ when l=−1
    log_s_opp = np.where(sign_lr > 0, lsn,
                np.where(sign_lr < 0, lsp,
                         np.zeros_like(log_ratio)))      # (B, T)
    return -sign_lr * log_s_opp                          # (B, T)


def compute_per_element_epsilon(
    advantages: np.ndarray,
    eps_pos: float,
    eps_neg: float,
) -> np.ndarray:
    """
    ε(Â) = ε_pos if Â > 0, else ε_neg.

    Args:
        advantages: (B, T)
        eps_pos:    scalar
        eps_neg:    scalar

    Returns:
        epsilon:    (B, T)
    """
    return np.where(advantages > 0, eps_pos, eps_neg)


# ============================================================================
# Loss aggregation
# ============================================================================

def aggregate_loss(
    per_token_loss: np.ndarray,
    response_mask: np.ndarray,
    mode: str = "token-mean",
) -> float:
    """
    Aggregate per-token loss.

    "token-mean" : Σ L·mask / Σ mask
    "seq-mean"   : (1/B) Σ_j (Σ_i L·mask / T_j)

    Args:
        per_token_loss: (B, T)
        response_mask:  (B, T)
        mode:           str

    Returns:
        scalar loss value
    """
    masked = per_token_loss * response_mask

    if mode == "token-mean":
        return float(masked.sum() / max(response_mask.sum(), 1))
    elif mode == "seq-mean":
        seq_lens = np.maximum(response_mask.sum(axis=-1), 1)
        return float((masked.sum(axis=-1) / seq_lens).mean())
    else:
        raise ValueError(f"Unknown loss_agg_mode: {mode}")


# ============================================================================
# Main entry point
# ============================================================================

def compute_policy_loss_vanilla(
    old_log_prob: np.ndarray,
    log_prob: np.ndarray,
    advantages: np.ndarray,
    response_mask: np.ndarray,
    loss_agg_mode: str = "token-mean",
    config: Optional[Any] = None,
    rollout_is_weights: np.ndarray | None = None,
    *,
    clip_ratio_c_override: float | None = _SENTINEL,
    unclip_pos_floor_override: bool | None = _SENTINEL,
) -> Tuple[float, dict[str, Any]]:
    """
    Compute the FiberPO clipped policy objective and related metrics.

    J^{FiberPO}(θ|θ_old) =
        E_g[ (1/|J_g|) Σ_j (1/T_j) Σ_i
             fiber_w_j · logclip(corrected_r_{j,i}, ε(Â)) · Â_{j,i} ]

    Args:
        old_log_prob:       (B, T)  log-probs under π_{θ_old}
        log_prob:           (B, T)  log-probs under π_θ
        advantages:         (B, T)  advantage estimates Â_{j,i}
        response_mask:      (B, T)  binary mask for valid tokens
        loss_agg_mode:      str     "token-mean" | "seq-mean"
        config:             optional  (may override eps_pos, eps_neg, delta)
        rollout_is_weights: (B, T)  optional (unused, API compat)
        clip_ratio_c_override:      keyword-only, overrides module default
        unclip_pos_floor_override:  keyword-only, overrides module default

    Returns:
        loss:    float, negated for gradient ascent
        metrics: dict of diagnostic scalars
                 includes "fiberpo/per_token_obj" (B,T) array
    """
    # ── Hyperparams ──────────────────────────────────────────────────────
    eps_pos = EPSILON_POS
    eps_neg = EPSILON_NEG
    delta = DELTA
    clip_ratio_c = CLIP_RATIO_C
    unclip_pos_floor = UNCLIP_POS_FLOOR
    if config is not None:
        eps_pos = getattr(config, "epsilon_pos", eps_pos)
        eps_neg = getattr(config, "epsilon_neg", eps_neg)
        delta = getattr(config, "delta", delta)
        clip_ratio_c = getattr(config, "clip_ratio_c", clip_ratio_c)
        unclip_pos_floor = getattr(config, "unclip_pos_floor", unclip_pos_floor)
    # keyword overrides take highest priority (used by plot scripts)
    if clip_ratio_c_override is not _SENTINEL:
        clip_ratio_c = clip_ratio_c_override
    if unclip_pos_floor_override is not _SENTINEL:
        unclip_pos_floor = unclip_pos_floor_override

    # ── Step 1: log r_{j,i}  (B, T) ─────────────────────────────────────
    log_ratio = compute_log_ratio(log_prob, old_log_prob)

    # ── Step 2: trajectory statistics  (B,) each ─────────────────────────
    log_s_pos, log_s_neg = compute_trajectory_log_s(log_ratio, response_mask)

    # ── Step 3: sequence lengths T_j  (B,) ───────────────────────────────
    seq_lens = np.maximum(response_mask.sum(axis=-1), 1)

    # ── Step 4: fiber weight  (B,) ───────────────────────────────────────
    fiber_w = compute_fiber_weight(log_s_pos, log_s_neg, delta, seq_lens)

    # ── Step 5: corrected log ratio  (B, T) ──────────────────────────────
    corrected_log_r = compute_corrected_log_ratio(
        log_ratio, log_s_pos, log_s_neg,
        response_mask=response_mask
    )

    # ── Step 5b: cross-sign denominator log  (B, T) ──────────────────────
    denom_log_r = compute_cross_sign_log(log_ratio, log_s_pos, log_s_neg)

    # ── Step 6: per-element epsilon  (B, T) ──────────────────────────────
    epsilon = compute_per_element_epsilon(advantages, eps_pos, eps_neg)

    # ── Step 7: logclip  (B, T) ──────────────────────────────────────────
    if clip_ratio_c is not None:
        # Â<0: remove upper clip bound (let ratio grow for punishment),
        #       keep lower clip bound (don't change small-ratio behavior).
        # Â≥0: native symmetric logclip.
        clip_hi = np.where(advantages < 0, np.inf, epsilon)
        clipped_ratio = np.exp(np.clip(corrected_log_r, -epsilon, clip_hi)) \
                      / np.exp(np.clip(denom_log_r, -epsilon, clip_hi))
    else:
        clipped_ratio = logclip(corrected_log_r, epsilon) \
                      / logclip(denom_log_r, epsilon)

    # ── Step 8: per-token surrogate objective  (B, T) ────────────────────
    fiber_w_2d = fiber_w[:, None]  # (B, 1) → broadcasts to (B, T)
    per_token_obj = fiber_w_2d * clipped_ratio * advantages

    # ── Step 8b: dual-clip floor when Â < 0 ──────────────────────────────
    #   log r ≤ 0 (suppressing bad token): native FiberPO ceiling.
    #   log r > 0 (amplifying bad token):  raw ratio × Â, floored at c·Â.
    #   Bypasses both logclip AND fiber weight for (A<0, log_r>0).
    if clip_ratio_c is not None:
        raw_ratio = np.exp(log_ratio)
        raw_obj = raw_ratio * advantages
        floored_obj = np.maximum(raw_obj, clip_ratio_c * advantages)
        per_token_obj = np.where(
            (advantages < 0) & (log_ratio > 0),
            floored_obj,
            per_token_obj
        )

    # ── Step 8c: unclip A>0 floor when log_r < 0 ──────────────────────
    #   Bypasses both logclip AND fiber weight — uses raw r * A.
    #   When policy abandons a good token (A>0, r<1), let the full gradient
    #   push recovery. No ceiling needed: raw_obj ∈ (0, A), naturally bounded.
    #   Mirror of Step 8b for the opposite quadrant.
    if unclip_pos_floor:
        if clip_ratio_c is None:
            raw_ratio = np.exp(log_ratio)
            raw_obj = raw_ratio * advantages
        # raw_ratio and raw_obj already computed in 8b if clip_ratio_c is set
        per_token_obj = np.where(
            (advantages > 0) & (log_ratio < 0),
            raw_obj,
            per_token_obj,
        )

    # ── Step 9: aggregate ────────────────────────────────────────────────
    loss = -aggregate_loss(per_token_obj, response_mask, mode=loss_agg_mode)

    # ── Metrics ──────────────────────────────────────────────────────────
    ratio = np.exp(log_ratio)
    mask_sum = max(response_mask.sum(), 1)

    # g_agg zone fracs (identity / decay / clamped) over batch
    _C = delta / 2.0
    _k_safe = np.maximum(seq_lens, 1.0)
    _upper = (1.0 + 1.0 / _k_safe) * _C  # (B,)
    B = log_s_pos.shape[0]
    gagg_pos_identity_frac = float((log_s_pos <= _C).sum() / B)
    gagg_pos_decay_frac = float(((log_s_pos > _C) & (log_s_pos < _upper)).sum() / B)
    gagg_pos_clamped_frac = float((log_s_pos >= _upper).sum() / B)
    gagg_neg_identity_frac = float((log_s_neg <= _C).sum() / B)
    gagg_neg_decay_frac = float(((log_s_neg > _C) & (log_s_neg < _upper)).sum() / B)
    gagg_neg_clamped_frac = float((log_s_neg >= _upper).sum() / B)

    # quadrant token fracs
    _apos_lrpos = ((advantages > 0) & (log_ratio > 0)).astype(float) * response_mask
    _apos_lrneg = ((advantages > 0) & (log_ratio < 0)).astype(float) * response_mask
    _aneg_lrpos = ((advantages < 0) & (log_ratio > 0)).astype(float) * response_mask
    _aneg_lrneg = ((advantages < 0) & (log_ratio < 0)).astype(float) * response_mask

    # unclip activation frac
    unclip_frac = float(_apos_lrneg.sum() / mask_sum) if unclip_pos_floor else 0.0

    metrics = {
        "fiberpo/loss": loss,
        "fiberpo/ratio_mean": float((ratio * response_mask).sum() / mask_sum),
        "fiberpo/ratio_max": float(
            (ratio * response_mask - (1 - response_mask) * 1e9).max()
        ),
        "fiberpo/log_s_pos_mean": float(log_s_pos.mean()),
        "fiberpo/log_s_neg_mean": float(log_s_neg.mean()),
        "fiberpo/fiber_weight_mean": float(fiber_w.mean()),
        "fiberpo/fiber_weight_std": float(fiber_w.std()),
        "fiberpo/corrected_log_r_abs_mean": float(
            (np.abs(corrected_log_r) * response_mask).sum() / mask_sum
        ),
        "fiberpo/clipped_frac": float(
            ((np.abs(corrected_log_r) > epsilon).astype(float) * response_mask).sum()
            / mask_sum
        ),
        "fiberpo/dual_clip_frac": float(
            (((advantages < 0) & (log_ratio > 0) & (clip_ratio_c * advantages > np.exp(log_ratio) * advantages)).astype(float)
             * response_mask).sum() / mask_sum
        ) if clip_ratio_c is not None else 0.0,
        # g_agg zone distribution (fraction of trajectories in each zone)
        "fiberpo/gagg_pos_identity_frac": gagg_pos_identity_frac,
        "fiberpo/gagg_pos_decay_frac": gagg_pos_decay_frac,
        "fiberpo/gagg_pos_clamped_frac": gagg_pos_clamped_frac,
        "fiberpo/gagg_neg_identity_frac": gagg_neg_identity_frac,
        "fiberpo/gagg_neg_decay_frac": gagg_neg_decay_frac,
        "fiberpo/gagg_neg_clamped_frac": gagg_neg_clamped_frac,
        # quadrant token fracs
        "fiberpo/quad_apos_lrpos_frac": float(_apos_lrpos.sum() / mask_sum),
        "fiberpo/quad_apos_lrneg_frac": float(_apos_lrneg.sum() / mask_sum),
        "fiberpo/quad_aneg_lrpos_frac": float(_aneg_lrpos.sum() / mask_sum),
        "fiberpo/quad_aneg_lrneg_frac": float(_aneg_lrneg.sum() / mask_sum),
        # bypass activation fracs
        "fiberpo/unclip_pos_frac": unclip_frac,
        "fiberpo/per_token_obj": per_token_obj,  # (B, T) array for plotting
    }

    return loss, metrics


# ============================================================================
# Quick sanity check
# ============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    B, T = 4, 16
    old_lp = rng.standard_normal((B, T))
    new_lp = old_lp + 0.05 * rng.standard_normal((B, T))
    adv = rng.standard_normal((B, T))
    mask = np.ones((B, T))
    mask[:, -3:] = 0  # simulate padding

    loss, info = compute_policy_loss_vanilla(old_lp, new_lp, adv, mask)
    print(f"loss = {loss:.6f}")
    for k, v in info.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: ndarray{v.shape}")
        else:
            print(f"  {k}: {v:.6f}")
