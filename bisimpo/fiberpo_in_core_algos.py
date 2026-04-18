# ============================================================================
# FiberPO helpers
# ============================================================================

def _fiberpo_g_agg(
    log_s: torch.Tensor,
    C: float,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    Piecewise linear aggregation with soft clamping (rollback).

    g^agg(x, C, k) =
        x                              if |x| <= C           (identity)
        sign(x)*(k+1)*C - k*x         if C < |x| < (1+1/k)*C  (decay)
        0                              otherwise             (clamped)

    Args:
        log_s: (B,) input values
        C:     scalar threshold (= delta/2)
        k:     (B,) decay rate (typically seq_lens T_j)

    Returns:
        out: (B,) same shape as log_s
    """
    abs_log_s = torch.abs(log_s)
    sign_log_s = torch.sign(log_s)

    k_safe = torch.clamp(k.float(), min=1.0)
    upper_bound = (1.0 + 1.0 / k_safe) * C  # (1 + 1/k) * C

    identity_mask = abs_log_s <= C
    decay_mask = (abs_log_s > C) & (abs_log_s < upper_bound)

    identity_val = log_s
    decay_val = sign_log_s * (k_safe + 1) * C - k_safe * log_s

    out = torch.zeros_like(log_s)
    out = torch.where(identity_mask, identity_val, out)
    out = torch.where(decay_mask, decay_val, out)

    return out


@register_policy_loss("fiberpo")
def compute_policy_loss_fiberpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute FiberPO (Fiber-corrected Proximal Optimization) loss.

    FiberPO decomposes the importance sampling ratio into trajectory-level
    fiber corrections and per-token residuals with piecewise-linear rollback:

    L_{j,i} = fiber_w_j * logclip(corrected_r_{j,i}, eps) / logclip(denom_{j,i}, eps) * A_{j,i}

    where:
    - fiber_w_j = exp(g_agg(log_s+_j, C, T_j) - g_agg(log_s-_j, C, T_j))
    - corrected_log_r = log_r - sign(log_r) * log_s^{sign(log_r)}
    - denom_log = -sign(log_r) * log_s^{-sign(log_r)}
    - g_agg provides piecewise linear rollback beyond threshold C = delta/2

    Optional dual-clip: when clip_ratio_c is set and A < 0, log_r > 0,
    floors the objective at c * A using the raw ratio.

    Args:
        old_log_prob: Log-prob under old policy, shape (batch_size, response_length)
        log_prob: Log-prob under current policy, shape (batch_size, response_length)
        advantages: Advantage estimates, shape (batch_size, response_length)
        response_mask: Valid token mask, shape (batch_size, response_length)
        loss_agg_mode: Loss aggregation mode
        config: ActorConfig with FiberPO hyperparameters
        rollout_is_weights: Optional IS correction weights

    Returns:
        pg_loss: Scalar policy gradient loss
        pg_metrics: Dictionary of metrics
    """
    assert config is not None
    assert isinstance(config, ActorConfig)

    # ── Hyperparams ──────────────────────────────────────────────────────
    eps_pos = config.get("fiberpo_epsilon_pos", 0.2)
    eps_neg = config.get("fiberpo_epsilon_neg", 0.2)
    delta = config.get("fiberpo_delta", 0.4)
    C = config.get("fiberpo_C", None)
    if C is None:
        C = delta / 2.0
    clip_ratio_c = config.get("fiberpo_clip_ratio_c", None)
    unclip_pos_floor = config.get("fiberpo_unclip_pos_floor", False)

    # ── Step 1: log ratio (B, T) ─────────────────────────────────────────
    log_ratio = log_prob - old_log_prob
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)

    # ── Step 2: trajectory statistics (B,) each ──────────────────────────
    seq_lens = response_mask.sum(dim=-1).clamp(min=1)  # (B,)
    log_s_pos = (torch.clamp(log_ratio, min=0.0) * response_mask).sum(dim=-1) / seq_lens  # (B,)
    log_s_neg = (torch.clamp(-log_ratio, min=0.0) * response_mask).sum(dim=-1) / seq_lens  # (B,)

    # ── Step 3: fiber weight (B,) ────────────────────────────────────────
    # w = exp(alpha * (g_agg(P,C,T) - g_agg(N,C,T))), alpha = delta/(2C)
    # C controls rollback threshold, delta controls peak weight exp(delta/2)
    # When C = delta/2 (default): alpha = 1, identical to NumPy reference
    g_pos = _fiberpo_g_agg(log_s_pos, C, seq_lens)
    g_neg = _fiberpo_g_agg(log_s_neg, C, seq_lens)
    alpha = delta / (2.0 * C) if C > 0 else 0.0
    fiber_w = torch.exp(alpha * (g_pos - g_neg))  # (B,)

    # ── Step 4: corrected log ratio — numerator (B, T) ───────────────────
    sign_lr = torch.sign(log_ratio)  # (B, T)
    lsp = log_s_pos.unsqueeze(-1)    # (B, 1)
    lsn = log_s_neg.unsqueeze(-1)    # (B, 1)
    # select same-sign statistic: s+ when sign=+1, s- when sign=-1, 0 when sign=0
    log_s_same = torch.where(sign_lr > 0, lsp,
                 torch.where(sign_lr < 0, lsn,
                             torch.zeros_like(log_ratio)))  # (B, T)
    corrected_log_r = log_ratio - sign_lr * log_s_same      # (B, T)

    # ── Step 5: cross-sign denominator log (B, T) ────────────────────────
    log_s_opp = torch.where(sign_lr > 0, lsn,
                torch.where(sign_lr < 0, lsp,
                            torch.zeros_like(log_ratio)))   # (B, T)
    denom_log_r = -sign_lr * log_s_opp                      # (B, T)

    # ── Step 6: per-element epsilon (B, T) ───────────────────────────────
    epsilon = torch.where(advantages > 0, eps_pos, eps_neg)

    # ── Step 7: logclip ratio (B, T) ────────────────────────────────────
    if clip_ratio_c is not None:
        # A<0: remove upper clip bound (let ratio grow for punishment)
        clip_hi = torch.where(advantages < 0,
                              torch.tensor(20.0, device=log_ratio.device, dtype=log_ratio.dtype),
                              epsilon)
        clipped_numer = torch.exp(torch.clamp(corrected_log_r, -epsilon, clip_hi))
        clipped_denom = torch.exp(torch.clamp(denom_log_r, -epsilon, clip_hi))
    else:
        clipped_numer = torch.exp(torch.clamp(corrected_log_r, -epsilon, epsilon))
        clipped_denom = torch.exp(torch.clamp(denom_log_r, -epsilon, epsilon))

    clipped_ratio = clipped_numer / clipped_denom.clamp(min=1e-10)

    # ── Step 8: per-token surrogate objective (B, T) ─────────────────────
    fiber_w_2d = fiber_w.unsqueeze(-1)  # (B, 1)
    per_token_obj = fiber_w_2d * clipped_ratio * advantages

    # ── Step 8b: dual-clip floor when A < 0 and log_r > 0 ───────────────
    #   Bypasses both logclip AND fiber weight — uses raw r * A, floored at c * A.
    #   Mirror: clip_ratio_c governs A<0 punishment cap.
    if clip_ratio_c is not None:
        raw_ratio = torch.exp(log_ratio)
        raw_obj = raw_ratio * advantages
        floored_obj = torch.max(raw_obj, clip_ratio_c * advantages)
        per_token_obj = torch.where(
            (advantages < 0) & (log_ratio > 0),
            floored_obj,
            per_token_obj,
        )

    # ── Step 8c: unclip A>0 floor when log_r < 0 ────────────────────────
    #   Bypasses both logclip AND fiber weight — uses raw r * A.
    #   When policy abandons a good token (A>0, r<1), let the full gradient
    #   push recovery. No ceiling needed: raw_obj ∈ (0, A), naturally bounded.
    #   Mirror of Step 8b for the opposite quadrant.
    if unclip_pos_floor:
        if clip_ratio_c is None:
            raw_ratio = torch.exp(log_ratio)
            raw_obj = raw_ratio * advantages
        # raw_ratio and raw_obj already computed in 8b if clip_ratio_c is set
        per_token_obj = torch.where(
            (advantages > 0) & (log_ratio < 0),
            raw_obj,
            per_token_obj,
        )

    # ── Step 9: negate for loss minimization ─────────────────────────────
    pg_losses = -per_token_obj

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # Aggregate loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info
    )

    # ── Metrics ──────────────────────────────────────────────────────────
    ratio = torch.exp(log_ratio)
    negative_approx_kl = log_ratio
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    mask_sum = response_mask.sum().clamp(min=1)

    pg_clipfrac = (
        (torch.abs(corrected_log_r) > epsilon).float() * response_mask
    ).sum() / mask_sum

    dual_clip_frac = torch.tensor(0.0)
    if clip_ratio_c is not None:
        dual_clip_frac = (
            ((advantages < 0) & (log_ratio > 0) &
             (clip_ratio_c * advantages > raw_ratio * advantages)).float()
            * response_mask
        ).sum() / mask_sum

    # ── g_agg zone fracs (identity / decay / clamped) over batch ────────
    #   identity: |x| ≤ C,  decay: C < |x| < (1+1/k)C,  clamped: |x| ≥ (1+1/k)C
    _k_safe = seq_lens.clamp(min=1).float()
    _upper = (1.0 + 1.0 / _k_safe) * C  # (B,)  per-trajectory upper bound
    gagg_pos_identity_frac = (log_s_pos <= C).float().mean()
    gagg_pos_decay_frac = ((log_s_pos > C) & (log_s_pos < _upper)).float().mean()
    gagg_pos_clamped_frac = (log_s_pos >= _upper).float().mean()
    gagg_neg_identity_frac = (log_s_neg <= C).float().mean()
    gagg_neg_decay_frac = ((log_s_neg > C) & (log_s_neg < _upper)).float().mean()
    gagg_neg_clamped_frac = (log_s_neg >= _upper).float().mean()

    # ── Quadrant token fracs ────────────────────────────────────────────
    _apos_lrpos = ((advantages > 0) & (log_ratio > 0)).float() * response_mask
    _apos_lrneg = ((advantages > 0) & (log_ratio < 0)).float() * response_mask
    _aneg_lrpos = ((advantages < 0) & (log_ratio > 0)).float() * response_mask
    _aneg_lrneg = ((advantages < 0) & (log_ratio < 0)).float() * response_mask

    # ── unclip activation frac ──────────────────────────────────────────
    unclip_frac = torch.tensor(0.0)
    if unclip_pos_floor:
        unclip_frac = (_apos_lrneg.sum() / mask_sum)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": torch.tensor(0.0).item(),
        "actor/fiberpo_fiber_weight_mean": fiber_w.mean().detach().item(),
        "actor/fiberpo_fiber_weight_std": fiber_w.std().detach().item(),
        "actor/fiberpo_log_s_pos_mean": log_s_pos.mean().detach().item(),
        "actor/fiberpo_log_s_neg_mean": log_s_neg.mean().detach().item(),
        "actor/fiberpo_corrected_log_r_abs_mean": (
            (torch.abs(corrected_log_r) * response_mask).sum() / mask_sum
        ).detach().item(),
        "actor/fiberpo_ratio_mean": (
            (ratio * response_mask).sum() / mask_sum
        ).detach().item(),
        "actor/fiberpo_dual_clip_frac": dual_clip_frac.detach().item(),
        # g_agg zone distribution (fraction of trajectories in each zone)
        "actor/fiberpo_gagg_pos_identity_frac": gagg_pos_identity_frac.detach().item(),
        "actor/fiberpo_gagg_pos_decay_frac": gagg_pos_decay_frac.detach().item(),
        "actor/fiberpo_gagg_pos_clamped_frac": gagg_pos_clamped_frac.detach().item(),
        "actor/fiberpo_gagg_neg_identity_frac": gagg_neg_identity_frac.detach().item(),
        "actor/fiberpo_gagg_neg_decay_frac": gagg_neg_decay_frac.detach().item(),
        "actor/fiberpo_gagg_neg_clamped_frac": gagg_neg_clamped_frac.detach().item(),
        # quadrant token fracs (fraction of valid tokens in each quadrant)
        "actor/fiberpo_quad_apos_lrpos_frac": (_apos_lrpos.sum() / mask_sum).detach().item(),
        "actor/fiberpo_quad_apos_lrneg_frac": (_apos_lrneg.sum() / mask_sum).detach().item(),
        "actor/fiberpo_quad_aneg_lrpos_frac": (_aneg_lrpos.sum() / mask_sum).detach().item(),
        "actor/fiberpo_quad_aneg_lrneg_frac": (_aneg_lrneg.sum() / mask_sum).detach().item(),
        # bypass activation frac
        "actor/fiberpo_unclip_pos_frac": unclip_frac.detach().item(),
    }

    return pg_loss, pg_metrics
