# forensiczip_hf.py
 

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F


@dataclass
class ForensicZipArgs:
    retention: float = 192.0  # if <=1.0, interpreted as ratio; else absolute count
    select_layer: int = -2
    keep_cls: bool = False

    # OT / scoring knobs
    birth_cost: float = 0.35
    death_cost: float = 0.35
    sinkhorn_eps: float = 0.1
    sinkhorn_iters: int = 20
    ema_beta: float = 0.6
    birth_weight: float = 0.75
    pos_lambda: float = 0.0

    # Optional forensic HF reweight (multiplies raw OT score)
    forensic_eta: float = 0.0

    eps: float = 1e-6


def _find_first_attr(obj: Any, names: list[str]) -> Any:
    if obj is None:
        return None
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def _resolve_select_layer(model: Any, select_layer: Optional[int]) -> int:
    if select_layer is not None:
        return int(select_layer)
    cfg = getattr(model, "config", None)
    if cfg is None:
        return -2
    if hasattr(cfg, "vision_feature_layer"):
        return int(cfg.vision_feature_layer)
    for key in ("vision_feature_select_layer", "mm_vision_select_layer"):
        if hasattr(cfg, key):
            return int(getattr(cfg, key))
    return -2


def _resolve_retention(retention: float, num_patches: int) -> int:
    if retention <= 0:
        return max(1, min(num_patches, 1))
    if retention <= 1.0:
        keep = int(round(retention * num_patches))
    else:
        keep = int(round(retention))
    keep = max(1, keep)
    keep = min(num_patches, keep)
    return keep


def _minmax_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = torch.nan_to_num(x, nan=0.0, neginf=0.0, posinf=0.0)
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def _infer_square_side(n: int) -> Optional[int]:
    s = int(math.isqrt(max(0, int(n))))
    return s if s * s == int(n) else None


@torch.no_grad()
def _robust_zscore(scores: torch.Tensor, eps: float) -> torch.Tensor:
    # scores: [T, N]
    med = scores.median(dim=-1, keepdim=True).values
    mad = (scores - med).abs().median(dim=-1, keepdim=True).values
    return (scores - med) / (mad + eps)


@torch.no_grad()
def _ema_smooth(scores: torch.Tensor, beta: float) -> torch.Tensor:
    # scores: [T, N]
    if scores.shape[0] <= 1:
        return scores
    out = torch.empty_like(scores)
    out[0] = scores[0]
    for t in range(1, scores.shape[0]):
        out[t] = (1.0 - beta) * out[t - 1] + beta * scores[t]
    return out


@torch.no_grad()
def _sinkhorn_balanced_log(cost: torch.Tensor, eps: float, iters: int) -> torch.Tensor:
    """
    Balanced entropic OT via Sinkhorn in log-domain.
    cost: [N, N] non-negative (float32)
    returns transport plan P: [N, N] (float32)
    """
    n = int(cost.shape[0])
    if n <= 0:
        return cost.new_zeros(cost.shape)

    eps = float(max(eps, 1e-6))
    logK = -cost / eps  # [n,n]

    loga = -math.log(n)
    logb = -math.log(n)
    logu = cost.new_zeros((n,))
    logv = cost.new_zeros((n,))

    for _ in range(int(max(1, iters))):
        logu = loga - torch.logsumexp(logK + logv.unsqueeze(0), dim=1)
        logv = logb - torch.logsumexp(logK.transpose(0, 1) + logu.unsqueeze(0), dim=1)

    logP = logu.unsqueeze(1) + logK + logv.unsqueeze(0)
    return torch.exp(logP)


@torch.no_grad()
def _forensiczip_scores_for_group(
    feats: torch.Tensor,  # [T, N, D] (projected patch tokens)
    images: Optional[torch.Tensor],  # [T, 3, H, W] or None
    birth_cost: float,
    death_cost: float,
    sinkhorn_eps: float,
    sinkhorn_iters: int,
    ema_beta: float,
    birth_weight: float,
    forensic_eta: float,
    pos_lambda: float,
    eps: float,
) -> torch.Tensor:
    """
    Compute FORENSICZIP novelty scores for a contiguous group of frames.
    Returns: [T, N] float32 (higher => keep)
    """
    T, N, D = feats.shape
    device = feats.device

    x = F.normalize(feats.float(), dim=-1, eps=eps)  # [T,N,D]

    # Optional: append a weak position channel to reduce degenerate matches on repeated textures.
    if pos_lambda > 0.0:
        side = _infer_square_side(N)
        if side is not None:
            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, side, device=device, dtype=torch.float32),
                torch.linspace(-1.0, 1.0, side, device=device, dtype=torch.float32),
                indexing="ij",
            )
            pos = torch.stack([yy, xx], dim=-1).view(-1, 2)  # [N,2]
            pos = pos.unsqueeze(0).expand(T, -1, -1)  # [T,N,2]
            x = torch.cat([x, (float(pos_lambda) ** 0.5) * pos], dim=-1)
            x = F.normalize(x, dim=-1, eps=eps)

    scores = torch.zeros((T, N), device=device, dtype=torch.float32)
    if T <= 1:
        # Fallback: outlierness wrt mean (works for still images).
        mu = x.mean(dim=1, keepdim=True)
        out = torch.sqrt(torch.sum(torch.square(x - mu), dim=-1) + eps)  # [1,N]
        return _minmax_norm(out, eps=eps).to(dtype=torch.float32)

    n_aug = N + 1
    cost = torch.empty((n_aug, n_aug), device=device, dtype=torch.float32)
    cost[n_aug - 1, n_aug - 1] = 0.0

    for t in range(1, T):
        prev = x[t - 1]  # [N,D']
        cur = x[t]       # [N,D']
        cost_nn = (1.0 - torch.matmul(prev, cur.transpose(0, 1))).clamp(0.0, 2.0)  # [N,N]
        cost[:N, :N] = cost_nn
        cost[N, :N] = float(birth_cost)
        cost[:N, N] = float(death_cost)
        cost[N, N] = 0.0

        P = _sinkhorn_balanced_log(cost, eps=sinkhorn_eps, iters=sinkhorn_iters)  # [N+1,N+1]

        denom = P[:, :N].sum(dim=0).clamp_min(eps)  # [N]
        exp_cost = (P[:, :N] * cost[:, :N]).sum(dim=0) / denom
        birth_frac = P[N, :N] / denom
        scores[t] = exp_cost + float(birth_weight) * birth_frac

    scores[0] = scores[1]

    # Optional forensic HF reweight (pixel-domain Laplacian pooled to patch grid).
    if forensic_eta > 0.0 and images is not None and images.dim() == 4 and images.size(0) == T:
        side = _infer_square_side(N)
        if side is not None:
            img = images.detach()
            if img.dtype != torch.float32:
                img = img.float()
            y = img.mean(dim=1, keepdim=True)
            k = torch.tensor(
                [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
                device=device,
                dtype=torch.float32,
            ).view(1, 1, 3, 3)
            hp = F.conv2d(y, k, padding=1).abs()  # [T,1,H,W]
            pooled = F.adaptive_avg_pool2d(hp, output_size=(side, side)).flatten(1)  # [T,N]
            pooled = _minmax_norm(pooled, eps=eps)
            scores = scores * (1.0 + float(forensic_eta) * pooled)

    # Robust normalization + temporal smoothing (module-2)
    scores = _robust_zscore(scores, eps=eps)
    scores = _ema_smooth(scores, beta=float(max(0.0, min(1.0, ema_beta))))
    return scores


def enable_forensiczip_for_llava(
    model: Any,
    retention: float = 192.0,
    select_layer: Optional[int] = None,
    keep_cls: bool = False,
    birth_cost: float = 0.35,
    death_cost: float = 0.35,
    sinkhorn_eps: float = 0.1,
    sinkhorn_iters: int = 20,
    ema_beta: float = 0.6,
    birth_weight: float = 0.75,
    pos_lambda: float = 0.0,
    forensic_eta: float = 0.0,
    verbose: bool = True,
) -> Any:
    """
    Patch a HuggingFace LLaVA-style model so that:
      - vision_tower.forward returns hidden_states with pruned patch tokens (FORENSICZIP).
      - model.config.image_seq_length updated to pruned length (best-effort).
      - stats recorded for verification.
    """
    lt = ForensicZipArgs(
        retention=float(retention),
        select_layer=_resolve_select_layer(model, select_layer),
        keep_cls=bool(keep_cls),
        birth_cost=float(birth_cost),
        death_cost=float(death_cost),
        sinkhorn_eps=float(sinkhorn_eps),
        sinkhorn_iters=int(sinkhorn_iters),
        ema_beta=float(ema_beta),
        birth_weight=float(birth_weight),
        pos_lambda=float(pos_lambda),
        forensic_eta=float(forensic_eta),
    )

    vision_tower = _find_first_attr(model, ["vision_tower"])
    if vision_tower is None and hasattr(model, "model"):
        vision_tower = _find_first_attr(model.model, ["vision_tower"])
    if vision_tower is None and hasattr(model, "get_vision_tower"):
        try:
            vision_tower = model.get_vision_tower()
        except Exception:
            vision_tower = None

    projector = _find_first_attr(model, ["multi_modal_projector", "mm_projector", "vision_projector"])
    if projector is None and hasattr(model, "model"):
        projector = _find_first_attr(model.model, ["multi_modal_projector", "mm_projector", "vision_projector"])

    if vision_tower is None:
        raise RuntimeError("[FORENSICZIP] Cannot find vision tower on model.")
    if projector is None:
        raise RuntimeError("[FORENSICZIP] Cannot find multimodal projector on model.")

    if not hasattr(model, "_forensiczip_stats"):
        model._forensiczip_stats = {"vision_tower_calls": 0, "prune_calls": 0}

    if hasattr(vision_tower, "forward") and not getattr(vision_tower.forward, "_forensiczip_wrapped", False):
        orig_vt_forward = vision_tower.forward

        def vt_forward_forensiczip(self, pixel_values: torch.Tensor, *args, **kwargs):
            kwargs["output_hidden_states"] = True
            outputs = orig_vt_forward(pixel_values, *args, **kwargs)

            model._forensiczip_stats["vision_tower_calls"] = model._forensiczip_stats.get("vision_tower_calls", 0) + 1

            hiddens = getattr(outputs, "hidden_states", None)
            if hiddens is None:
                return outputs

            num_layers = len(hiddens)
            sel = lt.select_layer if lt.select_layer >= 0 else (num_layers + lt.select_layer)
            if sel < 0 or sel >= num_layers:
                return outputs

            hidden_states = hiddens[sel]  # [B, S, D]
            if hidden_states is None or hidden_states.dim() != 3 or hidden_states.shape[1] <= 1:
                return outputs

            cls_tok = hidden_states[:, :1, :]
            patch_hidden = hidden_states[:, 1:, :]  # [B, N, Dv]

            bsz, num_patches, _ = patch_hidden.shape
            keep = _resolve_retention(lt.retention, num_patches)
            model._forensiczip_last_num_patches = int(num_patches)
            model._forensiczip_last_keep = int(keep)

            if keep >= num_patches:
                return outputs

            # Project to LLM space for OT scoring.
            patch_proj = projector(patch_hidden)  # [B, N, D]

            group_sizes = getattr(model, "_forensiczip_group_sizes", None)
            if group_sizes is None:
                # Default: treat each element independently (no temporal OT).
                group_sizes = [1] * int(bsz)
            if not isinstance(group_sizes, (list, tuple)) or sum(int(x) for x in group_sizes) != int(bsz):
                group_sizes = [1] * int(bsz)

            # Ensure pixel_values aligns (best-effort): only use HF reweight if pixel_values is [B,3,H,W].
            pv = pixel_values if isinstance(pixel_values, torch.Tensor) else None
            if pv is not None and pv.dim() != 4:
                pv = None

            select_list = []
            start = 0
            for L in group_sizes:
                L = int(L)
                if L <= 0:
                    continue
                end = start + L
                feats_g = patch_proj[start:end]  # [L,N,D]
                img_g = pv[start:end] if pv is not None else None
                scores = _forensiczip_scores_for_group(
                    feats=feats_g,
                    images=img_g,
                    birth_cost=lt.birth_cost,
                    death_cost=lt.death_cost,
                    sinkhorn_eps=lt.sinkhorn_eps,
                    sinkhorn_iters=lt.sinkhorn_iters,
                    ema_beta=lt.ema_beta,
                    birth_weight=lt.birth_weight,
                    forensic_eta=lt.forensic_eta,
                    pos_lambda=lt.pos_lambda,
                    eps=lt.eps,
                )  # [L,N]
                idx = torch.topk(scores, k=keep, dim=-1, largest=True, sorted=True).indices  # [L,keep]
                select_list.append(idx)
                start = end

            if not select_list:
                return outputs

            select_idx = torch.cat(select_list, dim=0)  # [B, keep]
            kept = torch.gather(
                patch_hidden,
                dim=1,
                index=select_idx.unsqueeze(-1).expand(-1, -1, patch_hidden.shape[-1]),
            )

            model._forensiczip_last_seq_len = int(kept.shape[1])
            model._forensiczip_stats["prune_calls"] = model._forensiczip_stats.get("prune_calls", 0) + 1

            new_hidden = torch.cat([cls_tok, kept], dim=1)
            hs_list = list(hiddens)
            hs_list[sel] = new_hidden
            outputs.hidden_states = tuple(hs_list)

            if hasattr(model, "config") and hasattr(model.config, "image_seq_length"):
                model.config.image_seq_length = int(keep)

            return outputs

        vt_forward_forensiczip._forensiczip_wrapped = True
        vision_tower.forward = types.MethodType(vt_forward_forensiczip, vision_tower)

    if hasattr(model, "config"):
        if hasattr(model.config, "vision_feature_layer"):
            model.config.vision_feature_layer = lt.select_layer
        if hasattr(model.config, "image_seq_length"):
            if retention <= 1.0:
                base_len = getattr(model.config, "image_seq_length", None)
                if base_len is not None:
                    model.config.image_seq_length = _resolve_retention(retention, int(base_len))
            else:
                model.config.image_seq_length = int(round(retention))

        model.config.forensiczip_resize_input_ids = True
        model.config.forensiczip_enabled = True
        model.config.forensiczip_retention = float(retention)
        model.config.forensiczip_select_layer = int(lt.select_layer)

    model._forensiczip_enabled = True
    model._forensiczip_args = lt

    if verbose:
        print(
            "[FORENSICZIP] Enabled on model: "
            f"retention={retention}, select_layer={lt.select_layer}, "
            f"birth_cost={lt.birth_cost}, death_cost={lt.death_cost}, "
            f"sinkhorn_eps={lt.sinkhorn_eps}, iters={lt.sinkhorn_iters}, "
            f"ema_beta={lt.ema_beta}, forensic_eta={lt.forensic_eta}"
        )

    return model


def assert_forensiczip_really_used(model: Any) -> None:
    if not getattr(model, "_forensiczip_enabled", False):
        raise RuntimeError("[FORENSICZIP] model is not enabled.")
    stats = getattr(model, "_forensiczip_stats", {})
    vt_calls = int(stats.get("vision_tower_calls", 0))
    prune_calls = int(stats.get("prune_calls", 0))
    if vt_calls <= 0 or prune_calls <= 0:
        raise RuntimeError("[FORENSICZIP] vision_tower or prune never called.")
    last_len = getattr(model, "_forensiczip_last_seq_len", None)
    last_num = getattr(model, "_forensiczip_last_num_patches", None)
    if last_len is None or last_num is None:
        raise RuntimeError("[FORENSICZIP] missing recorded seq length.")
    args = getattr(model, "_forensiczip_args", None)
    if args is not None:
        retention = float(getattr(args, "retention", 0))
        if (retention < 1.0) or (retention > 1.0 and retention < last_num):
            if last_len >= last_num:
                raise RuntimeError(f"[FORENSICZIP] pruning ineffective: last_len={last_len} >= original={last_num}")

