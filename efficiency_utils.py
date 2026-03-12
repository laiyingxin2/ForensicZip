#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

try:
    from sparsevlm.budget import get_budget_keep
except Exception:
    get_budget_keep = None


@dataclass
class ModelProfile:
    num_layers: int
    hidden_size: int
    intermediate_size: int


@dataclass
class KVCacheProfile:
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int


class SeqStats:
    def __init__(self) -> None:
        self.samples = 0
        self.seq_sum = 0
        self.img_sum = 0
        self.img_samples = 0
        self.img_runs_sum = 0
        self.img_runs_samples = 0

    def update(self, input_ids: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor], image_token_id: Optional[int]) -> None:
        if input_ids is None or attention_mask is None:
            return
        if input_ids.dim() != 2 or attention_mask.dim() != 2:
            return
        lengths = attention_mask.sum(dim=1).to(torch.long)
        self.seq_sum += int(lengths.sum().item())
        self.samples += int(lengths.numel())
        if image_token_id is not None:
            img_mask = (input_ids == image_token_id)
            img_counts = img_mask.sum(dim=1).to(torch.long)
            self.img_sum += int(img_counts.sum().item())
            self.img_samples += int(img_counts.numel())
            # Count contiguous image-token runs (≈ number of images for multi-image prompts).
            if img_mask.numel() > 0:
                start0 = img_mask[:, :1]
                transitions = img_mask[:, 1:] & (~img_mask[:, :-1])
                starts = torch.cat([start0, transitions], dim=1)
                run_counts = starts.sum(dim=1).to(torch.long)
                self.img_runs_sum += int(run_counts.sum().item())
                self.img_runs_samples += int(run_counts.numel())

    def averages(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if self.samples <= 0:
            return None, None, None
        avg_seq = self.seq_sum / float(self.samples)
        if self.img_samples > 0:
            avg_img = self.img_sum / float(self.img_samples)
            avg_text = avg_seq - avg_img
        else:
            avg_img = None
            avg_text = None
        return avg_seq, avg_img, avg_text

    def avg_image_runs(self) -> Optional[float]:
        if self.img_runs_samples <= 0:
            return None
        return self.img_runs_sum / float(self.img_runs_samples)


def resolve_output_paths(output_path: str, exp_name: str, run_id: int) -> Tuple[str, str, str, Optional[str]]:
    base_dir = None
    legacy_output = None
    if output_path:
        if output_path.endswith(".json"):
            legacy_output = output_path
            base_dir = os.path.dirname(output_path) or "."
        else:
            base_dir = output_path
    if not base_dir:
        base_dir = os.path.join("results", exp_name)

    responses_dir = os.path.join(base_dir, "responses")
    metrics_dir = os.path.join(base_dir, "metrics")
    os.makedirs(responses_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    resp_path = os.path.join(responses_dir, f"{exp_name}_{run_id}.json")
    metrics_path = os.path.join(metrics_dir, f"{exp_name}_{run_id}.json")
    return base_dir, resp_path, metrics_path, legacy_output


def cuda_time_ms(func):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = func()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return out, elapsed_ms


def count_generated_tokens(
    sequences: Optional[torch.Tensor],
    *,
    input_len: int,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """
    Count generated tokens per sample from HF `generate` output sequences.

    Assumes `sequences` is the concatenation of prompt+generated tokens:
      sequences: [B, T_total]
      generated: sequences[:, input_len:]

    Returns:
      lengths: LongTensor[B] (on CPU), counting up to EOS (inclusive) and/or PAD (exclusive).
    """

    if sequences is None or not isinstance(sequences, torch.Tensor) or sequences.dim() != 2:
        return None

    bsz, total_len = sequences.shape
    if input_len < 0:
        input_len = 0
    if input_len >= total_len:
        return torch.zeros((bsz,), dtype=torch.long)

    gen = sequences[:, input_len:]
    _, gen_len = gen.shape
    lengths = torch.full((bsz,), gen_len, dtype=torch.long, device=gen.device)

    def _first_true_pos(mask: torch.Tensor) -> torch.Tensor:
        # mask: [B, T] bool -> first true index per row, or T if none
        if mask.numel() == 0:
            return torch.full((mask.shape[0],), mask.shape[1], dtype=torch.long, device=mask.device)
        idx = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0).expand(mask.shape[0], -1)
        idx = idx.masked_fill(~mask, mask.shape[1])
        return idx.min(dim=1).values

    if eos_token_id is not None:
        eos_mask = gen == int(eos_token_id)
        if bool(eos_mask.any().item()):
            eos_pos = _first_true_pos(eos_mask)
            eos_len = eos_pos + 1  # include EOS
            lengths = torch.minimum(lengths, eos_len)

    if pad_token_id is not None and pad_token_id != eos_token_id:
        pad_mask = gen == int(pad_token_id)
        if bool(pad_mask.any().item()):
            pad_pos = _first_true_pos(pad_mask)
            lengths = torch.minimum(lengths, pad_pos)  # do not include PAD

    return lengths.to("cpu")


def get_model_profile(model) -> ModelProfile:
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
    if text_cfg is None:
        text_cfg = cfg

    num_layers = int(getattr(text_cfg, "num_hidden_layers", 0) or 0)
    hidden_size = int(getattr(text_cfg, "hidden_size", 0) or 0)
    inter_size = int(getattr(text_cfg, "intermediate_size", 0) or 0)
    return ModelProfile(num_layers=num_layers, hidden_size=hidden_size, intermediate_size=inter_size)


def get_kv_cache_profile(model) -> Optional[KVCacheProfile]:
    """
    Best-effort KV-cache shape inference for decoder-only LMs.

    KV cache per layer is typically:
      K: [B, num_kv_heads, S, head_dim]
      V: [B, num_kv_heads, S, head_dim]
    Total bytes ≈ num_layers * 2 * B * S * num_kv_heads * head_dim * dtype_bytes
    """

    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
    if text_cfg is None:
        text_cfg = cfg
    if text_cfg is None:
        return None

    num_layers = int(getattr(text_cfg, "num_hidden_layers", 0) or 0)
    hidden_size = int(getattr(text_cfg, "hidden_size", 0) or 0)
    num_heads = int(getattr(text_cfg, "num_attention_heads", 0) or 0)
    num_kv_heads = int(getattr(text_cfg, "num_key_value_heads", 0) or 0) or num_heads

    head_dim = int(getattr(text_cfg, "head_dim", 0) or 0)
    if head_dim <= 0 and num_heads > 0 and hidden_size > 0:
        head_dim = hidden_size // num_heads

    dtype = getattr(model, "dtype", None)
    dtype_bytes = 0
    try:
        if isinstance(dtype, torch.dtype):
            dtype_bytes = int(torch.tensor([], dtype=dtype).element_size())
    except Exception:
        dtype_bytes = 0

    if num_layers <= 0 or num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0 or dtype_bytes <= 0:
        return None

    return KVCacheProfile(
        num_layers=num_layers,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
    )


def estimate_kv_cache_bytes(seq_len: float, kv: KVCacheProfile, *, batch_size: int = 1) -> float:
    """
    Rough KV-cache memory estimate (bytes) at sequence length `seq_len` and batch size `batch_size`.
    """

    if seq_len <= 0 or batch_size <= 0:
        return 0.0
    return (
        float(kv.num_layers)
        * 2.0
        * float(batch_size)
        * float(seq_len)
        * float(kv.num_kv_heads)
        * float(kv.head_dim)
        * float(kv.dtype_bytes)
    )


def estimate_layer_flops(seq_len: float, profile: ModelProfile) -> float:
    if seq_len <= 0 or profile.hidden_size <= 0 or profile.intermediate_size <= 0:
        return 0.0
    d = float(profile.hidden_size)
    m = float(profile.intermediate_size)
    n = float(seq_len)
    return 4.0 * n * d * d + 2.0 * n * n * d + 2.0 * n * d * m


def estimate_prefill_flops(seq_len: float, profile: ModelProfile) -> float:
    if profile.num_layers <= 0:
        return 0.0
    return float(profile.num_layers) * estimate_layer_flops(seq_len, profile)


def estimate_prefill_flops_fastv(seq_len_full: float, seq_len_pruned: float, profile: ModelProfile, fastv_k: int) -> float:
    if profile.num_layers <= 0:
        return 0.0
    k = max(0, min(int(fastv_k), int(profile.num_layers)))
    full_layers = k
    pruned_layers = profile.num_layers - full_layers
    return float(full_layers) * estimate_layer_flops(seq_len_full, profile) + float(pruned_layers) * estimate_layer_flops(seq_len_pruned, profile)


def estimate_prefill_flops_sparsevlm(
    text_len: float,
    image_len: float,
    profile: ModelProfile,
    pruning_layers: Sequence[int],
    retain_token: int,
    v2: bool = True,
) -> float:
    if profile.num_layers <= 0:
        return 0.0
    seq_full = text_len + image_len
    layers = sorted([int(l) for l in pruning_layers if int(l) >= 0])
    layers = [l for l in layers if l < profile.num_layers]
    if not layers or get_budget_keep is None:
        return estimate_prefill_flops(seq_full, profile)

    total = 0.0
    prev = 0
    current_img = float(image_len)
    for stage, layer in enumerate(layers):
        seg_layers = layer - prev
        if seg_layers > 0:
            total += float(seg_layers) * estimate_layer_flops(text_len + current_img, profile)
        keep = int(get_budget_keep(retain_token, stage=stage, v2=v2))
        current_img = float(max(1, min(int(image_len), keep)))
        prev = layer

    remaining = profile.num_layers - prev
    if remaining > 0:
        total += float(remaining) * estimate_layer_flops(text_len + current_img, profile)
    return total


def get_param_bytes(model) -> int:
    total = 0
    for p in model.parameters():
        if p.device.type == "cuda":
            total += int(p.numel()) * int(p.element_size())
    return total


def attach_table_metrics(metrics: dict) -> None:
    """
    Attach convenience keys used in table-style reporting.
    Keeps original fields intact and adds:
      - FLOPs_T
      - FLOPs_ratio
      - Latency_ms (prefill + generate per-sample)
      - Prefill_ms
      - Gen_ms_per_sample
    """
    if not isinstance(metrics, dict):
        return
    eff = metrics.get("efficiency") or {}
    flops_t = eff.get("prefill_flops_t")
    metrics.setdefault("FLOPs_T", flops_t)
    metrics.setdefault("FLOPs_ratio", eff.get("prefill_flops_ratio"))
    prefill_ms = eff.get("avg_prefill_ms")
    prefill_ms_per_sample = eff.get("avg_prefill_ms_per_sample")
    gen_ms = eff.get("avg_generate_ms_per_sample")
    latency_ms = None
    if prefill_ms is not None and gen_ms is not None:
        latency_ms = prefill_ms + gen_ms
    latency_ms_per_sample = None
    if prefill_ms_per_sample is not None and gen_ms is not None:
        latency_ms_per_sample = prefill_ms_per_sample + gen_ms
    metrics.setdefault("Latency_ms", latency_ms)
    metrics.setdefault("Prefill_ms", prefill_ms)
    metrics.setdefault("Prefill_ms_per_sample", prefill_ms_per_sample)
    metrics.setdefault("Gen_ms_per_sample", gen_ms)
    metrics.setdefault("Latency_ms_per_sample", latency_ms_per_sample)
