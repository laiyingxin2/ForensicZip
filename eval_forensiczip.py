#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

# Ensure repo root is importable when running as `python scripts/*.py` without PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os
import json
import time
import argparse
import random
import numpy as np
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from forensiczip_hf import enable_forensiczip_for_llava, assert_forensiczip_really_used
from efficiency_utils import (
    SeqStats,
    resolve_output_paths,
    cuda_time_ms,
    count_generated_tokens,
    get_model_profile,
    get_kv_cache_profile,
    estimate_prefill_flops,
    estimate_kv_cache_bytes,
    get_param_bytes,
    attach_table_metrics,
)
from loki_utils import (
    load_json_auto,
    loki_media_path,
    build_prompt_with_images,
    extract_video_frames_ffmpeg,
    load_point_cloud_as_image,
    parse_options,
    parse_true_or_false,
    parse_multi_choice_info,
    parse_multi_choice_response,
    eval_open,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser("FakeVLM evaluation with FORENSICZIP (HF-LLaVA)")
    p.add_argument("--dataset_type", type=str, default="fakeclue", choices=["fakeclue", "loki"])
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--processor_path", type=str, default="<PROCESSOR_PATH>")
    p.add_argument("--processor_revision", type=str, default="a272c74")
    p.add_argument("--val_batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--data_base_test", type=str, required=True)
    p.add_argument("--test_json_file", type=str, required=True)
    p.add_argument("--output_path", type=str, default="")
    p.add_argument("--exp_name", type=str, default="fakevlm_forensiczip")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loki_media_root", type=str, default="")
    p.add_argument("--video_num_frames", type=int, default=4)

    # FORENSICZIP knobs
    p.add_argument("--forensiczip_retention", type=float, default=192.0)
    p.add_argument("--forensiczip_select_layer", type=int, default=-2)
    p.add_argument("--forensiczip_keep_cls", action="store_true")
    p.add_argument("--forensiczip_disable", action="store_true")

    p.add_argument("--forensiczip_birth_cost", type=float, default=0.35)
    p.add_argument("--forensiczip_death_cost", type=float, default=0.35)
    p.add_argument("--forensiczip_sinkhorn_eps", type=float, default=0.1)
    p.add_argument("--forensiczip_sinkhorn_iters", type=int, default=20)
    p.add_argument("--forensiczip_ema_beta", type=float, default=0.6)
    p.add_argument("--forensiczip_birth_weight", type=float, default=0.75)
    p.add_argument("--forensiczip_pos_lambda", type=float, default=0.0)
    p.add_argument("--forensiczip_forensic_eta", type=float, default=0.0)

    # profile
    p.add_argument("--profile_batches", type=int, default=0)
    p.add_argument("--efficiency_profile_batches", type=int, default=1)

    return p.parse_args()


class TestDataset(Dataset):
    def __init__(self, test_json_file: str, data_base_test: str):
        self.data_base_test = data_base_test
        with open(test_json_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        rel = item.get("image", "")
        img_path = os.path.join(self.data_base_test, rel) if not os.path.isabs(rel) else rel

        label = int(item.get("label", -1))
        cate = item.get("cate", "deepfake")

        prompt = None
        if "conversations" in item and len(item["conversations"]) > 0:
            prompt = item["conversations"][0].get("value", None)
        if prompt is None:
            prompt = "Is this image real or fake? Answer with 'real' or 'fake'."

        ref = None
        if "conversations" in item and len(item["conversations"]) > 1:
            ref = item["conversations"][1].get("value", None)

        return {
            "img_path": img_path,
            "label": label,
            "cate": cate,
            "prompt": prompt,
            "ref": ref,
        }


def safe_open_image(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def collate_fn(batch: List[Dict[str, Any]]):
    images, prompts = [], []
    labels, paths, cates, refs = [], [], [], []
    missing = []

    for b in batch:
        img = safe_open_image(b["img_path"])
        if img is None:
            missing.append(b["img_path"])
            continue
        images.append(img)
        prompts.append(b["prompt"])
        labels.append(b["label"])
        paths.append(b["img_path"])
        cates.append(b["cate"])
        refs.append(b.get("ref", None))

    return {
        "images": images,
        "prompts": prompts,
        "labels": labels,
        "paths": paths,
        "cates": cates,
        "refs": refs,
        "missing": missing,
    }


def load_loki_docs(path: str):
    if os.path.isdir(path):
        docs = []
        for root, _, files in os.walk(path):
            for name in sorted(files):
                if not name.endswith(".json"):
                    continue
                docs.extend(load_json_auto(os.path.join(root, name)))
        return docs
    return load_json_auto(path)


@torch.no_grad()
def eval_loki(
    args,
    model,
    processor,
    device,
    seq_stats: SeqStats,
    image_token_id: Optional[int],
    run_id: int,
    *,
    generate_state: Optional[Dict[str, Any]] = None,
):
    docs = load_loki_docs(args.test_json_file)

    outputs = []
    acc_list = []
    binary_true, binary_pred = [], []
    gold_yesno = 0
    parsed_yesno = 0
    other_on_yesno = 0
    rouge_list, css_list = [], []
    missing_all = []
    prefill_times = []
    gen_time_ms_total = 0.0
    gen_batches = 0
    prefill_profile_remaining = int(max(0, args.efficiency_profile_batches))

    def run_batch(prompts, images, metas):
        nonlocal gen_time_ms_total, gen_batches, prefill_profile_remaining, gold_yesno, parsed_yesno, other_on_yesno
        enc = processor(
            text=prompts,
            images=images if images is not None else None,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=args.max_length,
        )
        enc = _maybe_resize_image_tokens(args, model, processor, enc)
        seq_stats.update(enc.get("input_ids"), enc.get("attention_mask"), image_token_id)
        for k in enc:
            enc[k] = enc[k].to(device)

        # Set group sizes for multi-image (video) prompts: batch=1 prompt, images=list[T]
        if images is not None and isinstance(images, list) and len(prompts) == 1 and len(images) > 1:
            model._forensiczip_group_sizes = [len(images)]
        else:
            model._forensiczip_group_sizes = None

        if prefill_profile_remaining > 0:
            _, prefill_ms = cuda_time_ms(lambda: model(**enc, use_cache=False))
            prefill_times.append(prefill_ms)
            prefill_profile_remaining -= 1

        if generate_state is not None:
            generate_state["active"] = True
        try:
            gen_ids, gen_ms = cuda_time_ms(
                lambda: model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
            )
        finally:
            if generate_state is not None:
                generate_state["active"] = False
        model._forensiczip_group_sizes = None

        gen_time_ms_total += gen_ms
        gen_batches += 1
        input_len = int(enc["input_ids"].shape[1]) if enc.get("input_ids") is not None else 0

        for i in range(gen_ids.shape[0]):
            decoded = processor.decode(gen_ids[i][input_len:], skip_special_tokens=True)
            resp = extract_response(decoded)
            doc = metas[i]
            gold = doc.get("answer", "")
            pred_label = None
            correct = None

            metric = doc.get("metric")
            if metric == "open-ended":
                pred_label = parse_true_or_false(resp)
                gold_label = parse_true_or_false(str(gold))
                correct = pred_label == gold_label and pred_label != "other"
                if gold_label in ["yes", "no"]:
                    gold_yesno += 1
                    if pred_label in ["yes", "no"]:
                        parsed_yesno += 1
                    else:
                        other_on_yesno += 1
                if gold_label in ["yes", "no"] and pred_label in ["yes", "no"]:
                    binary_true.append(1 if gold_label == "yes" else 0)
                    binary_pred.append(1 if pred_label == "yes" else 0)
                if gold_label not in ["yes", "no"] and str(gold).strip():
                    correct = eval_open(gold, [resp])
                if not str(gold).strip():
                    correct = None
            elif metric == "multi-choice":
                index2ans, all_choices = parse_multi_choice_info(doc.get("choices", []))
                pred_label = parse_multi_choice_response(resp, all_choices, index2ans)
                correct = str(pred_label).strip() == str(gold).strip()
            elif metric == "model-as-judge":
                correct = None
            else:
                if str(gold).strip():
                    correct = eval_open(gold, [resp])
                else:
                    correct = None

            if correct is not None:
                acc_list.append(int(correct))

            r = rouge_l(resp, str(gold)) if gold else float("nan")
            c = css_score(resp, str(gold)) if gold else float("nan")
            if not np.isnan(r):
                rouge_list.append(r)
            if not np.isnan(c):
                css_list.append(c)

            outputs.append(
                {
                    "run_id": run_id,
                    "id": doc.get("id", ""),
                    "image_path": doc.get("image_path"),
                    "prompt": doc.get("prompt"),
                    "answer": gold,
                    "prediction": resp,
                    "pred_label": pred_label,
                    "correct": correct,
                }
            )

    image_batch = []
    text_batch = []
    for doc in tqdm(docs):
        images = None
        image_paths = []
        video_paths = None
        if "video_path" in doc:
            raw_video_paths = doc.get("video_path")
            if not raw_video_paths:
                missing_all.append("missing:video_path")
                continue
            if not isinstance(raw_video_paths, list):
                raw_video_paths = [raw_video_paths]
            raw_video_paths = [p for p in raw_video_paths if p]
            if not raw_video_paths:
                missing_all.append("missing:video_path")
                continue
            video_paths = [loki_media_path(p, args.loki_media_root) for p in raw_video_paths]
            try:
                images = []
                for vp in video_paths:
                    images.extend(extract_video_frames_ffmpeg(vp, args.video_num_frames))
                doc["image_path"] = list(video_paths)
            except Exception:
                missing_all.extend(video_paths)
                continue
        elif "point_path" in doc:
            point_path = loki_media_path(doc["point_path"], args.loki_media_root)
            try:
                images = [load_point_cloud_as_image(point_path)]
                doc["image_path"] = [point_path]
            except Exception:
                missing_all.append(point_path)
                continue
        elif "image_path" in doc:
            image_paths = doc.get("image_path")
            if not isinstance(image_paths, list):
                image_paths = [image_paths]
            image_paths = [loki_media_path(p, args.loki_media_root) for p in image_paths]
            try:
                images = [Image.open(p).convert("RGB") for p in image_paths]
                doc["image_path"] = image_paths
            except Exception:
                missing_all.extend(image_paths)
                continue

        question = doc.get("question", "")
        if images is None:
            prompt = question
        else:
            prompt_images = len(images)
            if video_paths and "<video>" in question:
                prompt_images = args.video_num_frames
            prompt = build_prompt_with_images(question, prompt_images)

        metric = doc.get("metric")
        if metric == "open-ended":
            gold_label = parse_true_or_false(str(doc.get("answer", "")))
            if gold_label in ["yes", "no"]:
                prompt = f"{prompt}\nAnswer with yes or no."
        elif metric == "multi-choice":
            prompt = f"{prompt}\n{parse_options(doc.get('choices', []))}\nAnswer with the option letter."

        doc["prompt"] = prompt

        if images is None:
            text_batch.append((prompt, doc))
            if len(text_batch) >= args.val_batch_size:
                prompts = [p for p, _ in text_batch]
                metas = [d for _, d in text_batch]
                run_batch(prompts, None, metas)
                text_batch = []
            continue

        if len(images) == 1:
            image_batch.append((prompt, images[0], doc))
            if len(image_batch) >= args.val_batch_size:
                prompts = [p for p, _, _ in image_batch]
                imgs = [im for _, im, _ in image_batch]
                metas = [d for _, _, d in image_batch]
                run_batch(prompts, imgs, metas)
                image_batch = []
            continue

        run_batch([prompt], images, [doc])

    if text_batch:
        prompts = [p for p, _ in text_batch]
        metas = [d for _, d in text_batch]
        run_batch(prompts, None, metas)
    if image_batch:
        prompts = [p for p, _, _ in image_batch]
        imgs = [im for _, im, _ in image_batch]
        metas = [d for _, _, d in image_batch]
        run_batch(prompts, imgs, metas)

    yesno_stats = {
        "gold_yesno": int(gold_yesno),
        "parsed_yesno": int(parsed_yesno),
        "other_on_yesno": int(other_on_yesno),
    }
    return (
        outputs,
        acc_list,
        binary_true,
        binary_pred,
        rouge_list,
        css_list,
        missing_all,
        prefill_times,
        gen_time_ms_total,
        gen_batches,
        yesno_stats,
    )


def _resize_image_tokens_in_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_token_id: int,
    retention: float,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_ids.dim() != 2 or attention_mask.dim() != 2:
        return input_ids, attention_mask

    bsz, seqlen = input_ids.shape
    new_input_ids = []
    new_attention = []

    batch_left_padding = True
    try:
        any_left = bool((attention_mask[:, 0] == 0).any().item())
        any_right = bool((attention_mask[:, -1] == 0).any().item())
        if any_right and not any_left:
            batch_left_padding = False
    except Exception:
        batch_left_padding = True

    for b in range(bsz):
        ids = input_ids[b]
        am = attention_mask[b]
        mask = am.bool()
        left_padding = batch_left_padding
        real_ids = ids[mask].tolist()

        rebuilt: list[int] = []
        i = 0
        while i < len(real_ids):
            if real_ids[i] == image_token_id:
                j = i
                while j < len(real_ids) and real_ids[j] == image_token_id:
                    j += 1
                run_len = j - i
                if retention <= 1.0:
                    target_len = max(1, int(round(run_len * retention)))
                else:
                    target_len = max(1, int(round(retention)))
                if target_len > run_len:
                    target_len = run_len
                rebuilt.extend([image_token_id] * target_len)
                i = j
            else:
                rebuilt.append(real_ids[i])
                i += 1

        if len(rebuilt) > seqlen:
            rebuilt = rebuilt[:seqlen]
        if len(rebuilt) < seqlen:
            pad_len = seqlen - len(rebuilt)
            if left_padding:
                rebuilt = ([pad_token_id] * pad_len) + rebuilt
                attn = ([0] * pad_len) + ([1] * len(rebuilt[pad_len:]))
            else:
                attn = [1] * len(rebuilt) + ([0] * pad_len)
                rebuilt.extend([pad_token_id] * pad_len)
        else:
            attn = [1] * len(rebuilt)

        new_input_ids.append(rebuilt)
        new_attention.append(attn)

    new_input_ids = torch.tensor(new_input_ids, dtype=input_ids.dtype, device=input_ids.device)
    new_attention = torch.tensor(new_attention, dtype=attention_mask.dtype, device=attention_mask.device)
    return new_input_ids, new_attention


def _maybe_resize_image_tokens(args, model, processor, enc):
    if getattr(args, "forensiczip_disable", False):
        return enc
    if not getattr(model.config, "forensiczip_resize_input_ids", False):
        return enc

    image_token_id = getattr(model.config, "image_token_index", None)
    if image_token_id is None:
        image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        return enc

    input_ids = enc.get("input_ids")
    attention_mask = enc.get("attention_mask")
    if input_ids is None or attention_mask is None:
        return enc

    pad_token_id = getattr(processor, "pad_token_id", None)
    if pad_token_id is None and hasattr(processor, "tokenizer"):
        pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        return enc

    new_input_ids, new_attention = _resize_image_tokens_in_batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_token_id=image_token_id,
        retention=args.forensiczip_retention,
        pad_token_id=pad_token_id,
    )
    enc["input_ids"] = new_input_ids
    enc["attention_mask"] = new_attention
    try:
        if hasattr(model, "config") and hasattr(model.config, "image_seq_length"):
            img_counts = (new_input_ids == image_token_id).sum(dim=1)
            max_img = int(img_counts.max().item()) if img_counts.numel() else 0
            if max_img > 0:
                model.config.image_seq_length = max_img
    except Exception:
        pass
    return enc


def extract_response(decoded: str) -> str:
    text = decoded.strip()
    for key in ["\nASSISTANT:", "ASSISTANT:", "\n### Assistant:", "### Assistant:"]:
        if key in text:
            text = text.split(key)[-1].strip()
    return text.strip()


def pred_from_response(resp: str) -> int:
    low = resp.lower()
    seg0 = low.split(".")[0] if "." in low else low
    if "real" in seg0:
        return 1
    if "fake" in seg0:
        return 0
    parts = low.split(".")
    if len(parts) > 1:
        if "real" in parts[1]:
            return 1
        if "fake" in parts[1]:
            return 0
    return random.choice([0, 1])


def rouge_l(pred: str, ref: str) -> float:
    if pred is None or ref is None:
        return float("nan")
    pred = pred.strip()
    ref = ref.strip()
    if len(pred) == 0 or len(ref) == 0:
        return 0.0
    a = pred.split()
    b = ref.split()
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[n][m]
    r = lcs / m
    p = lcs / n
    if r + p == 0:
        return 0.0
    return (2 * r * p) / (r + p)


def css_score(pred: str, ref: str) -> float:
    if pred is None or ref is None:
        return float("nan")
    pred = pred.strip()
    ref = ref.strip()
    if len(pred) == 0 or len(ref) == 0:
        return 0.0
    set_a = set(pred.split())
    set_b = set(ref.split())
    if len(set_b) == 0:
        return 0.0
    return len(set_a & set_b) / len(set_b)


def compute_metrics(y_true: List[int], y_pred: List[int]) -> tuple[float, float]:
    if len(y_true) == 0:
        return 0.0, 0.0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float((y_true == y_pred).mean())

    def f1_for_label(lbl: int) -> float:
        tp = int(((y_true == lbl) & (y_pred == lbl)).sum())
        fp = int(((y_true != lbl) & (y_pred == lbl)).sum())
        fn = int(((y_true == lbl) & (y_pred != lbl)).sum())
        if tp + fp == 0 or tp + fn == 0:
            return 0.0
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    f1_0 = f1_for_label(0)
    f1_1 = f1_for_label(1)
    macro_f1 = (f1_0 + f1_1) / 2.0
    return acc, macro_f1


def main():
    args = parse_args()
    set_seed(args.seed)

    run_id = random.randint(100000, 999999)
    base_dir, resp_path, metrics_path, legacy_output = resolve_output_paths(args.output_path, args.exp_name, run_id)

    proc_kwargs = {}
    if args.processor_revision and not os.path.exists(args.processor_path):
        proc_kwargs["revision"] = args.processor_revision
    processor = AutoProcessor.from_pretrained(args.processor_path, **proc_kwargs)
    if not hasattr(processor, "patch_size") or processor.patch_size is None:
        processor.patch_size = 14
    if not hasattr(processor, "vision_feature_select_strategy") or processor.vision_feature_select_strategy is None:
        processor.vision_feature_select_strategy = "default"
    if hasattr(processor, "tokenizer") and getattr(processor.tokenizer, "padding_side", None) != "left":
        processor.tokenizer.padding_side = "left"
    eos_token_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    pad_token_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)

    print("Loading model (HF LlavaForConditionalGeneration)...")
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        revision="a272c74",
        device_map="auto",
    )
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path, attn_implementation="flash_attention_2", **load_kwargs
        ).eval()
    except TypeError:
        model = LlavaForConditionalGeneration.from_pretrained(args.model_path, **load_kwargs).eval()

    print("Model loaded.")
    base_image_seq_len = int(getattr(model.config, "image_seq_length", 576))
    image_token_id = getattr(model.config, "image_token_index", None)
    if image_token_id is None:
        image_token_id = getattr(processor, "image_token_id", None)

    if not args.forensiczip_disable:
        model = enable_forensiczip_for_llava(
            model,
            retention=args.forensiczip_retention,
            select_layer=args.forensiczip_select_layer,
            keep_cls=args.forensiczip_keep_cls,
            birth_cost=args.forensiczip_birth_cost,
            death_cost=args.forensiczip_death_cost,
            sinkhorn_eps=args.forensiczip_sinkhorn_eps,
            sinkhorn_iters=args.forensiczip_sinkhorn_iters,
            ema_beta=args.forensiczip_ema_beta,
            birth_weight=args.forensiczip_birth_weight,
            pos_lambda=args.forensiczip_pos_lambda,
            forensic_eta=args.forensiczip_forensic_eta,
            verbose=True,
        )
    else:
        print("[FORENSICZIP] disabled by --forensiczip_disable")

    outputs = []
    y_true, y_pred = [], []
    acc_list = []
    binary_true, binary_pred = [], []
    rouge_list, css_list = [], []
    missing_all = []
    seq_stats = SeqStats()
    prefill_times = []
    prefill_batch_sizes = []
    prefill_tokens_total = 0
    gen_time_ms_total = 0.0
    gen_batches = 0
    gen_tokens_total = 0
    gen_tokens_samples = 0
    gen_end_seq_len_tokens_sum = 0
    gen_end_seq_len_samples = 0
    kv_cache_bytes_sum = 0.0
    kv_cache_bytes_batches = 0
    kv_cache_bytes_peak = None
    kv_cache_peak_seq_len = None
    kv_cache_peak_batch_size = None
    prefill_profile_remaining = int(max(0, args.efficiency_profile_batches))
    kv_profile = get_kv_cache_profile(model)

    # Capture vision-time inside `model.generate()` via CUDA events (no extra sync during generation).
    vision_events = []
    projector_events = []
    _vision_starts = {}
    _proj_starts = {}
    generate_state = {"active": False}

    def _pre_hook(starts_dict):
        def _fn(module, _inputs):
            if not torch.cuda.is_available():
                return
            if not generate_state.get("active", False):
                return
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            starts_dict[id(module)] = ev

        return _fn

    def _post_hook(starts_dict, events_list):
        def _fn(module, _inputs, _output):
            if not torch.cuda.is_available():
                return
            if not generate_state.get("active", False):
                return
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            start = starts_dict.pop(id(module), None)
            if start is not None:
                events_list.append((start, end))

        return _fn

    vision_pre_h = None
    vision_post_h = None
    proj_pre_h = None
    proj_post_h = None
    try:
        if hasattr(model, "vision_tower") and model.vision_tower is not None:
            vision_pre_h = model.vision_tower.register_forward_pre_hook(_pre_hook(_vision_starts))
            vision_post_h = model.vision_tower.register_forward_hook(_post_hook(_vision_starts, vision_events))
        if hasattr(model, "multi_modal_projector") and model.multi_modal_projector is not None:
            proj_pre_h = model.multi_modal_projector.register_forward_pre_hook(_pre_hook(_proj_starts))
            proj_post_h = model.multi_modal_projector.register_forward_hook(_post_hook(_proj_starts, projector_events))
    except Exception:
        vision_pre_h = None
        vision_post_h = None
        proj_pre_h = None
        proj_post_h = None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.time()
    yesno_stats = None
    if args.dataset_type == "loki":
        (
            outputs,
            acc_list,
            binary_true,
            binary_pred,
            rouge_list,
            css_list,
            missing_all,
            prefill_times,
            gen_time_ms_total,
            gen_batches,
            yesno_stats,
        ) = eval_loki(args, model, processor, "cuda", seq_stats, image_token_id, run_id, generate_state=generate_state)
    else:
        dataset = TestDataset(args.test_json_file, args.data_base_test)
        dataloader = DataLoader(
            dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        for bi, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
            missing_all.extend(batch["missing"])
            if len(batch["images"]) == 0:
                continue

            enc = processor(
                text=batch["prompts"],
                images=batch["images"],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=args.max_length,
            )
            enc = _maybe_resize_image_tokens(args, model, processor, enc)
            seq_stats.update(enc.get("input_ids"), enc.get("attention_mask"), image_token_id)
            for k in enc:
                enc[k] = enc[k].to("cuda")

            # Per-sample images only -> no temporal grouping.
            model._forensiczip_group_sizes = None

            if prefill_profile_remaining > 0:
                _, prefill_ms = cuda_time_ms(lambda: model(**enc, use_cache=False))
                prefill_times.append(prefill_ms)
                prefill_batch_sizes.append(int(enc["input_ids"].shape[0]))
                prefill_tokens_total += int(enc["input_ids"].shape[0]) * int(enc["input_ids"].shape[1])
                prefill_profile_remaining -= 1

            generate_state["active"] = True
            try:
                gen_ids, gen_ms = cuda_time_ms(
                    lambda: model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
                )
            finally:
                generate_state["active"] = False
            model._forensiczip_group_sizes = None

            gen_time_ms_total += gen_ms
            gen_batches += 1
            input_len = int(enc["input_ids"].shape[1]) if enc.get("input_ids") is not None else 0
            if gen_ids is not None and isinstance(gen_ids, torch.Tensor) and gen_ids.dim() == 2:
                lens = count_generated_tokens(
                    gen_ids,
                    input_len=input_len,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )
                if lens is not None:
                    gen_tokens_total += int(lens.sum().item())
                    gen_tokens_samples += int(lens.numel())
                end_len = int(gen_ids.shape[1])
                bsz = int(gen_ids.shape[0])
                gen_end_seq_len_tokens_sum += end_len * bsz
                gen_end_seq_len_samples += bsz
                if kv_profile is not None:
                    kv_bytes = float(estimate_kv_cache_bytes(end_len, kv_profile, batch_size=bsz))
                    kv_cache_bytes_sum += kv_bytes
                    kv_cache_bytes_batches += 1
                    if kv_cache_bytes_peak is None or kv_bytes > kv_cache_bytes_peak:
                        kv_cache_bytes_peak = kv_bytes
                        kv_cache_peak_seq_len = end_len
                        kv_cache_peak_batch_size = bsz

            for i in range(gen_ids.shape[0]):
                decoded = processor.decode(gen_ids[i][input_len:], skip_special_tokens=True)
                resp = extract_response(decoded)
                pred = pred_from_response(resp)
                gold = int(batch["labels"][i])
                y_true.append(gold)
                y_pred.append(pred)
                acc_list.append(int(pred == gold))

                ref = batch["refs"][i]
                r = rouge_l(resp, ref) if ref else float("nan")
                c = css_score(resp, ref) if ref else float("nan")
                if not np.isnan(r):
                    rouge_list.append(r)
                if not np.isnan(c):
                    css_list.append(c)

                outputs.append(
                    {
                        "run_id": run_id,
                        "path": batch["paths"][i],
                        "cate": batch["cates"][i],
                        "prompt": batch["prompts"][i],
                        "ref": batch["refs"][i],
                        "prediction": resp,
                        "pred_label": pred,
                        "label": gold,
                        "correct": int(pred == gold),
                    }
                )

            if args.profile_batches > 0 and (bi + 1) >= args.profile_batches:
                break

    dt = time.time() - t0

    # Remove hooks (safety).
    for h in (vision_pre_h, vision_post_h, proj_pre_h, proj_post_h):
        try:
            if h is not None:
                h.remove()
        except Exception:
            pass

    if not args.forensiczip_disable:
        assert_forensiczip_really_used(model)

    # profile (lightweight)
    elapsed = float(dt)
    if args.dataset_type == "loki":
        acc = float(np.mean(acc_list)) if acc_list else None
        if binary_true:
            _, macro_f1 = compute_metrics(binary_true, binary_pred)
        else:
            macro_f1 = None
    else:
        acc, macro_f1 = compute_metrics(y_true, y_pred)
    rouge_avg = float(np.mean(rouge_list)) if len(rouge_list) > 0 else None
    css_avg = float(np.mean(css_list)) if len(css_list) > 0 else None

    avg_seq_len, avg_img_len, avg_text_len = seq_stats.averages()
    avg_img_runs = seq_stats.avg_image_runs()
    profile = get_model_profile(model)
    if avg_text_len is None and avg_seq_len is not None and avg_img_len is not None:
        avg_text_len = avg_seq_len - avg_img_len

    base_seq_len = None
    if avg_text_len is not None:
        runs = avg_img_runs if avg_img_runs is not None else 1.0
        base_seq_len = avg_text_len + float(runs) * float(base_image_seq_len)

    prefill_flops = None
    prefill_flops_base = None
    prefill_ratio = None
    if avg_text_len is not None and avg_img_len is not None:
        prefill_flops = estimate_prefill_flops(avg_text_len + avg_img_len, profile)
        if base_seq_len is not None:
            prefill_flops_base = estimate_prefill_flops(base_seq_len, profile)
            if prefill_flops_base and prefill_flops_base > 0:
                prefill_ratio = prefill_flops / prefill_flops_base

    peak_alloc = None
    peak_reserved = None
    activation_gb = None
    param_gb = None
    param_tb = None
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        param_bytes = get_param_bytes(model)
        param_gb = param_bytes / (1024 ** 3)
        param_tb = param_bytes / (1024 ** 4)
        activation_gb = max(0.0, (peak_alloc - param_bytes) / (1024 ** 3))

    total_prefill_ms = float(sum(prefill_times)) if prefill_times else 0.0
    total_prefill_samples = int(sum(prefill_batch_sizes)) if prefill_batch_sizes else 0
    avg_prefill_ms_per_sample = (total_prefill_ms / float(total_prefill_samples)) if total_prefill_samples > 0 else None
    prefill_tokens_per_sec = None
    if total_prefill_ms > 0 and prefill_tokens_total > 0:
        prefill_tokens_per_sec = float(prefill_tokens_total) / (total_prefill_ms / 1000.0)

    gen_tokens_per_sec = None
    gen_ms_per_token = None
    if gen_time_ms_total > 0 and gen_tokens_total > 0:
        gen_tokens_per_sec = float(gen_tokens_total) / (gen_time_ms_total / 1000.0)
        gen_ms_per_token = float(gen_time_ms_total) / float(gen_tokens_total)

    samples_per_sec = (float(len(outputs)) / elapsed) if elapsed > 0 and len(outputs) > 0 else None
    gen_samples_per_sec = (float(len(outputs)) / (gen_time_ms_total / 1000.0)) if gen_time_ms_total > 0 and len(outputs) > 0 else None

    kv_cache_mb_per_sample = None
    kv_cache_mb_batch_peak = None
    kv_cache_mb_batch_avg = None
    kv_cache_mb_per_sample_peak = None
    avg_end_seq_len = None
    if kv_profile is not None and gen_end_seq_len_samples > 0:
        avg_end_seq_len = float(gen_end_seq_len_tokens_sum) / float(gen_end_seq_len_samples)
        kv_cache_mb_per_sample = float(estimate_kv_cache_bytes(avg_end_seq_len, kv_profile, batch_size=1)) / (1024 ** 2)
        if kv_cache_bytes_batches > 0:
            kv_cache_mb_batch_avg = float(kv_cache_bytes_sum / float(kv_cache_bytes_batches)) / (1024 ** 2)
        if kv_cache_bytes_peak is not None:
            kv_cache_mb_batch_peak = float(kv_cache_bytes_peak) / (1024 ** 2)
        if kv_cache_peak_seq_len is not None:
            kv_cache_mb_per_sample_peak = float(estimate_kv_cache_bytes(kv_cache_peak_seq_len, kv_profile, batch_size=1)) / (1024 ** 2)

    with open(resp_path, "w") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    metrics = {
        "exp_name": args.exp_name,
        "run_id": run_id,
        "dataset_type": args.dataset_type,
        "num_samples": len(acc_list) if args.dataset_type == "loki" else len(y_true),
        "acc": acc,
        "macro_f1": macro_f1,
        "rouge_l": rouge_avg,
        "css": css_avg,
        "elapsed_sec": elapsed,
        "method": "forensiczip",
        "forensiczip": {
            "retention": args.forensiczip_retention,
            "select_layer": args.forensiczip_select_layer,
            "keep_cls": bool(args.forensiczip_keep_cls),
            "birth_cost": args.forensiczip_birth_cost,
            "death_cost": args.forensiczip_death_cost,
            "sinkhorn_eps": args.forensiczip_sinkhorn_eps,
            "sinkhorn_iters": args.forensiczip_sinkhorn_iters,
            "ema_beta": args.forensiczip_ema_beta,
            "birth_weight": args.forensiczip_birth_weight,
            "pos_lambda": args.forensiczip_pos_lambda,
            "forensic_eta": args.forensiczip_forensic_eta,
        },
        "efficiency": {
            "avg_seq_len": avg_seq_len,
            "avg_text_len": avg_text_len,
            "avg_image_len": avg_img_len,
            "base_image_len": base_image_seq_len,
            "prefill_flops_t": (prefill_flops / 1e12) if prefill_flops is not None else None,
            "prefill_flops_base_t": (prefill_flops_base / 1e12) if prefill_flops_base is not None else None,
            "prefill_flops_ratio": prefill_ratio,
            "avg_prefill_ms": (sum(prefill_times) / len(prefill_times)) if prefill_times else None,
            "avg_prefill_ms_per_sample": avg_prefill_ms_per_sample,
            "prefill_tokens_per_sec": prefill_tokens_per_sec,
            "total_generate_ms": gen_time_ms_total,
            "avg_generate_ms_per_batch": (gen_time_ms_total / gen_batches) if gen_batches > 0 else None,
            "avg_generate_ms_per_sample": (gen_time_ms_total / len(outputs)) if len(outputs) > 0 else None,
            # Table-4 style decomposition (approx):
            # - model_generate_s: measured wall time for `model.generate` (includes vision+LLM)
            # - llm_generate_s: subtract vision_tower+projector GPU time from model.generate wall time
            "model_generate_s": (gen_time_ms_total / 1000.0) if gen_time_ms_total is not None else None,
            "generate_tokens_total": int(gen_tokens_total),
            "generate_tokens_per_sec": gen_tokens_per_sec,
            "generate_ms_per_token": gen_ms_per_token,
            "samples_per_sec": samples_per_sec,
            "generate_samples_per_sec": gen_samples_per_sec,
            "max_memory_alloc_gb": (peak_alloc / (1024 ** 3)) if peak_alloc is not None else None,
            "max_memory_reserved_gb": (peak_reserved / (1024 ** 3)) if peak_reserved is not None else None,
            "param_gb": param_gb,
            "param_tb": param_tb,
            "activation_gb": activation_gb,
            "avg_end_seq_len": avg_end_seq_len,
            "kv_cache_mb_per_sample": kv_cache_mb_per_sample,
            "kv_cache_mb_per_sample_peak": kv_cache_mb_per_sample_peak,
            "kv_cache_mb_per_batch_avg": kv_cache_mb_batch_avg,
            "kv_cache_mb_per_batch_peak": kv_cache_mb_batch_peak,
        },
        "outputs": {
            "responses_json": resp_path,
            "metrics_json": metrics_path,
            "legacy_output_path": legacy_output,
            "base_dir": base_dir,
        },
    }

    if args.dataset_type == "loki" and isinstance(yesno_stats, dict):
        gold = int(yesno_stats.get("gold_yesno", 0) or 0)
        parsed = int(yesno_stats.get("parsed_yesno", 0) or 0)
        yesno_stats = dict(yesno_stats)
        yesno_stats["parsed_rate"] = (parsed / gold) if gold > 0 else None
        metrics["loki_yesno"] = yesno_stats

    eff = metrics.get("efficiency", {})
    keep_eff = getattr(model, "_forensiczip_last_keep", None)
    base_eff = getattr(model, "_forensiczip_last_num_patches", None)
    if keep_eff is None and avg_img_len is not None:
        keep_eff = avg_img_len
    if base_eff is None and base_image_seq_len is not None:
        runs = avg_img_runs if avg_img_runs is not None else 1.0
        base_eff = float(runs) * float(base_image_seq_len)
    eff["effective_image_len"] = keep_eff
    eff["effective_image_ratio"] = (keep_eff / base_eff) if (keep_eff is not None and base_eff) else None
    metrics["efficiency"] = eff

    metrics.setdefault("KVCache_MB", kv_cache_mb_per_sample)

    # Compute (vision_tower + projector) time in ms from CUDA events.
    vision_ms_total = None
    projector_ms_total = None
    llm_generate_s = None
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            vision_ms_total = float(sum(s.elapsed_time(e) for s, e in vision_events)) if vision_events else 0.0
        except Exception:
            vision_ms_total = None
        try:
            projector_ms_total = float(sum(s.elapsed_time(e) for s, e in projector_events)) if projector_events else 0.0
        except Exception:
            projector_ms_total = None

        model_ms_total = float(gen_time_ms_total) if gen_time_ms_total is not None else None
        if model_ms_total is not None and vision_ms_total is not None:
            vms = float(vision_ms_total) + (float(projector_ms_total) if projector_ms_total is not None else 0.0)
            llm_ms = max(0.0, model_ms_total - vms)
            llm_generate_s = llm_ms / 1000.0

    eff = metrics.get("efficiency", {})
    eff["vision_tower_ms_total"] = vision_ms_total
    eff["projector_ms_total"] = projector_ms_total
    eff["llm_generate_s"] = llm_generate_s
    metrics["efficiency"] = eff

    attach_table_metrics(metrics)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n==== FORENSICZIP done ====")
    print(metrics)
    if missing_all:
        print(f"[WARN] missing images: {len(missing_all)}")


if __name__ == "__main__":
    main()
