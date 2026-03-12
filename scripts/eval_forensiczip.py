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

from forensiczip import enable_forensiczip_for_llava, assert_forensiczip_really_used

def resolve_output_paths(output_path: str, exp_name: str, run_id: int):
    if output_path:
        base_dir = output_path
    else:
        base_dir = os.path.join("results", exp_name or f"run_{run_id}")
    os.makedirs(base_dir, exist_ok=True)
    resp_path = os.path.join(base_dir, "responses.json")
    metrics_path = os.path.join(base_dir, "metrics.json")
    legacy_output = None
    return base_dir, resp_path, metrics_path, legacy_output
from forensiczip.loki_utils import (
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
def eval_loki(args, model, processor, device, image_token_id: Optional[int], run_id: int):
    docs = load_loki_docs(args.test_json_file)

    outputs = []
    acc_list = []
    binary_true, binary_pred = [], []
    gold_yesno = 0
    parsed_yesno = 0
    other_on_yesno = 0
    rouge_list, css_list = [], []
    missing_all = []

    def run_batch(prompts, images, metas):
        nonlocal gold_yesno, parsed_yesno, other_on_yesno
        enc = processor(
            text=prompts,
            images=images if images is not None else None,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=args.max_length,
        )
        enc = _maybe_resize_image_tokens(args, model, processor, enc)
        for k in enc:
            enc[k] = enc[k].to(device)

        if images is not None and isinstance(images, list) and len(prompts) == 1 and len(images) > 1:
            model._forensiczip_group_sizes = [len(images)]
        else:
            model._forensiczip_group_sizes = None

        gen_ids = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
        model._forensiczip_group_sizes = None
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

            outputs.append({
                "run_id": run_id,
                "id": doc.get("id", ""),
                "image_path": doc.get("image_path"),
                "prompt": doc.get("prompt"),
                "answer": gold,
                "prediction": resp,
                "pred_label": pred_label,
                "correct": correct,
            })

    image_batch = []
    text_batch = []
    for doc in tqdm(docs):
        images = None
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
    return outputs, acc_list, binary_true, binary_pred, rouge_list, css_list, missing_all, yesno_stats


def main():
    args = parse_args()
    set_seed(args.seed)

    run_id = random.randint(100000, 999999)
    base_dir, resp_path, metrics_path, legacy_output = resolve_output_paths(args.output_path, args.exp_name, run_id)

    proc_kwargs = {}
    if args.processor_revision and args.processor_path != "<PROCESSOR_PATH>" and not os.path.exists(args.processor_path):
        proc_kwargs["revision"] = args.processor_revision
    processor = AutoProcessor.from_pretrained(args.processor_path if args.processor_path != "<PROCESSOR_PATH>" else args.model_path, **proc_kwargs)
    if not hasattr(processor, "patch_size") or processor.patch_size is None:
        processor.patch_size = 14
    if not hasattr(processor, "vision_feature_select_strategy") or processor.vision_feature_select_strategy is None:
        processor.vision_feature_select_strategy = "default"
    if hasattr(processor, "tokenizer") and getattr(processor.tokenizer, "padding_side", None) != "left":
        processor.tokenizer.padding_side = "left"

    print("Loading model (HF LlavaForConditionalGeneration)...")
    load_kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, revision=args.processor_revision if args.processor_revision else None, device_map="auto")
    if load_kwargs["revision"] is None:
        load_kwargs.pop("revision")
    try:
        model = LlavaForConditionalGeneration.from_pretrained(args.model_path, attn_implementation="flash_attention_2", **load_kwargs).eval()
    except TypeError:
        model = LlavaForConditionalGeneration.from_pretrained(args.model_path, **load_kwargs).eval()

    print("Model loaded.")
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
    t0 = time.time()
    yesno_stats = None

    if args.dataset_type == "loki":
        outputs, acc_list, binary_true, binary_pred, rouge_list, css_list, missing_all, yesno_stats = eval_loki(
            args, model, processor, "cuda", image_token_id, run_id
        )
    else:
        dataset = TestDataset(args.test_json_file, args.data_base_test)
        dataloader = DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
        for batch in tqdm(dataloader, total=len(dataloader)):
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
            for k in enc:
                enc[k] = enc[k].to("cuda")
            model._forensiczip_group_sizes = None
            gen_ids = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
            model._forensiczip_group_sizes = None
            input_len = int(enc["input_ids"].shape[1]) if enc.get("input_ids") is not None else 0
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
                outputs.append({
                    "run_id": run_id,
                    "path": batch["paths"][i],
                    "cate": batch["cates"][i],
                    "prompt": batch["prompts"][i],
                    "ref": batch["refs"][i],
                    "prediction": resp,
                    "pred_label": pred,
                    "label": gold,
                    "correct": int(pred == gold),
                })

    dt = time.time() - t0

    if not args.forensiczip_disable:
        assert_forensiczip_really_used(model)

    acc = float(sum(acc_list) / len(acc_list)) if acc_list else None
    macro_f1 = None
    if args.dataset_type == "fakeclue" and y_true and y_pred:
        acc_tmp, macro_f1 = compute_acc_f1(y_true, y_pred)
        acc = acc_tmp

    metrics = {
        "run_id": run_id,
        "exp_name": args.exp_name,
        "dataset_type": args.dataset_type,
        "num_samples": len(outputs),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "rouge_l_mean": float(sum(rouge_list) / len(rouge_list)) if rouge_list else None,
        "css_mean": float(sum(css_list) / len(css_list)) if css_list else None,
        "elapsed_seconds": dt,
        "compression": {
            "retention": args.forensiczip_retention,
            "select_layer": args.forensiczip_select_layer,
            "birth_cost": args.forensiczip_birth_cost,
            "death_cost": args.forensiczip_death_cost,
            "sinkhorn_eps": args.forensiczip_sinkhorn_eps,
            "sinkhorn_iters": args.forensiczip_sinkhorn_iters,
            "ema_beta": args.forensiczip_ema_beta,
            "birth_weight": args.forensiczip_birth_weight,
            "pos_lambda": args.forensiczip_pos_lambda,
            "forensic_eta": args.forensiczip_forensic_eta,
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

    with open(resp_path, "w") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n==== FORENSICZIP done ====")
    print(metrics)
    if missing_all:
        print(f"[WARN] missing images: {len(missing_all)}")


if __name__ == "__main__":
    main()
