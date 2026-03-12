#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from typing import List

import numpy as np
from PIL import Image


def load_json_auto(path: str):
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        return json.loads(raw.decode("utf-16"))
    for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16le", "utf-16be"):
        try:
            return json.loads(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    return json.loads(raw.decode("utf-8", errors="ignore"))


def loki_media_path(path: str, loki_media_root: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    if path.startswith("media_data/"):
        return os.path.join(loki_media_root, path[len("media_data/"):])
    return os.path.join(loki_media_root, path)


def build_prompt_with_images(question: str, num_images: int) -> str:
    if num_images <= 0:
        return question
    if "<video>" in question:
        return question.replace("<video>", "\n".join(["<image>"] * num_images))
    if "<image>" in question:
        return question
    return "\n".join(["<image>"] * num_images) + "\n" + question


def _find_ffmpeg():
    return shutil.which("ffmpeg")


def _find_ffprobe():
    return shutil.which("ffprobe")


def _probe_duration_seconds(video_path: str) -> float:
    ffprobe = _find_ffprobe()
    if not ffprobe:
        return 0.0
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return float(out)
    except Exception:
        return 0.0


def extract_video_frames_ffmpeg(video_path: str, num_frames: int) -> List[Image.Image]:
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH; required for video decoding.")
    duration = _probe_duration_seconds(video_path)
    fps = num_frames / duration if duration and duration > 1e-3 else 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
        out_pat = os.path.join(tmpdir, "frame_%05d.jpg")
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vf",
            f"fps={fps}",
            "-vframes",
            str(num_frames),
            "-q:v",
            "2",
            out_pat,
        ]
        subprocess.run(cmd, check=True)
        frames = []
        for name in sorted(os.listdir(tmpdir)):
            if name.endswith(".jpg"):
                frames.append(Image.open(os.path.join(tmpdir, name)).convert("RGB"))
        return frames


def load_point_cloud_as_image(point_path: str, size: int = 512) -> Image.Image:
    # Minimal PLY loader for vertex positions (x,y,z). Supports ASCII and binary little endian.
    with open(point_path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading PLY header.")
            header.append(line)
            if line.strip() == b"end_header":
                break
        header_text = b"".join(header).decode("utf-8", errors="ignore")
        fmt = "ascii"
        vertex_count = 0
        props = []
        for line in header_text.splitlines():
            if line.startswith("format"):
                if "binary_little_endian" in line:
                    fmt = "binary_little_endian"
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line.startswith("property"):
                parts = line.split()
                if len(parts) >= 3:
                    props.append((parts[1], parts[2]))

        name_to_idx = {name: i for i, (_, name) in enumerate(props)}
        if not {"x", "y", "z"}.issubset(name_to_idx):
            raise ValueError("PLY missing x/y/z properties.")

        if fmt == "ascii":
            xyz = []
            for _ in range(vertex_count):
                line = f.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                vals = line.split()
                x = float(vals[name_to_idx["x"]])
                y = float(vals[name_to_idx["y"]])
                z = float(vals[name_to_idx["z"]])
                xyz.append((x, y, z))
            xyz = np.asarray(xyz, dtype=np.float32)
        else:
            type_map = {
                "char": "i1",
                "int8": "i1",
                "uchar": "u1",
                "uint8": "u1",
                "short": "i2",
                "int16": "i2",
                "ushort": "u2",
                "uint16": "u2",
                "int": "i4",
                "int32": "i4",
                "uint": "u4",
                "uint32": "u4",
                "float": "f4",
                "float32": "f4",
                "double": "f8",
                "float64": "f8",
            }
            dtype = np.dtype([(name, type_map[typ]) for typ, name in props])
            data = np.fromfile(f, dtype=dtype, count=vertex_count)
            xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)

    if xyz.shape[0] == 0:
        return Image.new("RGB", (size, size), color=(0, 0, 0))
    x = xyz[:, 0]
    y = xyz[:, 1]
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    y = (y - y.min()) / (y.max() - y.min() + 1e-6)
    ix = np.clip((x * (size - 1)).astype(np.int32), 0, size - 1)
    iy = np.clip(((1.0 - y) * (size - 1)).astype(np.int32), 0, size - 1)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[iy, ix] = 255
    return Image.fromarray(canvas)


def parse_options(options) -> str:
    if isinstance(options, str):
        options = json.loads(options)
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    return "\n".join(
        [
            f"{letter}. {opt.replace(letter + '.', '').strip()}"
            for letter, opt in zip(option_letters, options)
        ]
    )


def parse_true_or_false(pred_ans: str) -> str:
    """
    Robust yes/no parser for LOKI open-ended tasks.

    Notes:
      - Some model outputs include explanations ("The answer is yes ..."). The old prefix-only
        parser mislabeled most of these as "other", making Macro-F1 meaningless.
      - Keep this lightweight (regex only) and conservative:
          * avoid counting prompt/instruction echoes (e.g. "Answer with yes or no.")
          * prefer explicit answer patterns ("answer is/answer: yes") and anchored tokens.
    """
    if pred_ans is None:
        return "other"
    text = str(pred_ans).strip().lower()
    if not text:
        return "other"
    text = text.replace("\n", " ").strip()

    # Fast paths.
    if text in ("yes", "y", "true"):
        return "yes"
    if text in ("no", "n", "false"):
        return "no"

    # Search within a small prefix to avoid picking up prompt echoes far away in long generations.
    scan = text[:192]

    # Remove common instruction fragments that contain both "yes" and "no" and should not be treated as answers.
    # These often appear due to prompt echoing in generations.
    scan = re.sub(r"\banswer\s+with\s+yes\s+or\s+no\b", " ", scan)
    scan = re.sub(r"\banswer\s+yes\s+if\b", " ", scan)
    scan = re.sub(r"\banswer\s+no\s+otherwise\b", " ", scan)

    # Prefer explicit "answer is/answer:" patterns.
    m = re.search(r"\banswer\s*(?:is|:)\s*(yes|no|true|false)\b", scan)
    if m:
        w = m.group(1)
        return "yes" if w in ("yes", "true") else "no"

    # Prefer anchored answers (allow leading punctuation/quotes).
    m = re.match(r"^[\s\"'“”‘’\(\[\{<]*\b(yes|no|true|false)\b", scan)
    if m:
        w = m.group(1)
        return "yes" if w in ("yes", "true") else "no"

    m_yes = re.search(r"\b(yes|true)\b", scan)
    m_no = re.search(r"\b(no|false)\b", scan)

    if m_yes and m_no:
        # Ambiguous (often instruction echo like "yes or no"); treat as unparseable.
        return "other"
    if m_yes:
        return "yes"
    if m_no:
        return "no"
    return "other"


def parse_multi_choice_info(options):
    if isinstance(options, str):
        options = json.loads(options)
    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))
    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
    if len(candidates) == 0:
        for choice in all_choices:
            if f"**{choice}**" in response:
                candidates.append(choice)
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)
    if len(candidates) == 0:
        for ans in index2ans.values():
            if f"{ans}" in response:
                candidates.append(ans)
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False
    if len(candidates) == 0:
        # Do NOT guess randomly here: it makes results non-reproducible across runs and
        # can hide real model behavior. Use a sentinel that will be scored as incorrect.
        return "other"
    if len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if "(" in response:
                for can in candidates:
                    start_indexes.append(response.find(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.find(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().find(index2ans[can].lower()))
        return candidates[int(np.argmin(start_indexes))]
    return candidates[0]


def check_is_number(string):
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(string):
    string = string.strip()
    is_number = check_is_number(string)
    if is_number:
        string = string.replace(",", "")
        string = float(string)
        string = round(string, 2)
        return [string]
    string = string.lower()
    if len(string) == 1:
        return [" " + string, string + " "]
    return [string]


def eval_open(gold_i, pred_i):
    correct = False
    if isinstance(gold_i, list):
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(str(gold_i))
    for pred in pred_i:
        if isinstance(pred, str):
            for norm_ans in norm_answers:
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct
