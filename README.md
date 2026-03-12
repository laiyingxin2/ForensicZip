<div align="center">
<h2>
  <img src="./imgs/logo.jpg" alt="ForensicZip Logo" width="50" height="50" align="absmiddle"> ForensicZip: More Tokens are Better but Not Necessary in Forensic Vision-Language Models
</h2>
</div>

<div align="center">

Yingxin Lai<sup>1</sup>, [Zitong Yu](https://scholar.google.com/citations?hl=en&user=ziHejLwAAAAJ&view_op=list_works&sortby=pubdate)<sup>1⋆</sup>, Jun Wang<sup>1⋆</sup>, Linlin Shen<sup>2</sup>, Yong Xu<sup>3</sup>, and Xiaochun Cao<sup>4</sup>

<sup>1</sup> Great Bay University  
<sup>2</sup> Shenzhen University  
<sup>3</sup> Harbin Institute of Technology  
<sup>4</sup> School of Cyber Science and Technology, Sun Yat-sen University

</div>

<div align="center">

[![GitHub](https://img.shields.io/badge/Code-ForensicZip-181717?logo=github&style=for-the-badge)](https://github.com/laiyingxin2/ForensicZip)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-f9d649?logo=huggingface&style=for-the-badge)](https://huggingface.co/lingcco/fakeVLM)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-f9d649?logo=huggingface&style=for-the-badge)](https://huggingface.co/datasets/lingcco/FakeClue/)
[![Base Framework](https://img.shields.io/badge/Base-FakeVLM-4c78ff?style=for-the-badge)](https://github.com/opendatalab/FakeVLM)

</div>

## Overview

Multimodal Large Language Models (MLLMs) enable interpretable multimedia forensics by generating textual rationales for forgery detection. However, processing dense visual sequences incurs high computational cost, especially for high-resolution images and videos. Existing visual token pruning methods are mostly semantic-driven: they preserve salient objects while often discarding background regions where manipulation traces such as high-frequency anomalies and temporal jitters reside.

To address this issue, we introduce **ForensicZip**, a training-free framework that reformulates token compression from a forgery-driven perspective. ForensicZip models temporal token evolution as a **Birth-Death Optimal Transport** problem with a slack dummy node, quantifying physical discontinuities associated with transient generative artifacts. The final forensic score further integrates transport-based novelty with high-frequency priors, allowing forensic evidence to be preserved under large-ratio compression.

On deepfake and AIGC benchmarks, ForensicZip delivers strong detection performance at aggressive compression ratios, achieving **2.97× speedup** and **over 90% FLOPs reduction** at **10% token retention** while maintaining state-of-the-art accuracy.

<div align="center">
<img src="imgs/framework.jpg" alt="ForensicZip framework" width="90%" height="auto">

**Figure 1.** Overview of the ForensicZip framework. The method preserves forgery-relevant evidence under aggressive token compression by combining transport-based novelty with forensic priors.
</div>

## Model and Dataset

- **Model weights:** https://huggingface.co/lingcco/fakeVLM
- **Dataset:** https://huggingface.co/datasets/lingcco/FakeClue/
- **Base framework:** https://github.com/opendatalab/FakeVLM

See `docs/data_preparation.md` for the expected local file layout.

## Repository Structure

- `forensiczip/` — method implementation and helper utilities
- `fakevlm/` — FakeVLM-compatible skeleton modules
- `scripts/` — evaluation entrypoints
- `docs/` — running and data preparation notes
- `imgs/` — method figures

## Quick Start

### FakeClue

```bash
MODEL_PATH_7B=<MODEL_PATH> \
FAKECLUE_TEST_JSON=<FAKECLUE_JSON> \
FAKECLUE_DATA_BASE=<FAKECLUE_MEDIA_DIR> \
CUDA_DEVICES=0 \
PYTHON_BIN=python \
bash scripts/eval_forensiczip_fakeclue.sh
```

### LOKI

```bash
MODEL_PATH_7B=<MODEL_PATH> \
LOKI_JSON_DIR=<LOKI_JSON_DIR> \
LOKI_MEDIA_ROOT=<LOKI_MEDIA_ROOT> \
CUDA_DEVICES=0 \
PYTHON_BIN=python \
bash scripts/eval_forensiczip_loki.sh
```

See `docs/running.md` for configuration details.

## Acknowledgement

This codebase is built on top of [FakeVLM](https://github.com/opendatalab/FakeVLM). We thank the FakeVLM project for providing the base model and evaluation structure used in this release.

## Citation

If you find this repository useful, please consider citing:

```bibtex
@article{lai2026forensiczip,
  title={ForensicZip: More Tokens are Better but Not Necessary in Forensic Vision-Language Models},
  author={Lai, Yingxin and Yu, Zitong and Wang, Jun and Shen, Linlin and Xu, Yong and Cao, Xiaochun},
  journal={arXiv preprint},
  year={2026}
}
```

## Contact

For questions about this repository, please contact: `yingxinlai2@gmail.com`
