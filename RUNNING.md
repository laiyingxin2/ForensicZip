# Running Guide

## Expected layout

The package expects a local checkpoint directory for the model and benchmark JSON/media paths.

Required variables for the shell launcher:
- `MODEL_PATH_7B`
- `FAKECLUE_TEST_JSON`
- `FAKECLUE_DATA_BASE`
- `LOKI_JSON_DIR` (if `loki` is enabled)
- `LOKI_MEDIA_ROOT` (if `loki` is enabled)

## Quick run

```bash
MODEL_PATH_7B=<MODEL_PATH>     FAKECLUE_TEST_JSON=<FAKECLUE_JSON>     FAKECLUE_DATA_BASE=<FAKECLUE_MEDIA_DIR>     CUDA_DEVICES=0     PYTHON_BIN=python     bash scripts/eval_forensiczip.sh
```

## FakeClue-only run

```bash
MODEL_PATH_7B=<MODEL_PATH>     FAKECLUE_TEST_JSON=<FAKECLUE_JSON>     FAKECLUE_DATA_BASE=<FAKECLUE_MEDIA_DIR>     DATASETS="fakeclue"     RETENTION_RATIOS_STR="0.50 0.25 0.10"     CUDA_DEVICES=0     PYTHON_BIN=python     bash scripts/eval_forensiczip.sh
```

## LOKI run

```bash
MODEL_PATH_7B=<MODEL_PATH>     FAKECLUE_DATA_BASE=<PLACEHOLDER_OR_MEDIA_DIR>     LOKI_JSON_DIR=<LOKI_JSON_DIR>     LOKI_MEDIA_ROOT=<LOKI_MEDIA_ROOT>     DATASETS="loki"     CUDA_DEVICES=0     PYTHON_BIN=python     bash scripts/eval_forensiczip.sh
```

## Main algorithm knobs

- `FORENSICZIP_SELECT_LAYER`
- `FORENSICZIP_BIRTH_COST`
- `FORENSICZIP_DEATH_COST`
- `FORENSICZIP_SINKHORN_EPS`
- `FORENSICZIP_SINKHORN_ITERS`
- `FORENSICZIP_EMA_BETA`
- `FORENSICZIP_BIRTH_WEIGHT`
- `FORENSICZIP_POS_LAMBDA`
- `FORENSICZIP_FORENSIC_ETA`
