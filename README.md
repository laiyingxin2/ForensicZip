# ForensicZip Code Package

This package contains the core code needed to inspect and run the ForensicZip method in a FakeVLM-style evaluation setup.

Included components:
- `scripts/eval_forensiczip.sh`
- `scripts/eval_forensiczip.py`
- `forensiczip_hf.py`
- `efficiency_utils.py`
- `loki_utils.py`
- `fakevlm_skeleton/`
- `RUNNING.md`

Scope:
- method implementation and evaluation entrypoints
- FakeVLM-compatible model/data skeleton
- LLaVA-style HuggingFace patching for visual token compression
- efficiency accounting and benchmark helpers
