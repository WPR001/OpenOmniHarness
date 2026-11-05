#!/bin/bash
# Qwen3.8-27B, Qwen3-ASR-1.7B and Qwen3-ForcedAligner-0.6B from Hugging Face into $OH_MODEL_DIR (default ./models),
# where omniharness/config.py looks for them. Needs the `hf` CLI (pip install -U huggingface_hub).
set -euo pipefail
DIR=${OH_MODEL_DIR:-models}
for m in Qwen3.8-27B Qwen3-ASR-1.7B Qwen3-ForcedAligner-0.6B; do
    hf download Qwen/$m --local-dir "$DIR/$m"
done
