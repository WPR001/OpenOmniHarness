#!/bin/bash
# ASR stage on one GPU per array task: transcribe the benchmark videos with Qwen3-ASR + forced aligner.
# Submit from the repository root:
#   BENCH="daily_omni worldsense omnivideobench lvomnibench" sbatch --array=0-3 asr/slurm/build_asr.sbatch
# Env: BENCH (benchmarks, default all four) or VIDEOS (a list file of video paths), BACKEND (vllm|hf, default vllm),
# PYTHON (the ASR env's python, default python), EXTRA (more build.py flags).
#SBATCH --job-name=omniharness-asr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# one torch compile cache per array task: concurrent tasks sharing one cache read half-written files
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${USER}_asr_${SLURM_ARRAY_TASK_ID:-0}
# VIDEOS (a file with one video path per line) replaces the benchmark list
if [ -n "${VIDEOS:-}" ]; then SOURCE=(--videos "$VIDEOS"); else SOURCE=(--bench ${BENCH:-daily_omni worldsense omnivideobench lvomnibench}); fi
"${PYTHON:-python}" asr/build.py "${SOURCE[@]}" --backend "${BACKEND:-vllm}" \
    --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}" --shard-id "${SLURM_ARRAY_TASK_ID:-0}" ${EXTRA:-}
