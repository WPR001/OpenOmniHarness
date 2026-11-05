#!/bin/bash
# Agent inference on one GPU per array task (Qwen3.8-27B, local). Run asr/slurm/build_asr.sbatch first.
# Submit from the repository root:
#   BENCH=lvomnibench sbatch --array=0-3 infer/slurm/run_bench.sbatch
# Env: BENCH (required; several names share one model), BACKEND (vllm|hf, default vllm), OUT (run dir, one benchmark),
# PYTHON (the agent env's python, default python), EXTRA (more run_bench.py flags, e.g. --limit, --uids-file, --tag).
#SBATCH --job-name=omniharness-agent
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
BACKEND=${BACKEND:-vllm}
# per-question sandboxes live in RAM and are removed when the job ends
export OH_SCRATCH=/dev/shm/${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
trap 'rm -rf "$OH_SCRATCH"' EXIT
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# one torch compile cache per array task: concurrent tasks sharing one cache read half-written files
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${USER}_agent_${SLURM_ARRAY_TASK_ID:-0}
"${PYTHON:-python}" infer/run_bench.py --bench $BENCH --backend "$BACKEND" ${OUT:+--out "$OUT"} \
    --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}" --shard-id "${SLURM_ARRAY_TASK_ID:-0}" --scratch "$OH_SCRATCH" ${EXTRA:-}
