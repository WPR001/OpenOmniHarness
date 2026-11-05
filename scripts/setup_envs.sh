#!/bin/bash
# The two conda envs: omniharness (the agent, requirements.txt) and omniharness-asr (the ASR stage, requirements-asr.txt).
# Run from the repository root. ffmpeg / ffprobe must be on PATH.
set -euo pipefail
eval "$(conda shell.bash hook)"
conda create -n omniharness -c conda-forge --override-channels python=3.12 -y
conda run -n omniharness pip install -r requirements.txt
conda create -n omniharness-asr -c conda-forge --override-channels python=3.12 -y
conda run -n omniharness-asr pip install -r requirements-asr.txt
