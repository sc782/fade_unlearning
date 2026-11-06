#!/bin/bash
# Fine-tune Stable Diffusion v1.5 on the full 50-style UnlearnCanvas dataset.
#
# Most users should NOT need to run this -- the resulting checkpoint is already
# published at https://huggingface.co/sungjuncho/fade-unlearncanvas-full
# (see scripts/download_checkpoints.sh). This is here for reproducibility.
#
# Usage: ./scripts/train_full.sh [GPUS]
#   GPUS defaults to 0,1,2,3,4,5,6,7

set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${1:-0,1,2,3,4,5,6,7}"

python3 main.py \
  --name sd_unlearn_canvas_full \
  --base configs/train_sd_original.yaml \
  --train \
  --gpus "${GPUS}"
