#!/bin/bash
# Fine-tune a retain-only oracle: Stable Diffusion v1.5 fine-tuned on the 47/50
# UnlearnCanvas styles that exclude Monet/Picasso/Van_Gogh (see constants/const.py).
#
# Most users should NOT need to run this -- the 3 published retain seeds are at
# https://huggingface.co/sungjuncho/fade-unlearncanvas-retain-seed{0,1,2}
# (see scripts/download_checkpoints.sh). This is here for reproducibility, e.g. to
# train additional seeds.
#
# Usage: ./scripts/train_retain.sh SEED [GPUS]
#   SEED: integer random seed (0, 1, 2 were used for the published checkpoints)
#   GPUS defaults to 0,1,2,3

set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
  echo "Usage: $0 SEED [GPUS]" >&2
  exit 1
fi

SEED="$1"
GPUS="${2:-0,1,2,3}"

python3 main.py \
  --name "sd_unlearn_canvas_retain_seed${SEED}" \
  --base configs/train_sd_retain.yaml \
  --train \
  --seed "${SEED}" \
  --gpus "${GPUS}"
