#!/bin/bash
# Run UnlearnCanvas's own style/class classification accuracy + FID metrics against
# a directory of generated images (a secondary, non-FADE evaluation -- see
# compute_fade_all.sh for the FADE metric itself).
#
# Usage: ./scripts/evaluate_images.sh IMAGES_DIR OUTPUT_DIR [DEVICE]
#   IMAGES_DIR: e.g. output_images/CA/Monet
#   OUTPUT_DIR: e.g. output_eval/CA/Monet
#   DEVICE defaults to 0.
#
# Requires the style/class classifier checkpoints from
# https://github.com/OPTML-Group/UnlearnCanvas (ckpts/cls_model/style50.pth,
# ckpts/cls_model/style50_cls.pth) and the reference dataset for FID.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 2 ]; then
  echo "Usage: $0 IMAGES_DIR OUTPUT_DIR [DEVICE]" >&2
  exit 1
fi

IMAGES_DIR="$1"
OUTPUT_DIR="$2"
DEVICE="${3:-0}"
CKPT_PATH=../ckpts/cls_model/style50.pth
CKPT_PATH_CLS=../ckpts/cls_model/style50_cls.pth

CUDA_VISIBLE_DEVICES=${DEVICE} python3 evaluation/classification.py --task style \
  --ckpt ${CKPT_PATH} --input_dir "${IMAGES_DIR}" --output_dir "${OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES=${DEVICE} python3 evaluation/classification.py --task class \
  --ckpt ${CKPT_PATH_CLS} --input_dir "${IMAGES_DIR}" --output_dir "${OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES=${DEVICE} python3 evaluation/fid.py \
  --p1 ./data/unlearn_canvas --p2 "${IMAGES_DIR}" --output-path "${OUTPUT_DIR}/fid.pth"
