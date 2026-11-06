#!/bin/bash
# Generate images from a single checkpoint for one or more UnlearnCanvas styles.
# Thin wrapper around sampling/stable_diffusion/sample_compvis_automated.py.
#
# Usage: ./scripts/sample.sh CKPT OUTPUT_DIR [THEME ...] [-- EXTRA_ARGS ...]
#   CKPT:       path to a .ckpt/.pth checkpoint (compvis format)
#   OUTPUT_DIR: where to write generated images
#   THEME ...:  one or more style names (e.g. Monet Picasso Van_Gogh); if omitted,
#               samples every style in constants/const.py's theme_available_full
#   EXTRA_ARGS: anything after a literal `--` is passed straight through to
#               sample_compvis_automated.py (e.g. --steps, --seed, --cfg-text-list)
#
# Examples:
#   ./scripts/sample.sh logs/retain_seed0.ckpt output_images/retain0_forget Monet Picasso Van_Gogh
#   ./scripts/sample.sh logs/retain_seed0.ckpt output_images/quick_test Monet -- --steps 20 --seed 0 1 2

set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 2 ]; then
  echo "Usage: $0 CKPT OUTPUT_DIR [THEME ...] [-- EXTRA_ARGS ...]" >&2
  exit 1
fi

CKPT="$1"
OUTPUT_DIR="$2"
shift 2

THEMES=()
EXTRA_ARGS=()
seen_separator=false
for arg in "$@"; do
  if [ "$seen_separator" = true ]; then
    EXTRA_ARGS+=("$arg")
  elif [ "$arg" = "--" ]; then
    seen_separator=true
  else
    THEMES+=("$arg")
  fi
done

CMD=(python3 sampling/stable_diffusion/sample_compvis_automated.py --ckpt "${CKPT}" --output_dir "${OUTPUT_DIR}")
if [ ${#THEMES[@]} -gt 0 ]; then
  CMD+=(--themes "${THEMES[@]}")
fi
CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
