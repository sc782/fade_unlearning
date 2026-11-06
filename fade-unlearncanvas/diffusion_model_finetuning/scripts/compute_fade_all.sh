#!/bin/bash
# FADE for every model against the retain-only oracle retain_seed0, for the 3 forget
# styles, both directions:
#   - retain0 samples once and is scored against all the other models (term1), and
#   - each other model samples and is scored against retain0 (term2).
# The other models are retain seeds 1 and 2 (the floor), the full model, and the 5
# unlearned models. Then aggregate.py prints the FADE table.
#
# Downloads every checkpoint it needs (full, 3 retain, 15 unlearned) from HuggingFace
# first if not already present locally -- see scripts/download_checkpoints.py.
#
# Usage: ./scripts/compute_fade_all.sh [GPU ...]
#   Styles are spread over the given GPUs (default: 0). NUM_SEEDS (default 100) sets the
#   number of images per direction to NUM_SEEDS x 20 object classes.

set -euo pipefail
cd "$(dirname "$0")/.."

python3 scripts/download_checkpoints.py

THEMES=( Monet Picasso Van_Gogh )
GPUS=( "${@:-0}" )
SEEDS=$(seq 0 $(( ${NUM_SEEDS:-100} - 1 )))
MU=../machine_unlearning

run_theme () {
  local THEME="$1" GPU="$2"
  local LOW
  LOW=$(echo "${THEME}" | tr '[:upper:]' '[:lower:]')
  # CA uses last-v1.ckpt -- last.ckpt in the same directory is a stale checkpoint left
  # over from an interrupted training run.
  local -A MODELS=(
    [retain1]=logs/retain_seed1.ckpt
    [retain2]=logs/retain_seed2.ckpt
    [full]=../ckpts/diffusion/compvis/style50/compvis.ckpt
    [CA]=${MU}/mu_concept_ablation_ca/logs/ca_${LOW}/checkpoints/last-v1.ckpt
    [EDiff]=${MU}/mu_erase_diff_ediff/results/${THEME}/sd.ckpt
    [ESD]=${MU}/mu_erasing_concept_esd/results/style50/${THEME}.pth
    [SalUn]=${MU}/mu_saliency_unlearn_salun/results/style50/${THEME}/sd_0.4.ckpt
    [SHS]=${MU}/mu_scissorhands_shs/results/${THEME}/sd.ckpt
  )
  local NAMES=( retain1 retain2 full CA EDiff ESD SalUn SHS )
  local OTHERS=()
  for NAME in "${NAMES[@]}"; do OTHERS+=( "${NAME}=${MODELS[$NAME]}" ); done

  CUDA_VISIBLE_DEVICES=${GPU} python3 compute_fade.py --theme "${THEME}" --seeds ${SEEDS} \
    --driver retain0=logs/retain_seed0.ckpt --others "${OTHERS[@]}"
  for NAME in "${NAMES[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU} python3 compute_fade.py --theme "${THEME}" --seeds ${SEEDS} \
      --driver "${NAME}=${MODELS[$NAME]}" --others retain0=logs/retain_seed0.ckpt
  done
}

PIDS=()
for g in "${!GPUS[@]}"; do
  (
    for t in "${!THEMES[@]}"; do
      if (( t % ${#GPUS[@]} == g )); then run_theme "${THEMES[$t]}" "${GPUS[$g]}"; fi
    done
  ) &
  PIDS+=( $! )
done
for PID in "${PIDS[@]}"; do wait "${PID}"; done

python3 aggregate.py
