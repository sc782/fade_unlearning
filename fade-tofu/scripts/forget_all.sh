#!/usr/bin/env bash
# Runs every unlearning method on every forget split, for each seed and LoRA rank,
# starting from the matching full model (ckpts/full/seed*, downloaded if missing).
# Adapters are saved after each of the 5 epochs under
# ckpts/<method>/r<rank>/<split>/seed<seed>/checkpoint-<step>.
#
#   ./scripts/forget_all.sh
#   METHODS="npo" SPLITS="forget10" RANKS="32" ./scripts/forget_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

METHODS=${METHODS:-"ga gd npo dpo ihl"}
SPLITS=${SPLITS:-"forget01 forget05 forget10"}
SEEDS=${SEEDS:-"0 1 2"}
RANKS=${RANKS:-"4 8 16 32"}

for METHOD in ${METHODS}; do
  for SPLIT in ${SPLITS}; do
    for SEED in ${SEEDS}; do
      for RANK in ${RANKS}; do
        python forget.py method="${METHOD}" split="${SPLIT}" seed="${SEED}" \
          LoRA.r="${RANK}" LoRA.alpha=$(( RANK * 2 ))
      done
    done
  done
done
