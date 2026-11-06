#!/usr/bin/env bash
# Computes FADE against each retain-only oracle (per split and seed) for:
#   - another retain seed (seed k vs k+1 mod 3): the noise floor between two models
#     that never saw the forget set
#   - the full model before unlearning
#   - each method's final rank-32 adapter
# Needs the samples from generate_all.sh. Existing results are skipped.
#
#   ./scripts/fade_all.sh
#   SPLITS="forget01" ./scripts/fade_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SPLITS=${SPLITS:-"forget01 forget05 forget10"}
SEEDS=${SEEDS:-"0 1 2"}
METHODS=${METHODS:-"ga gd npo dpo ihl"}
RANK=${RANK:-32}

declare -A RETAIN=([forget01]=retain99 [forget05]=retain95 [forget10]=retain90)
declare -A FINAL_STEP=([forget01]=5 [forget05]=30 [forget10]=60)

for SPLIT in ${SPLITS}; do
  for SEED in ${SEEDS}; do
    R="retain/${RETAIN[$SPLIT]}/seed${SEED}"
    US="retain/${RETAIN[$SPLIT]}/seed$(( (SEED + 1) % 3 )) full/seed${SEED}"
    for METHOD in ${METHODS}; do
      US="${US} ${METHOD}/r${RANK}/${SPLIT}/seed${SEED}/checkpoint-${FINAL_STEP[$SPLIT]}"
    done
    # shellcheck disable=SC2086
    python compute_fade.py --split "${SPLIT}" --model_r "${R}" --model_u ${US} --skip_existing
  done
done

python aggregate.py
