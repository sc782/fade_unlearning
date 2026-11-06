#!/usr/bin/env bash
# Samples forget-question answers from every model FADE is computed on: per split and
# seed, the retain-only oracle, the full model, and each method's final rank-32
# adapter. Existing sample files are skipped, so this can be re-run to resume.
#
#   ./scripts/generate_all.sh                       # all splits
#   SPLITS="forget01" ./scripts/generate_all.sh     # one split
#   METHODS="ga npo" RANK=8 ./scripts/generate_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SPLITS=${SPLITS:-"forget01 forget05 forget10"}
SEEDS=${SEEDS:-"0 1 2"}
METHODS=${METHODS:-"ga gd npo dpo ihl"}
RANK=${RANK:-32}
NUM_SAMPLES=${NUM_SAMPLES:-10}

declare -A RETAIN=([forget01]=retain99 [forget05]=retain95 [forget10]=retain90)
declare -A FINAL_STEP=([forget01]=5 [forget05]=30 [forget10]=60)

for SPLIT in ${SPLITS}; do
  for SEED in ${SEEDS}; do
    MODELS="retain/${RETAIN[$SPLIT]}/seed${SEED} full/seed${SEED}"
    for METHOD in ${METHODS}; do
      MODELS="${MODELS} ${METHOD}/r${RANK}/${SPLIT}/seed${SEED}/checkpoint-${FINAL_STEP[$SPLIT]}"
    done
    for MODEL in ${MODELS}; do
      if [ -f "outputs/samples/${MODEL}/${SPLIT}.json" ]; then
        echo "skip ${MODEL} ${SPLIT} (samples exist)"
        continue
      fi
      python generate.py --model "${MODEL}" --split "${SPLIT}" --num_samples "${NUM_SAMPLES}"
    done
  done
done
