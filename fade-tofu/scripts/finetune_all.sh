#!/usr/bin/env bash
# Trains the full model and the 3 retain-only oracles, for each seed, on 4 GPUs
# (batch 1 x grad-accum 8 per GPU -> effective batch 32, as for the published models).
# Outputs land where the published checkpoints would (ckpts/full/seed*,
# ckpts/retain/retain*/seed*) and are then used in place of downloading them.
#
#   ./scripts/finetune_all.sh
#   SPLITS="retain99" SEEDS="0" NUM_GPUS=1 ./scripts/finetune_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SPLITS=${SPLITS:-"full retain99 retain95 retain90"}
SEEDS=${SEEDS:-"0 1 2"}
NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-28765}

for SEED in ${SEEDS}; do
  for SPLIT in ${SPLITS}; do
    torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
      finetune.py split="${SPLIT}" seed="${SEED}" \
      gradient_accumulation_steps=$(( 32 / NUM_GPUS ))
  done
done
