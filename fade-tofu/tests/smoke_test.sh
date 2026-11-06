#!/usr/bin/env bash
# End-to-end smoke test of training, sampling and FADE on a 135M model (a few minutes
# on one GPU): fine-tune briefly, run every unlearning method for 2 steps (the first is
# LR warmup), then sample and compute FADE between the fine-tuned model and one
# unlearned adapter.
# Checks that the code paths run, not that any number is meaningful.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-$(mktemp -d)}
COMMON="model_family=smoke-test deepspeed=null"
echo "smoke test outputs in ${OUT}"

python finetune.py ${COMMON} num_epochs=0.02 batch_size=4 gradient_accumulation_steps=2 \
  save_dir="${OUT}/full"
for METHOD in ga gd npo dpo ihl; do
  python forget.py ${COMMON} method="${METHOD}" split=forget01 num_epochs=2 \
    model_path="${OUT}/full" LoRA.r=8 LoRA.alpha=16 save_dir="${OUT}/${METHOD}"
done

for MODEL in full npo/checkpoint-2; do
  python generate.py --model_family smoke-test --model "${OUT}/${MODEL}" --base "${OUT}/full" \
    --split forget01 --num_samples 4 --gpu_memory_utilization 0.3 --out "${OUT}/samples/${MODEL}.json"
done
python compute_fade.py --model_family smoke-test --split forget01 \
  --model_r "${OUT}/full" --model_u "${OUT}/npo/checkpoint-2" --base_u "${OUT}/full" \
  --samples_r "${OUT}/samples/full.json" --samples_u "${OUT}/samples/npo/checkpoint-2.json" \
  --out "${OUT}/fade.json"
python -m pytest -q tests/test_estimators.py
