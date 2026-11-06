# fade-tofu

FADE on [TOFU](https://locuslab.github.io/tofu/) (Task of Fictitious Unlearning, for
language models): how far a Llama-3.1-8B that has unlearned a set of fictitious
authors is, in distribution, from a retain-only oracle that was never trained on them
-- measured on answers to the forget questions sampled from each. See the
[top-level README](../README.md) for FADE in general.

## The FADE metric

For a forget split with questions x, retain-only oracle r and evaluated model u:

```
term1 = E_x KL( r(.|x) || u(.|x) )      estimated on answers sampled from r
term2 = E_x KL( u(.|x) || r(.|x) )      estimated on answers sampled from u
FADE  = term1 + term2                   (symmetric KL, in nats per answer)
```

Each term is the mean over forget questions of the mean over that question's sampled
answers (10 by default; `--num_samples` / `NUM_SAMPLES=` to change). The per-answer KL
is estimated with the **Rao-Blackwellized** estimator of
[Amini, Vieira & Cotterell (2025)](https://arxiv.org/abs/2504.10637): for an answer
y ~ p, sum over positions of the *exact* next-token KL between the two models given the
sampled prefix,

```
RB(y) = sum_n KL( p(.|x, y_<n) || q(.|x, y_<n) )      (full 128k vocabulary, incl. EOS)
```

rather than the plain Monte Carlo log-ratio `log p(y|x) - log q(y|x)`. Both are
unbiased; RB has variance no larger than MC and is non-negative by construction (every
term is a KL), whereas MC estimates can go negative for heavily unlearned models. The
MC estimate comes out of the same forward passes and is stored alongside (`fade_mc`).
See `estimators.py`; `tests/test_estimators.py` checks both against the exact KL of
toy models whose answers can be enumerated.

Two things the estimators rely on, both handled by `generate.py`:

- **Answers are sampled from the model itself** -- temperature 1, no top-p/top-k (the
  `temperature 0.6, top_p 0.9` in the checkpoints' `generation_config.json` are not
  used). With truncated sampling neither estimator targets the model's KL.
- **Answers are scored exactly as sampled**: samples are stored as token ids together
  with the prompt ids they followed, so there is no decode/re-tokenize round trip. The
  prompt is `[INST] {question} [/INST]`, identical to the training format, and an
  answer includes its terminal `<|eot_id|>` (that step's KL counts). Answers that hit
  `--max_length` (200 tokens incl. prompt) are scored as truncated. Sampling and
  scoring are both in bf16, so the scored distribution is the sampled one.

## Checkpoints

Everything FADE needs is published in the
**[FADE TOFU Checkpoints](https://huggingface.co/collections/sungjuncho/fade-tofu-checkpoints)**
collection and downloaded on first use. Models are named by *specs* mirroring that
layout (and the local `ckpts/` directory):

| spec | model |
|---|---|
| `full/seed{0,1,2}` | fine-tuned on all of TOFU (the starting point for unlearning) |
| `retain/{retain99,retain95,retain90}/seed{0,1,2}` | retain-only oracles for forget01 / forget05 / forget10 |
| `{ga,gd,npo,dpo,ihl}/r{4,8,16,32}/{split}/seed{k}/checkpoint-{step}` | LoRA adapter on `full/seed{k}`, saved after each of 5 epochs |

Methods: **ga** gradient ascent, **gd** gradient difference, **npo** NPO + retain
loss, **dpo** SFT on "I don't know" refusals of the forget questions (the paper's
"DPO" baseline -- not literal DPO), **ihl** inverted hinge loss + retain loss. Final
steps are 5 / 30 / 60 for forget01 / 05 / 10.

`python scripts/download_checkpoints.py` pre-fetches what the full pipeline uses
(~200GB; `--help` for subsets). A checkpoint you train yourself at the same path is
used as-is.

## Quickstart

```bash
conda env create -f environment.yaml && conda activate fade_tofu

./scripts/generate_all.sh      # sample 10 answers/question from every model (vLLM)
./scripts/fade_all.sh          # FADE for every (retain oracle, model) pair + summary table
```

Both skip work already done, and take `SPLITS=`, `SEEDS=`, `METHODS=`, `RANK=`
environment overrides (e.g. `SPLITS=forget01` for a quick run). The full run takes
about 2.5 hours on one 96GB GPU (sampling ~1.3h, FADE ~1.2h), plus checkpoint
download. For each split and seed, FADE is computed against `retain/<split's retain>/seed{k}` for: another
retain seed (`seed{k+1 mod 3}`, the noise floor between two models that never saw the
forget set), the full model before unlearning, and each method's final rank-32
adapter. Results: `outputs/fade/<split>/*.json` (incl. per-question and per-sample
values) and `outputs/fade_results.csv`.

Single runs:

```bash
python generate.py --model retain/retain99/seed0 --split forget01
python generate.py --model npo/r32/forget01/seed0/checkpoint-5 --split forget01
python compute_fade.py --split forget01 --model_r retain/retain99/seed0 \
    --model_u npo/r32/forget01/seed0/checkpoint-5
```

`compute_fade.py` holds both models in bf16 on one GPU (~33GB plus activations).

## Results

RB FADE (nats), mean +- std over 3 seeds, as produced by the Quickstart above:

| | forget01 | forget05 | forget10 |
|---|---|---|---|
| retain vs retain (another seed) | 25.4 +- 0.3 | 21.7 +- 0.4 | 21.3 +- 0.2 |
| full (before unlearning) | 125.6 +- 3.2 | 125.5 +- 1.4 | 129.8 +- 0.5 |
| ga | 110.9 +- 0.5 | 2213.5 +- 464.1 | 5285.0 +- 437.4 |
| gd | 117.8 +- 5.0 | 337.5 +- 107.0 | 6674.8 +- 4819.2 |
| npo | 109.6 +- 1.2 | 309.6 +- 111.0 | 335.4 +- 11.5 |
| dpo | 114.5 +- 3.4 | 76.4 +- 1.4 | 89.7 +- 1.4 |
| ihl | 109.7 +- 4.5 | 123.7 +- 13.6 | 1022.9 +- 507.6 |

Lower is closer to the retain-only oracle; the retain-vs-retain row is the floor two
models that never saw the forget set reach. Per-seed values, both terms and the MC
estimates are in `outputs/fade_results.csv`.

## Training from scratch

```bash
./scripts/finetune_all.sh      # full + retain99/95/90 models, 3 seeds, 4 GPUs (DeepSpeed ZeRO-3)
./scripts/forget_all.sh        # 5 methods x 3 splits x 3 seeds x 4 LoRA ranks, 1 GPU each
```

or individually, e.g. `torchrun --nproc_per_node=4 finetune.py split=retain99 seed=0`
and `python forget.py method=npo split=forget01 seed=0 LoRA.r=32 LoRA.alpha=64`
(Hydra overrides of `config/finetune.yaml` / `config/forget.yaml`).

## Files

| file | |
|---|---|
| `estimators.py` | RB and MC KL estimators |
| `generate.py` | sample answers to forget questions (vLLM) |
| `compute_fade.py` | FADE between a retain oracle and one or more models |
| `aggregate.py` | collect results into a table |
| `finetune.py`, `forget.py` | training; `trainers.py` has the 5 unlearning losses, `data.py` the TOFU datasets |
| `models.py` | prompt format, checkpoint specs, download, loading |
| `scripts/` | full pipelines (sampling, FADE, training) and checkpoint download |
| `tests/` | estimator tests and an end-to-end smoke test |

## Tests

```bash
python -m pytest tests/test_estimators.py   # RB and MC vs the exact KL of enumerable toy models
./tests/smoke_test.sh                        # training, sampling and FADE end to end on a 135M model
```

## Dataset

The TOFU questions and answers are loaded from HuggingFace
([`locuslab/TOFU`](https://huggingface.co/datasets/locuslab/TOFU)) on first use; there
is nothing to download by hand.

## License

Code is MIT licensed -- see [LICENSE](../LICENSE). The published checkpoints are
fine-tunes of Llama-3.1-8B-Instruct and are subject to the
[Llama 3.1 Community License](https://www.llama.com/llama3_1/license/) regardless of
the code license.

## Credits

The fine-tuning and unlearning code builds on the
[TOFU codebase](https://github.com/locuslab/tofu) (Maini et al.); the RB estimator is
from [Amini, Vieira & Cotterell (2025)](https://arxiv.org/abs/2504.10637).

## Citation

```
@article{cho2025referencespecific,
  title={Reference-Specific Unlearning Metrics Can Hide the Truth: A Reality Check},
  author={Cho, Sungjun and Hwang, Dasol and Sala, Frederic and Hwang, Sangheum and Cho, Kyunghyun and Cha, Sungmin},
  journal={arXiv preprint arXiv:2510.12981},
  year={2025}
}

@article{maini2024tofu,
  title={TOFU: A Task of Fictitious Unlearning for LLMs},
  author={Maini, Pratyush and Feng, Zhili and Schwarzschild, Avi and Lipton, Zachary C. and Kolter, J. Zico},
  journal={arXiv preprint arXiv:2401.06121},
  year={2024}
}
```
