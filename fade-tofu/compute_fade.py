"""
Computes FADE between a retain-only oracle r and an (unlearned) model u on a TOFU
forget split, from answers sampled by each (generate.py):

    term1 = KL(r || u)   estimated on r's samples
    term2 = KL(u || r)   estimated on u's samples
    FADE  = term1 + term2   (symmetric KL)

Each term is the mean over forget questions of the per-question mean over that
question's sampled answers, each estimated with the Rao-Blackwellized estimator (see
estimators.py). The plain Monte Carlo estimate comes out of the same forward passes and
is stored alongside as fade_mc.

Usage
-----
python compute_fade.py --split forget01 --model_r retain/retain99/seed0 \
    --model_u ga/r32/forget01/seed0/checkpoint-5 full/seed0 retain/retain99/seed1
python compute_fade.py --split forget01 --model_r retain/retain99/seed0 --model_u full/seed0

Samples default to generate.py's output paths; results go to
outputs/fade/<split>/<model_r>__vs__<model_u>.json (with '/' -> '_'), including
per-sample values for variance analysis.
"""

import json
import os
from argparse import ArgumentParser

import numpy as np
import torch

from estimators import sequence_kl
from generate import default_out as default_samples
from models import REPO_ROOT, load_hf_model, load_tokenizer, resolve


def load_samples(path, split, model):
    with open(path) as f:
        samples = json.load(f)
    if samples["split"] != split:
        raise ValueError(f"{path} holds {samples['split']} samples, not {split}")
    if samples["model"] != model:
        print(f"note: {path} was sampled from {samples['model']!r}, scoring it as {model!r}")
    if samples["sampling"]["temperature"] != 1.0 or samples["sampling"]["top_p"] != 1.0:
        print(f"WARNING: {path} was not sampled at temperature 1 / top_p 1, so neither "
              "estimator targets the model's KL on it")
    return samples


def estimate(model_p, model_q, samples, pad_token_id, max_tokens_per_batch):
    """KL(p || q) on samples from p: per-question means, then the mean over questions."""
    prompts, answers, owner = [], [], []
    for qi, record in enumerate(samples["questions"]):
        for answer in record["answers"]:
            prompts.append(record["prompt_ids"])
            answers.append(answer)
            owner.append(qi)
    rb, mc = sequence_kl(model_p, model_q, prompts, answers, pad_token_id, max_tokens_per_batch)
    owner, rb, mc = np.array(owner), np.array(rb), np.array(mc)
    n_q = len(samples["questions"])
    rb_q = np.array([rb[owner == qi].mean() for qi in range(n_q)])
    mc_q = np.array([mc[owner == qi].mean() for qi in range(n_q)])
    return {
        "rb": float(rb_q.mean()),
        "mc": float(mc_q.mean()),
        "rb_per_question": rb_q.tolist(),
        "mc_per_question": mc_q.tolist(),
        "rb_per_sample": rb.tolist(),
        "mc_per_sample": mc.tolist(),
        "question_of_sample": owner.tolist(),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--model_r", required=True, help="retain-only oracle (spec or directory)")
    parser.add_argument("--model_u", required=True, nargs="+",
                        help="model(s) to evaluate against model_r, scored one after another "
                             "with model_r kept loaded")
    parser.add_argument("--base_r", default=None, help="base model, if --model_r is a local adapter")
    parser.add_argument("--base_u", default=None, help="base model, if --model_u is a local adapter")
    parser.add_argument("--samples_r", default=None)
    parser.add_argument("--samples_u", default=None, help="only with a single --model_u")
    parser.add_argument("--model_family", default="llama3.1-8b")
    parser.add_argument("--max_tokens_per_batch", type=int, default=16384)
    parser.add_argument("--out", default=None, help="only with a single --model_u")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()
    if len(args.model_u) > 1 and (args.samples_u or args.out):
        parser.error("--samples_u/--out need a single --model_u")

    outs = {}
    for model_u in args.model_u:
        name = f"{args.model_r}__vs__{model_u}".replace("/", "_")
        outs[model_u] = args.out or os.path.join(REPO_ROOT, "outputs", "fade", args.split, f"{name}.json")
    todo = [u for u in args.model_u if not (args.skip_existing and os.path.exists(outs[u]))]
    if not todo:
        return

    samples_r = load_samples(args.samples_r or default_samples(args.model_r, args.split),
                             args.split, args.model_r)
    tokenizer = load_tokenizer(args.model_family)
    model_r = load_hf_model(resolve(args.model_r, base=args.base_r))

    for model_u_spec in todo:
        samples_u = load_samples(args.samples_u or default_samples(model_u_spec, args.split),
                                 args.split, model_u_spec)
        model_u = load_hf_model(resolve(model_u_spec, base=args.base_u))
        term1 = estimate(model_r, model_u, samples_r, tokenizer.pad_token_id, args.max_tokens_per_batch)
        term2 = estimate(model_u, model_r, samples_u, tokenizer.pad_token_id, args.max_tokens_per_batch)
        del model_u
        torch.cuda.empty_cache()

        result = {
            "split": args.split,
            "model_r": args.model_r,
            "model_u": model_u_spec,
            "samples_r": samples_r["sampling"],
            "samples_u": samples_u["sampling"],
            "fade": term1["rb"] + term2["rb"],
            "fade_mc": term1["mc"] + term2["mc"],
            "term1": term1,
            "term2": term2,
        }
        out = outs[model_u_spec]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + ".tmp", "w") as f:  # rename after, so --skip_existing never sees a partial file
            json.dump(result, f)
        os.replace(out + ".tmp", out)
        print(f"{args.split} r={args.model_r} u={model_u_spec}\n"
              f"  RB: term1 {term1['rb']:.3f}  term2 {term2['rb']:.3f}  FADE {result['fade']:.3f}\n"
              f"  MC: term1 {term1['mc']:.3f}  term2 {term2['mc']:.3f}  FADE {result['fade_mc']:.3f}\n"
              f"  -> {out}", flush=True)


if __name__ == "__main__":
    main()
