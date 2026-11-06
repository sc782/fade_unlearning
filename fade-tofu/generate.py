"""
Samples answers to the TOFU forget questions from one model, for FADE.

Answers are saved as token ids together with the exact prompt ids they were sampled
after, so compute_fade.py scores precisely the sequences that were sampled (no
decode/re-tokenize round trip). Sampling is from the model's own distribution
(temperature 1, no top-p/top-k truncation) by default, because both FADE estimators
assume samples drawn from the model being scored; the temperature 0.6 / top-p 0.9 in
the checkpoints' generation_config.json are not used. Override only for sensitivity
checks.

Usage
-----
python generate.py --model retain/retain99/seed0 --split forget01
python generate.py --model ga/r32/forget01/seed0/checkpoint-5 --split forget01
python generate.py --model path/to/adapter --base full/seed0 --split forget01 --out samples.json

Output: outputs/samples/<model spec>/<split>.json by default.
"""

import json
import os
from argparse import ArgumentParser

from datasets import load_dataset

from models import REPO_ROOT, format_prompt, load_family, load_tokenizer, resolve, terminal_token_ids


def default_out(model, split):
    return os.path.join(REPO_ROOT, "outputs", "samples", model.strip("/"), f"{split}.json")


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, help="checkpoint spec or local directory")
    parser.add_argument("--base", default=None, help="base model, if --model is a local LoRA adapter")
    parser.add_argument("--split", required=True, help="TOFU forget split, e.g. forget01")
    parser.add_argument("--model_family", default="llama3.1-8b")
    parser.add_argument("--num_samples", type=int, default=10, help="answers per question")
    parser.add_argument("--max_length", type=int, default=200,
                        help="max prompt + answer tokens; longer answers are truncated")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    out = args.out or default_out(args.model, args.split)
    family = load_family(args.model_family)
    tokenizer = load_tokenizer(args.model_family)
    ref = resolve(args.model, base=args.base)
    terminals = terminal_token_ids(tokenizer, ref.path)

    questions = load_dataset("locuslab/TOFU", args.split)["train"]["question"]
    prompt_ids = [tokenizer(format_prompt(q, family), add_special_tokens=True).input_ids
                  for q in questions]

    lora_rank = 0
    if ref.adapter is not None:
        with open(os.path.join(ref.adapter, "adapter_config.json")) as f:
            lora_rank = json.load(f)["r"]
    llm = LLM(
        model=ref.path,
        tokenizer=family["hf_key"],
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=max(args.max_length, 512),
        enable_lora=ref.adapter is not None,
        max_lora_rank=max(lora_rank, 8),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=1024,
        generation_config="vllm",  # ignore the checkpoint's temperature/top_p defaults
    )
    params = [
        SamplingParams(
            n=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=-1,
            max_tokens=args.max_length - len(ids),
            stop_token_ids=terminals,
            detokenize=False,
        )
        for ids in prompt_ids
    ]  # no per-request seeds: they force vLLM's much slower per-row sampler; LLM(seed=) suffices
    lora = LoRARequest("adapter", 1, ref.adapter) if ref.adapter is not None else None
    results = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids],
                           params, lora_request=lora)

    records = []
    for question, ids, result in zip(questions, prompt_ids, results):
        answers = []
        for completion in result.outputs:
            answer = list(completion.token_ids)
            # vLLM keeps a stop token in token_ids; make sure every answer that stopped
            # ends on its terminal token, since that token's step is part of the KL.
            if completion.finish_reason == "stop" and (not answer or answer[-1] not in terminals):
                answer.append(completion.stop_reason if isinstance(completion.stop_reason, int)
                              else tokenizer.eos_token_id)
            answers.append(answer)
        records.append({"question": question, "prompt_ids": ids, "answers": answers})

    os.makedirs(os.path.dirname(out), exist_ok=True)
    # write-then-rename so an interrupted run never leaves a partial file that
    # generate_all.sh would then skip
    with open(out + ".tmp", "w") as f:
        json.dump({
            "model": args.model,
            "split": args.split,
            "model_family": args.model_family,
            "sampling": {"num_samples": args.num_samples, "max_length": args.max_length,
                         "temperature": args.temperature, "top_p": args.top_p,
                         "seed": args.seed, "terminal_token_ids": terminals},
            "questions": records,
        }, f)
    os.replace(out + ".tmp", out)
    n_trunc = sum(a[-1] not in terminals for r in records for a in r["answers"])
    print(f"wrote {out}: {len(records)} questions x {args.num_samples} samples "
          f"({n_trunc} answers hit max_length)")


if __name__ == "__main__":
    main()
