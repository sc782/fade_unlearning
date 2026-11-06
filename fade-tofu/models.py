"""
Model families, prompt formatting, and checkpoint resolution/loading.

Checkpoints are referred to by a *spec* that mirrors the layout of the published
HuggingFace collection (https://huggingface.co/collections/sungjuncho/fade-tofu-checkpoints):

    full/seed0                               full-data fine-tune (starting point for unlearning)
    retain/retain99/seed0                    retain-only oracle for forget01 (retain95 -> forget05, ...)
    <method>/r32/forget01/seed0/checkpoint-5 LoRA adapter on top of full/seed<k>, method in METHODS

A spec resolves to ckpts/<spec> locally; anything missing is downloaded from the
matching HF repo on first use. An existing local directory can also be passed directly
(a LoRA adapter directory then needs `base=` since its adapter_config.json records the
path it was trained from, not a portable one).
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import yaml

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_ROOT = os.path.join(REPO_ROOT, "ckpts")
HF_PREFIX = "sungjuncho/Llama-3.1-8B-fade-tofu"

# Unlearning methods, keyed by the short name used in checkpoint specs / the paper.
# `loss` is the forget objective implemented in trainers.py. "dpo" is the paper's name
# for SFT on "I don't know"-style refusals of the forget questions (not literal DPO).
METHODS = {
    "ga": {"loss": "grad_ascent", "hf_repo": f"{HF_PREFIX}-unlearned-ga"},
    "gd": {"loss": "grad_diff", "hf_repo": f"{HF_PREFIX}-unlearned-gd"},
    "npo": {"loss": "npo_grad_diff", "hf_repo": f"{HF_PREFIX}-unlearned-npo"},
    "dpo": {"loss": "idk", "hf_repo": f"{HF_PREFIX}-unlearned-dpo"},
    "ihl": {"loss": "grad_diff_hinge", "hf_repo": f"{HF_PREFIX}-unlearned-ihl"},
}

# forget split -> matching retain split and the final unlearning step (5 epochs of
# batch 2 x grad-accum 16 over 40/200/400 forget examples).
SPLITS = {
    "forget01": {"retain": "retain99", "final_step": 5},
    "forget05": {"retain": "retain95", "final_step": 30},
    "forget10": {"retain": "retain90", "final_step": 60},
}


def load_family(model_family):
    with open(os.path.join(REPO_ROOT, "config", "model_config.yaml")) as f:
        return yaml.safe_load(f)[model_family]


def format_prompt(question, family_cfg):
    """Text the answer is generated/scored after -- identical to the question half of the
    training examples (data.convert_raw_data_to_model_format)."""
    return (family_cfg["question_start_tag"] + question + family_cfg["question_end_tag"]
            + family_cfg["answer_tag"])


def load_tokenizer(model_family):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(load_family(model_family)["hf_key"])
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def terminal_token_ids(tokenizer, model_path):
    """Token ids that end an answer: the tokenizer's EOS (what training appends to every
    answer) plus any extra EOS ids the checkpoint's generation_config declares."""
    ids = {tokenizer.eos_token_id}
    gen_cfg = os.path.join(model_path, "generation_config.json")
    if os.path.exists(gen_cfg):
        with open(gen_cfg) as f:
            eos = json.load(f).get("eos_token_id")
        ids.update(eos if isinstance(eos, list) else [eos] if eos is not None else [])
    return sorted(ids)


@dataclass
class ModelRef:
    name: str                      # spec (or path) as given, used for bookkeeping
    path: str                      # full model weights (the base model, for an adapter)
    adapter: Optional[str] = None  # LoRA adapter directory, if any


def _hf_location(spec):
    """spec -> (hf_repo, subfolder, local_dir the repo is mirrored into)."""
    parts = spec.split("/")
    kind = parts[0]
    if kind == "full" and re.fullmatch(r"seed\d+", parts[1]) and len(parts) == 2:
        return f"{HF_PREFIX}-full", spec[len("full/"):], os.path.join(CKPT_ROOT, "full")
    if kind == "retain" and len(parts) == 3:
        return f"{HF_PREFIX}-retain", spec[len("retain/"):], os.path.join(CKPT_ROOT, "retain")
    if kind in METHODS and len(parts) == 5:
        return METHODS[kind]["hf_repo"], spec[len(kind) + 1:], os.path.join(CKPT_ROOT, kind)
    raise ValueError(f"not a checkpoint spec or existing directory: {spec!r}")


def ensure_downloaded(spec):
    """Download one checkpoint spec into ckpts/<spec> unless it is already there."""
    repo, subfolder, local_dir = _hf_location(spec)
    target = os.path.join(local_dir, subfolder)
    in_progress = os.path.join(target, ".downloading")
    present = any(os.path.exists(os.path.join(target, f))
                  for f in ("config.json", "adapter_config.json"))
    # A checkpoint you trained yourself at the same path is used as-is; only a missing
    # or interrupted download (marker left behind) is (re)fetched.
    if not present or os.path.exists(in_progress):
        from huggingface_hub import snapshot_download

        print(f"downloading {repo}/{subfolder} -> {target}")
        os.makedirs(target, exist_ok=True)
        open(in_progress, "w").close()
        snapshot_download(repo, allow_patterns=[f"{subfolder}/*"], local_dir=local_dir)
        os.remove(in_progress)
    return target


def resolve(spec, base=None, download=True):
    """Resolve a checkpoint spec or local directory to a ModelRef."""
    if os.path.isdir(spec):
        if os.path.exists(os.path.join(spec, "adapter_config.json")):
            if base is None:
                raise ValueError(f"{spec} is a LoRA adapter; pass its base model too")
            return ModelRef(spec, resolve(base, download=download).path, spec)
        return ModelRef(spec, spec)

    path = ensure_downloaded(spec) if download else os.path.join(CKPT_ROOT, spec)
    kind, *rest = spec.split("/")
    if kind in METHODS:
        seed = rest[2]  # <method>/r<rank>/<split>/seed<k>/checkpoint-<n>
        return ModelRef(spec, resolve(f"full/{seed}", download=download).path, path)
    return ModelRef(spec, path)


def load_hf_model(ref, device="cuda:0", attn_implementation="sdpa"):
    """Load a ModelRef as a plain (adapter-merged) bf16 HF causal LM in eval mode.

    bf16 to match generate.py's sampling: the estimators need the scoring model to be
    the distribution the answers were drawn from."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        ref.path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map=device,
    )
    if ref.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, ref.adapter).merge_and_unload()
    model.eval()
    return model
