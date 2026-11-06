"""
TOFU datasets for fine-tuning (full / retain-only models) and unlearning.
"""

import os

import datasets
import torch
from torch.utils.data import Dataset

from models import REPO_ROOT, format_prompt

IDK_FILE = os.path.join(REPO_ROOT, "data", "idontknow.jsonl")


def convert_raw_data_to_model_format(tokenizer, max_length, question, answer, family_cfg):
    """(input_ids, labels, attention_mask) for one QA pair, padded to max_length, with
    the question masked out of the labels and EOS appended as the answer's last label."""
    prompt = format_prompt(question, family_cfg)
    num_question_tokens = len(tokenizer.tokenize(prompt, add_special_tokens=True))
    encoded = tokenizer(prompt + answer, add_special_tokens=True, max_length=max_length,
                        truncation=True)
    pad_length = max_length - len(encoded.input_ids)
    input_ids = encoded.input_ids + [tokenizer.eos_token_id] * pad_length
    attention_mask = encoded.attention_mask + [0] * pad_length
    if pad_length == 0:
        labels = list(encoded.input_ids)
    else:
        labels = encoded.input_ids + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)
    for i in range(num_question_tokens):
        labels[i] = -100
    return torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attention_mask)


class TextDatasetQA(Dataset):
    """One TOFU split (e.g. full, retain90) as plain supervised QA examples."""

    def __init__(self, data_path, tokenizer, family_cfg, max_length=500, split="full"):
        self.data = datasets.load_dataset(data_path, split)["train"]
        self.tokenizer = tokenizer
        self.family_cfg = family_cfg
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return convert_raw_data_to_model_format(self.tokenizer, self.max_length, row["question"],
                                                row["answer"], self.family_cfg)


class TextForgetDatasetQA(Dataset):
    """
    Pairs each forget example with a random retain example. For loss "idk" the forget
    question is paired with a random refusal from data/idontknow.jsonl instead of its
    real answer.
    """

    def __init__(self, data_path, tokenizer, family_cfg, max_length=500, split="forget10",
                 loss_type="grad_ascent"):
        self.forget_data = datasets.load_dataset(data_path, split)["train"]
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        self.tokenizer = tokenizer
        self.family_cfg = family_cfg
        self.max_length = max_length
        self.idk = None
        if loss_type == "idk":
            with open(IDK_FILE) as f:
                self.idk = [line.strip() for line in f]

    def __len__(self):
        return len(self.forget_data)

    def _convert(self, question, answer):
        return convert_raw_data_to_model_format(self.tokenizer, self.max_length, question,
                                                answer, self.family_cfg)

    def __getitem__(self, idx):
        forget = self.forget_data[idx]
        answer = forget["answer"]
        if self.idk is not None:
            answer = self.idk[torch.randint(0, len(self.idk), (1,)).item()]
        retain_idx = (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
        retain = self.retain_data[retain_idx]
        return [self._convert(forget["question"], answer),
                self._convert(retain["question"], retain["answer"])]


def data_collator(samples):
    return tuple(torch.stack([s[k] for s in samples]) for k in range(3))


def forget_data_collator(samples):
    return [data_collator([s[0] for s in samples]), data_collator([s[1] for s in samples])]
