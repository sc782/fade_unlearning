"""
Fine-tunes the base model on a TOFU split (full model or retain-only oracle).

Usage
-----
torchrun --nproc_per_node=4 finetune.py split=full seed=0
torchrun --nproc_per_node=4 finetune.py split=retain99 seed=0
python finetune.py split=full deepspeed=null       # single GPU, no DeepSpeed
"""

import os

import hydra
import torch
import transformers
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, set_seed

from data import TextDatasetQA, data_collator
from models import REPO_ROOT, load_family, load_tokenizer
from trainers import FinetuneTrainer


def repo_path(path):
    return path if path is None or os.path.isabs(path) else os.path.join(REPO_ROOT, path)


@hydra.main(version_base="1.3", config_path="config", config_name="finetune")
def main(cfg):
    set_seed(cfg.seed)
    os.environ["WANDB_DISABLED"] = "true"
    family = load_family(cfg.model_family)
    save_dir = repo_path(cfg.save_dir or os.path.join(
        "ckpts", "full" if cfg.split == "full" else f"retain/{cfg.split}", f"seed{cfg.seed}"))
    os.makedirs(save_dir, exist_ok=True)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        OmegaConf.save(cfg, os.path.join(save_dir, "cfg.yaml"))

    tokenizer = load_tokenizer(cfg.model_family)
    dataset = TextDatasetQA(cfg.data_path, tokenizer, family, cfg.max_length, cfg.split)

    num_devices = int(os.environ.get("WORLD_SIZE", 1))
    steps_per_epoch = len(dataset) / (cfg.batch_size * cfg.gradient_accumulation_steps * num_devices)
    max_steps = int(cfg.num_epochs * steps_per_epoch)
    print(f"{len(dataset)} examples, {num_devices} devices -> {max_steps} steps")

    args = transformers.TrainingArguments(
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=max(1, int(max_steps // cfg.num_epochs)),
        max_steps=max_steps,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        bf16=True,
        logging_steps=max(1, max_steps // 20),
        logging_dir=os.path.join(save_dir, "logs"),
        output_dir=save_dir,
        optim=cfg.optim,
        save_strategy="no",
        ddp_find_unused_parameters=False,
        deepspeed=repo_path(cfg.deepspeed),
        seed=cfg.seed,
        report_to=[],
    )

    model = AutoModelForCausalLM.from_pretrained(family["hf_key"], torch_dtype=torch.bfloat16)
    if family["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    trainer = FinetuneTrainer(model=model, args=args, train_dataset=dataset,
                              data_collator=data_collator)
    trainer.train()
    trainer.save_model(save_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
