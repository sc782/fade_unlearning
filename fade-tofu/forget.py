"""
Unlearns a TOFU forget split from a full-data model with a LoRA adapter, saving the
adapter after every epoch (checkpoint-<step>).

Usage
-----
python forget.py method=ga split=forget01 seed=0
python forget.py method=npo split=forget10 seed=2 LoRA.r=8 LoRA.alpha=16
"""

import os

import hydra
import torch
import transformers
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, set_seed

from data import TextForgetDatasetQA, forget_data_collator
from finetune import repo_path
from models import METHODS, load_family, load_tokenizer, resolve
from trainers import ForgetTrainer

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@hydra.main(version_base="1.3", config_path="config", config_name="forget")
def main(cfg):
    if cfg.method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}, got {cfg.method!r}")
    forget_loss = METHODS[cfg.method]["loss"]
    set_seed(cfg.seed)
    os.environ["WANDB_DISABLED"] = "true"
    family = load_family(cfg.model_family)
    save_dir = repo_path(cfg.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        OmegaConf.save(cfg, os.path.join(save_dir, "cfg.yaml"))

    tokenizer = load_tokenizer(cfg.model_family)
    dataset = TextForgetDatasetQA(cfg.data_path, tokenizer, family, cfg.max_length, cfg.split,
                                  loss_type=forget_loss)

    num_devices = int(os.environ.get("WORLD_SIZE", 1))
    steps_per_epoch = len(dataset) // (cfg.batch_size * cfg.gradient_accumulation_steps * num_devices)
    max_steps = cfg.num_epochs * len(dataset) // (cfg.batch_size * cfg.gradient_accumulation_steps * num_devices)
    print(f"{forget_loss} on {cfg.split}: {max_steps} steps, saving every {steps_per_epoch}")

    args = transformers.TrainingArguments(
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=max(1, steps_per_epoch),
        max_steps=max_steps,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        bf16=True,
        logging_steps=max(1, max_steps // 20),
        logging_dir=os.path.join(save_dir, "logs"),
        output_dir=save_dir,
        optim=cfg.optim,
        save_strategy="steps",
        save_steps=steps_per_epoch,
        save_only_model=True,
        ddp_find_unused_parameters=False,
        deepspeed=repo_path(cfg.deepspeed),
        seed=cfg.seed,
        report_to=[],
    )

    model = AutoModelForCausalLM.from_pretrained(resolve(cfg.model_path).path,
                                                 torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=cfg.LoRA.r,
        lora_alpha=cfg.LoRA.alpha,
        lora_dropout=cfg.LoRA.dropout,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    trainer = ForgetTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=forget_data_collator,
        forget_loss=forget_loss,
        beta=cfg.beta,
        npo_coeff=cfg.npo_coeff,
        grad_diff_coeff=cfg.grad_diff_coeff,
    )
    trainer.train()


if __name__ == "__main__":
    main()
