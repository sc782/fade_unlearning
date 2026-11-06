"""
Downloads FADE UnlearnCanvas checkpoints from HuggingFace and converts them into the
pickle {"state_dict": ...} format every loading call site in this codebase expects
(compute_fade.py, sample_compvis_automated.py, the
unlearning methods' own train/eval scripts, etc.) -- so downloaded checkpoints behave
exactly as if you had trained/generated them yourself, with no code changes needed.

Collection: https://huggingface.co/collections/sungjuncho/fade-unlearncanvas-checkpoints-6a61761b36b8f53ae21b57b8

Usage
-----
python scripts/download_checkpoints.py                 # everything
python scripts/download_checkpoints.py --only retain    # just the 3 retain seeds
python scripts/download_checkpoints.py --only unlearned # just the 15 unlearned checkpoints
python scripts/download_checkpoints.py --only full      # just the starting-point checkpoint
"""

import os
from argparse import ArgumentParser

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# repo root is the parent of diffusion_model_finetuning/scripts/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MANIFEST = {
    "full": [
        {
            "hf_repo": "sungjuncho/fade-unlearncanvas-full",
            "hf_file": "full.safetensors",
            "local_path": "ckpts/diffusion/compvis/style50/compvis.ckpt",
        },
    ],
    "retain": [
        {
            "hf_repo": f"sungjuncho/fade-unlearncanvas-retain-seed{seed}",
            "hf_file": f"retain_seed{seed}.safetensors",
            "local_path": f"diffusion_model_finetuning/logs/retain_seed{seed}.ckpt",
        }
        for seed in (0, 1, 2)
    ],
    "unlearned": [
        {
            "hf_repo": "sungjuncho/fade-unlearncanvas-unlearned-ca",
            "hf_file": f"ca_{style_lower}.safetensors",
            "local_path": f"machine_unlearning/mu_concept_ablation_ca/logs/ca_{style_lower}/checkpoints/last-v1.ckpt",
        }
        for style_lower in ("monet", "picasso", "van_gogh")
    ] + [
        {
            "hf_repo": "sungjuncho/fade-unlearncanvas-unlearned-ediff",
            "hf_file": f"ediff_{style_lower}.safetensors",
            "local_path": f"machine_unlearning/mu_erase_diff_ediff/results/{style}/sd.ckpt",
        }
        for style_lower, style in (("monet", "Monet"), ("picasso", "Picasso"), ("van_gogh", "Van_Gogh"))
    ] + [
        {
            "hf_repo": "sungjuncho/fade-unlearncanvas-unlearned-esd",
            "hf_file": f"esd_{style_lower}.safetensors",
            "local_path": f"machine_unlearning/mu_erasing_concept_esd/results/style50/{style}.pth",
        }
        for style_lower, style in (("monet", "Monet"), ("picasso", "Picasso"), ("van_gogh", "Van_Gogh"))
    ] + [
        {
            "hf_repo": "sungjuncho/fade-unlearncanvas-unlearned-salun",
            "hf_file": f"salun_{style_lower}.safetensors",
            "local_path": f"machine_unlearning/mu_saliency_unlearn_salun/results/style50/{style}/sd_0.4.ckpt",
        }
        for style_lower, style in (("monet", "Monet"), ("picasso", "Picasso"), ("van_gogh", "Van_Gogh"))
    ] + [
        {
            "hf_repo": "sungjuncho/fade-unlearncanvas-unlearned-shs",
            "hf_file": f"shs_{style_lower}.safetensors",
            "local_path": f"machine_unlearning/mu_scissorhands_shs/results/{style}/sd.ckpt",
        }
        for style_lower, style in (("monet", "Monet"), ("picasso", "Picasso"), ("van_gogh", "Van_Gogh"))
    ],
}


def fetch_and_convert(entry):
    local_path = os.path.join(REPO_ROOT, entry["local_path"])
    if os.path.exists(local_path):
        print(f"  already present, skipping: {entry['local_path']}")
        return

    print(f"  downloading {entry['hf_repo']}/{entry['hf_file']} ...")
    safetensors_path = hf_hub_download(repo_id=entry["hf_repo"], filename=entry["hf_file"])
    state_dict = load_file(safetensors_path)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    torch.save({"state_dict": state_dict}, local_path)
    print(f"  saved {entry['local_path']} ({os.path.getsize(local_path) / 1e9:.2f} GB)")


def main():
    parser = ArgumentParser()
    parser.add_argument("--only", choices=["full", "retain", "unlearned"], default=None,
                        help="download only this category (default: everything)")
    args = parser.parse_args()

    categories = [args.only] if args.only else list(MANIFEST.keys())
    for category in categories:
        print(f"=== {category} ===")
        for entry in MANIFEST[category]:
            fetch_and_convert(entry)


if __name__ == "__main__":
    main()
