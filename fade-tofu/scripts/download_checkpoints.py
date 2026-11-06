"""
Downloads FADE TOFU checkpoints from HuggingFace into ckpts/, laid out exactly like the
collection (https://huggingface.co/collections/sungjuncho/fade-tofu-checkpoints), so a
spec like `retain/retain99/seed0` means ckpts/retain/retain99/seed0 everywhere.

Every other script downloads what it needs on first use (models.resolve), so this is
only for fetching ahead of time (e.g. before going offline).

Usage
-----
python scripts/download_checkpoints.py                         # what scripts/*_all.sh use (~200GB)
python scripts/download_checkpoints.py --splits forget01       # just one split
python scripts/download_checkpoints.py --only full retain      # skip the LoRA adapters
python scripts/download_checkpoints.py --ranks 4 8 16 32 --all_steps   # every published adapter
"""

import os
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models import METHODS, SPLITS, ensure_downloaded  # noqa: E402


def specs(only, splits, seeds, methods, ranks, all_steps):
    for seed in seeds:
        if "full" in only:
            yield f"full/seed{seed}"
        for split in splits:
            if "retain" in only:
                yield f"retain/{SPLITS[split]['retain']}/seed{seed}"
            if "unlearned" in only:
                final = SPLITS[split]["final_step"]
                steps = range(final // 5, final + 1, final // 5) if all_steps else [final]
                for method in methods:
                    for rank in ranks:
                        for step in steps:
                            yield f"{method}/r{rank}/{split}/seed{seed}/checkpoint-{step}"


def main():
    parser = ArgumentParser()
    parser.add_argument("--only", nargs="+", default=["full", "retain", "unlearned"],
                        choices=["full", "retain", "unlearned"])
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--ranks", nargs="+", type=int, default=[32])
    parser.add_argument("--all_steps", action="store_true",
                        help="every saved epoch (5 per run), not just the final one")
    args = parser.parse_args()

    for spec in dict.fromkeys(specs(args.only, args.splits, args.seeds, args.methods,
                                    args.ranks, args.all_steps)):
        print(ensure_downloaded(spec))


if __name__ == "__main__":
    main()
