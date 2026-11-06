"""
Collects compute_fade.py results into the FADE table.

    python aggregate.py            # reads output_fade/<theme>/*.pt

For each forget style and each model X with both directions against the reference
retain-only oracle r on disk,

    term1 = E_{x ~ r}[D(r, X; x)]   (output_fade/<theme>/<r>_vs_<X>.pt)
    term2 = E_{x ~ X}[D(X, r; x)]   (output_fade/<theme>/<X>_vs_<r>.pt)
    FADE  = term1 + term2

in nats per latent dimension. Writes <fade_dir>/fade_results.csv and prints one table per
style (X = another retain seed gives the floor).
"""

import csv
import glob
import os
from argparse import ArgumentParser

import torch

FIELDS = ["theme", "model", "term1", "term2", "fade", "n_term1", "n_term2"]
ORDER = ["retain1", "retain2", "full", "CA", "EDiff", "ESD", "SalUn", "SHS"]


def load_diffs(path):
    return torch.load(path, map_location="cpu")["diffs"].double()


def main():
    parser = ArgumentParser()
    parser.add_argument("--fade_dir", default="output_fade")
    parser.add_argument("--reference", default="retain0")
    args = parser.parse_args()
    ref = args.reference

    rows = []
    for theme_dir in sorted(glob.glob(os.path.join(args.fade_dir, "*", ""))):
        theme = os.path.basename(os.path.dirname(theme_dir))
        models = {os.path.basename(p)[len(ref) + 4:-3]
                  for p in glob.glob(os.path.join(theme_dir, f"{ref}_vs_*.pt"))}
        models = [m for m in models if os.path.exists(os.path.join(theme_dir, f"{m}_vs_{ref}.pt"))]
        for model in sorted(models, key=lambda m: (ORDER.index(m) if m in ORDER else len(ORDER), m)):
            d1 = load_diffs(os.path.join(theme_dir, f"{ref}_vs_{model}.pt"))
            d2 = load_diffs(os.path.join(theme_dir, f"{model}_vs_{ref}.pt"))
            rows.append({"theme": theme, "model": model,
                         "term1": d1.mean().item(), "term2": d2.mean().item(),
                         "fade": (d1.mean() + d2.mean()).item(),
                         "n_term1": len(d1), "n_term2": len(d2)})

    out = os.path.join(args.fade_dir, "fade_results.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} results -> {out}\n")

    for theme in dict.fromkeys(row["theme"] for row in rows):
        print(f"{theme:<10} vs {ref:<8} {'term1':>9} {'term2':>9} {'FADE':>9}   n")
        for row in rows:
            if row["theme"] == theme:
                print(f"{'':10} {row['model']:<11} {row['term1']:>9.4f} {row['term2']:>9.4f} "
                      f"{row['fade']:>9.4f} {min(row['n_term1'], row['n_term2']):>4}")
        print()


if __name__ == "__main__":
    main()
