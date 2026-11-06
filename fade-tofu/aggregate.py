"""
Collects compute_fade.py results into one table.

    python aggregate.py            # reads outputs/fade/*/*.json

Writes outputs/fade_results.csv (one row per retain/evaluated-model pair) and prints a
per-split summary: mean +/- std over seeds of the RB FADE, with the MC FADE alongside.
"""

import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

from models import METHODS, REPO_ROOT

FIELDS = ["split", "category", "method", "seed", "model_r", "model_u",
          "rb_term1", "rb_term2", "rb_fade", "mc_term1", "mc_term2", "mc_fade"]


def describe(model_r, model_u):
    """(category, method, seed label) for a compute_fade result."""
    seed_r = model_r.rsplit("seed", 1)[1]
    kind = model_u.split("/")[0]
    if kind == "retain":
        return "baseline", "-", f"{seed_r}v{model_u.rsplit('seed', 1)[1]}"
    if kind == "full":
        return "full_vs_retain", "-", seed_r
    if kind in METHODS:
        return "method", kind, seed_r
    return "other", model_u, seed_r


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "outputs", "fade", "*", "*.json"))):
        with open(path) as f:
            r = json.load(f)
        category, method, seed = describe(r["model_r"], r["model_u"])
        rows.append({
            "split": r["split"], "category": category, "method": method, "seed": seed,
            "model_r": r["model_r"], "model_u": r["model_u"],
            "rb_term1": r["term1"]["rb"], "rb_term2": r["term2"]["rb"],
            "rb_fade": r["term1"]["rb"] + r["term2"]["rb"],
            "mc_term1": r["term1"]["mc"], "mc_term2": r["term2"]["mc"],
            "mc_fade": r["term1"]["mc"] + r["term2"]["mc"],
        })

    out = os.path.join(REPO_ROOT, "outputs", "fade_results.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} results -> {out}\n")

    groups = defaultdict(list)
    for row in rows:
        label = row["method"] if row["category"] == "method" else row["category"]
        groups[(row["split"], label)].append(row)
    order = ["baseline", "full_vs_retain"] + list(METHODS)
    for split in sorted({row["split"] for row in rows}):
        print(f"{split:<10} {'':16} {'RB FADE':>20} {'MC FADE':>20}   n")
        for label in order + sorted({k[1] for k in groups if k[0] == split} - set(order)):
            g = groups.get((split, label))
            if not g:
                continue
            rb = np.array([row["rb_fade"] for row in g])
            mc = np.array([row["mc_fade"] for row in g])
            print(f"{'':10} {label:<16} {rb.mean():>11.2f} ± {rb.std():<6.2f} "
                  f"{mc.mean():>11.2f} ± {mc.std():<6.2f} {len(g):>3}")
        print()


if __name__ == "__main__":
    main()
