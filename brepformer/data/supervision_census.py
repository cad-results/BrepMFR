#!/usr/bin/env python3
"""Count supervised faces per split in a preprocessed dataset directory.

Run this on any new `*_processed` dir BEFORE training on it. Two failure modes it catches:

  1. A corpus silently preprocessed to all -1, which trains on nothing (the pcq corpora did
     exactly this).
  2. Two corpora whose validation splits have different supervision densities, which makes
     their face metrics look comparable when they are not -- `defeature_processed` val is
     99.6% supervised, `trial6_processed` val is 63.0%, so a face_f1 from one arm cannot be
     compared to a face_f1 from the other.

It also prints the class mix, whose majority share is the collapse baseline: a run whose
val/face_acc sits exactly on that number is predicting one class for every face.

Usage:
    python brepformer/data/supervision_census.py brepformer/data/trial6_processed
"""

import argparse
import collections
import glob
import pickle
from pathlib import Path


def census(data_dir: Path, splits=("train", "val", "test")):
    for split in splits:
        files = sorted(glob.glob(str(data_dir / split / "*.pkl")))
        if not files:
            continue
        total = supervised = with_any = 0
        classes = collections.Counter()
        for path in files:
            with open(path, "rb") as fh:
                rec = pickle.load(fh)
            labels = rec["face_labels"]
            labels = labels.reshape(-1).tolist() if hasattr(labels, "reshape") else list(labels)
            total += len(labels)
            kept = [int(x) for x in labels if x >= 0]
            supervised += len(kept)
            with_any += bool(kept)
            classes.update(kept)

        n = max(sum(classes.values()), 1)
        pct = 100 * supervised / max(total, 1)
        print(f"{split}: {len(files)} models, {total} faces, {supervised} supervised "
              f"({pct:.1f}%), {with_any} models with any supervised face")
        mix = {c: f"{100 * v / n:.1f}%" for c, v in sorted(classes.items())}
        print(f"    class mix: {mix}")
        if classes:
            top, share = classes.most_common(1)[0]
            print(f"    collapse baseline: predicting class {top} everywhere scores "
                  f"{100 * share / n:.1f}% face accuracy")
        if supervised == 0:
            raise SystemExit(f"FATAL: {split} has no supervised faces; training on it is a no-op")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()
    census(args.data_dir, args.splits)


if __name__ == "__main__":
    main()
