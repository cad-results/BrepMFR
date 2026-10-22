#!/usr/bin/env python3
"""Combine several preprocessed corpora into one train/val/test tree for training.

``preprocess.py`` and ``train_preprocessed.py`` each take a single ``--data_dir``, so
training on mftrcad plus a pseudo-labelled corpus needs the two merged. Re-running
``preprocess.py`` over the union would recompute descriptors for all ~28.5k mftrcad
models (hours) to add a few hundred new ones, so this merges the *already preprocessed*
per-sample pickles instead: symlinks by default, so the merge is near-instant and costs
no extra disk.

Splits are preserved rather than reshuffled. Each input corpus keeps its own
train/val/test assignment, which is what makes the result comparable to a run on
mftrcad alone: the mftrcad test set is bit-identical, so a metric change is attributable
to the added data and not to a different split. It also preserves ``preprocess.py``'s
group-aware splitting — re-splitting here would need the slug groups again and could
leak parametric variants across splits.

**Feature compatibility is checked, not assumed.** Samples from two corpora are only
interchangeable if they were preprocessed with the same descriptor settings; mixing a
``compute_descriptors=True`` corpus with a ``False`` one yields samples missing
``d2_distance``/``angle_distance``, which fails at collate time after the run has
started. Mismatched ``num_classes``/``num_spatial``/bin counts are rejected up front.

Usage:
    python -m brepformer.data.combine_processed \
        --inputs brepformer/data/mftrcad_processed \
                 brepformer/data/suitable_samples_pcq_processed \
        --output_dir brepformer/data/trial6_processed
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

SPLITS = ("train", "val", "test")

#: Settings that must agree across corpora for their samples to be interchangeable.
COMPAT_KEYS = ("num_classes", "compute_descriptors", "num_spatial",
               "d2_bins", "angle_bins")


def load_metadata(d: Path) -> dict:
    p = d / "metadata.json"
    if not p.exists():
        raise SystemExit(f"{d}: no metadata.json — is this a preprocess.py output dir?")
    return json.loads(p.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, nargs="+", required=True,
                    help="Preprocessed dirs to combine (each with train/val/test/ and "
                         "metadata.json). Order sets the sample numbering.")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--copy", action="store_true",
                    help="Copy pickles instead of symlinking. Symlinks are the default "
                         "because the inputs are large and read-only during training; "
                         "copy if the output must survive the inputs being deleted.")
    ap.add_argument("--allow_incompatible", action="store_true",
                    help="Proceed despite mismatched preprocessing settings. Almost "
                         "always wrong — mismatched descriptors fail at collate time, "
                         "mid-run.")
    args = ap.parse_args()

    metas = {d: load_metadata(d) for d in args.inputs}
    ref_dir, ref = next(iter(metas.items()))
    for d, m in metas.items():
        diffs = [f"{k}: {m.get(k)!r} vs {ref.get(k)!r}"
                 for k in COMPAT_KEYS if m.get(k) != ref.get(k)]
        if diffs:
            msg = (f"incompatible preprocessing between {d} and {ref_dir}:\n  "
                   + "\n  ".join(diffs))
            if not args.allow_incompatible:
                raise SystemExit(msg + "\nRe-preprocess with matching settings, or pass "
                                       "--allow_incompatible if you are certain.")
            print("WARNING: " + msg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    provenance: dict[str, dict[str, str]] = {s: {} for s in SPLITS}
    model_index: dict[str, str] = {}

    for split in SPLITS:
        out_split = args.output_dir / split
        out_split.mkdir(parents=True, exist_ok=True)
        n = 0
        for d in args.inputs:
            src_split = d / split
            if not src_split.is_dir():
                print(f"  {d.name}/{split}: missing, skipped")
                continue
            pkls = sorted(p for p in src_split.iterdir() if p.suffix == ".pkl")
            for src in pkls:
                dst = out_split / f"{n:06d}.pkl"
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                if args.copy:
                    shutil.copyfile(src, dst)
                else:
                    # Absolute target: the output tree is often read from a different
                    # cwd than it was built from.
                    dst.symlink_to(os.path.abspath(src))
                provenance[split][f"{n:06d}"] = f"{d.name}/{split}/{src.name}"
                n += 1
            print(f"  {d.name}/{split}: {len(pkls)} samples")
        counts[split] = n
        print(f"{split}: {n} total")

    # Merge per-corpus model_id indices if present, tagging the source corpus so a
    # prediction can be traced back to which dataset it came from.
    for d in args.inputs:
        idx = d / "model_id_index.json"
        if idx.exists():
            try:
                for k, v in json.loads(idx.read_text()).items():
                    model_index[f"{d.name}:{k}"] = v
            except Exception:  # noqa: BLE001
                pass

    metadata = dict(ref)
    metadata.update({
        "num_samples": sum(counts.values()),
        "split_sizes": counts,
        "combined_from": [str(d) for d in args.inputs],
        "combined_source_sizes": {
            str(d): metas[d].get("split_sizes") for d in args.inputs
        },
    })
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    if model_index:
        (args.output_dir / "model_id_index.json").write_text(json.dumps(model_index, indent=2))

    print(f"\nwrote {args.output_dir} "
          f"({'copies' if args.copy else 'symlinks'}): "
          f"{sum(counts.values())} samples, metadata.json + provenance.json")


if __name__ == "__main__":
    main()
