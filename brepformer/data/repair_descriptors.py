#!/usr/bin/env python3
"""Recompute the D2 descriptor in already-preprocessed pickles. No STEP, no OCC.

Why this is possible at all, and why it is not a re-preprocessing job. ``compute_d2_descriptor``
builds its histogram from **point clouds read out of ``face_grid``**, which every pickle already
stores and which were always correct. The centroids it is also handed are used for one thing
only: estimating ``max_dist``, the upper edge of the histogram range. Those centroids were read
from ``face_attr[:, 2:5]`` — the cone/sphere/torus one-hot bits, zero on most parts — so
``max_dist`` came out at ``1e-6``, every real distance fell outside the range, and
``np.histogram`` returned all zeros. Measured on ``defeature_processed/test/00000.pkl``: **99.2%
of face pairs have an entirely empty D2 histogram**, and with the correct column (11:14)
``max_dist`` is 2.17 and each pair gets ~7 populated bins.

So the fix needs the centroid column and the grid, both already on disk. Re-running
``preprocess.py`` would re-read 28.5k STEP files through OCC to regenerate data it already has.

``angle_distance`` is **not** rewritten: its histogram range is a fixed ``[0, pi]`` and its
normals come from the grid, so it was never affected. Only the D2 channel was empty.

Cost is quadratic in faces and dominated by the (num_faces + 1)^2 x 64 float32 array, which is
256 MB for a 1000-face part. Files are rewritten one at a time via a temporary file and an atomic
rename, so an interrupted run leaves no half-written pickle.

Usage:
    # Look first: how many pairs are empty, without writing anything.
    python -m brepformer.data.repair_descriptors --dirs brepformer/data/defeature_processed \
        --check_only

    python -m brepformer.data.repair_descriptors \
        --dirs brepformer/data/defeature_processed brepformer/data/trial6_processed
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from brepformer.data.preprocessing import compute_d2_descriptor


def empty_pair_fraction(d2: np.ndarray) -> float:
    """Fraction of real face pairs whose histogram carries no mass at all.

    Row/col 0 is the virtual graph token and is excluded: it is legitimately empty, so
    including it would flatter a repaired file and understate a broken one.
    """
    if d2 is None or d2.ndim != 3 or d2.shape[0] < 2:
        return float("nan")
    m = d2[1:, 1:]
    return float((m.sum(axis=-1) == 0).mean())


def repair_one(pkl: Path, check_only: bool, bins: int) -> tuple[float, float] | None:
    """Return (before, after) empty-pair fractions, or None if the file is unusable.

    ``pkl`` is resolved before it is written. ``trial6_processed`` and ``trial6a_processed`` are
    **symlink farms** pointing into ``defeature_processed`` and ``mftrcad_processed``, and
    ``os.replace`` onto a symlink path replaces the *symlink* with a regular file: the corpus
    would silently de-link, the 17 GB of real pickles would stay broken, and the disk would grow
    by another 17 GB. Writing through to the target repairs the file the symlink refers to, which
    is what both corpora then see.
    """
    pkl = pkl.resolve()
    try:
        with open(pkl, "rb") as f:
            s = pickle.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"  {pkl.name}: unreadable ({type(e).__name__}: {e})", file=sys.stderr)
        return None

    fa = np.asarray(s.get("face_attr"))
    fg = np.asarray(s.get("face_grid"))
    if fa.ndim != 2 or fa.shape[1] < 14 or fg.ndim != 4 or fg.shape[1] < 7:
        return None

    before = empty_pair_fraction(s.get("d2_distance"))
    d2 = compute_d2_descriptor(fa[:, 11:14].astype(np.float64), num_bins=bins, face_grids=fg)
    after = empty_pair_fraction(d2)

    if not check_only:
        s["d2_distance"] = d2.astype(np.float32)
        tmp = pkl.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(s, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, pkl)  # atomic: never leaves a truncated pickle behind
    return before, after


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", type=Path, nargs="+", required=True,
                    help="Preprocessed corpus roots; every */*.pkl under each is processed.")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--bins", type=int, default=64,
                    help="Must match BrepClassifierConfig.d2_descriptor_dim, or the collator "
                         "builds a differently-shaped batch than the encoder expects.")
    ap.add_argument("--limit", type=int, default=None, help="Files per split, for a dry run.")
    ap.add_argument("--check_only", action="store_true",
                    help="Report the empty-pair fraction and write nothing.")
    args = ap.parse_args()

    # One real file can be reached by several paths: the symlink farms above mean
    # `--dirs mftrcad_processed trial6a_processed` names the same 28.5k pickles twice, and
    # repairing a file a second time is pure quadratic-in-faces waste.
    seen: set[Path] = set()

    for root in args.dirs:
        for split in args.splits:
            d = root / split
            if not d.is_dir():
                continue
            pkls = sorted(p for p in d.iterdir() if p.suffix == ".pkl")
            if args.limit:
                pkls = pkls[:args.limit]
            befores, afters, n_bad, n_dup = [], [], 0, 0
            t0 = time.time()
            for i, pkl in enumerate(pkls, 1):
                real = pkl.resolve()
                if real in seen:
                    n_dup += 1
                    continue
                seen.add(real)
                r = repair_one(pkl, args.check_only, args.bins)
                if r is None:
                    n_bad += 1
                    continue
                befores.append(r[0])
                afters.append(r[1])
                if i % 200 == 0 or i == len(pkls):
                    rate = (time.time() - t0) / i
                    print(f"  {root.name}/{split} [{i}/{len(pkls)}] "
                          f"empty pairs {np.nanmean(befores):.3f} -> {np.nanmean(afters):.3f} "
                          f"(eta {rate * (len(pkls) - i) / 60:.0f}m)", flush=True)
            verb = "would repair" if args.check_only else "repaired"
            if n_dup:
                print(f"  {root.name}/{split}: {n_dup} already handled via another path "
                      f"(symlinks into a corpus named earlier in --dirs)")
            if not befores:
                continue
            print(f"{root.name}/{split}: {verb} {len(befores)} files, skipped {n_bad}; "
                  f"mean empty-pair fraction {np.nanmean(befores):.4f} -> "
                  f"{np.nanmean(afters):.4f}")


if __name__ == "__main__":
    main()
