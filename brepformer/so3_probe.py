#!/usr/bin/env python3
"""Is the segmenter rotation-robust, or has it memorised the world axes?

Why this is worth measuring. FoV-Net reports UV-Net dropping from **98.95 to 35.44**
per-face accuracy on MFCAD++ when the test parts are randomly rotated, because the
network keys on axis-aligned surface normals that the training corpus never varies.
Our pipeline augments with 90-degree rotations about the coordinate axes *only* (the 24
octahedral rotations), so a model can satisfy that augmentation and still be a
world-frame detector. Real GrabCAD parts arrive in whatever frame their author used,
so if this drop reproduces here it is a live cause of the synthetic-to-real gap and it
is fixable with augmentation alone -- no relabelling, no architecture change.

What is rotated. The preprocessed sample is already normalised, so this script rotates
the derived tensors rather than re-running OCC on a rotated STEP. Exactly these slots
carry the frame (layouts read off ``step_to_graph.normalize_geometry``):

    face_grid[:, 0:3]   sampled points        face_grid[:, 3:6]   surface normals
    edge_grid[:, 0:3]   sampled points        edge_grid[:, 3:6]   curve tangents
    face_attr[:, 11:14] face centroid

Everything else is a rotation invariant and is deliberately left alone: ``edge_attr``
after normalisation is length plus convexity/curve-type one-hots, ``spatial_pos`` and
``in_degree`` are topological, and ``d2_distance`` / ``angle_distance`` are pairwise
shape descriptors. So a drop here cannot be blamed on a corrupted input -- the only
thing that changed is the frame.

``--renormalize`` (default on) reproduces what preprocessing a rotated STEP would have
produced: the normaliser divides by *half the max axis-aligned bbox extent*, which is
itself frame-dependent, so after rotating we recompute the bbox centre and scale and
re-apply them. Turn it off to separate "sensitive to the frame" from "sensitive to the
scale the frame implies".

Usage:
    python -m brepformer.so3_probe \
        --ckpts "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt" \
        --data_dir brepformer/data/defeature_processed --split test \
        --n_rotations 4 --device cpu
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from brepformer.scan_checkpoints import score_checkpoint


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform sample from SO(3) via a random unit quaternion (Shoemake)."""
    u1, u2, u3 = rng.random(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def octahedral_rotation(rng: np.random.Generator) -> np.ndarray:
    """One of the 24 axis-aligned 90-degree rotations -- the augmentation we already do.

    This is the control. A model that holds up here but collapses under `random_rotation`
    has learned the training frame up to the augmentation and no further, which points at
    augmentation coverage rather than at anything in the architecture.
    """
    perm = rng.permutation(3)
    signs = rng.choice([-1.0, 1.0], size=3)
    R = np.zeros((3, 3))
    for i, p in enumerate(perm):
        R[i, p] = signs[i]
    if np.linalg.det(R) < 0:          # keep it a rotation, not a reflection
        R[0] *= -1
    return R


def rotate_sample(sample: dict, R: np.ndarray, renormalize: bool = True) -> dict:
    """Apply R to every frame-dependent slot of a preprocessed sample.

    Copies only the three arrays that are actually rewritten, and shares the rest with
    ``sample``. A deepcopy here would duplicate ``d2_distance``, which is
    (F+1, F+1, 64) float32 -- 256 MB for a 1000-face part -- on every call, which makes
    this unusable as a training-time augmentation. Nothing downstream mutates the shared
    arrays, so the sharing is safe; do not write into the returned sample in place.
    """
    s = dict(sample)
    Rt = R.T.astype(np.float32)

    fg = np.array(s["face_grid"], dtype=np.float32, copy=True)   # (F, 7, U, V)
    for lo in (0, 3):                                       # points, then normals
        v = fg[:, lo:lo + 3, :, :]
        f, _, u, w = v.shape
        v = v.transpose(0, 2, 3, 1).reshape(-1, 3) @ Rt
        fg[:, lo:lo + 3, :, :] = v.reshape(f, u, w, 3).transpose(0, 3, 1, 2)
    s["face_grid"] = fg

    eg = np.array(s["edge_grid"], dtype=np.float32, copy=True)   # (E, 12, N)
    if eg.size:
        for lo in (0, 3):                                   # points, then tangents
            v = eg[:, lo:lo + 3, :]
            e, _, n = v.shape
            v = v.transpose(0, 2, 1).reshape(-1, 3) @ Rt
            eg[:, lo:lo + 3, :] = v.reshape(e, n, 3).transpose(0, 2, 1)
        s["edge_grid"] = eg

    fa = np.array(s["face_attr"], dtype=np.float32, copy=True)
    fa[:, 11:14] = fa[:, 11:14] @ Rt
    s["face_attr"] = fa

    if renormalize:
        # The normaliser's scale is half the max AXIS-ALIGNED bbox extent, so it changes
        # under rotation. Recompute it exactly as normalize_geometry would on the rotated
        # solid, using the same masked point set.
        mask = fg[:, 6, :, :] > 0.5
        pts = np.concatenate([fg[i, 0:3][:, mask[i]].T for i in range(fg.shape[0])
                              if mask[i].any()], axis=0) if mask.any() else None
        if pts is not None and len(pts):
            lo3, hi3 = pts.min(axis=0), pts.max(axis=0)
            centre = (lo3 + hi3) / 2.0
            scale = float((hi3 - lo3).max()) / 2.0 or 1.0
            c = centre.astype(np.float32).reshape(1, 3, 1, 1)
            fg[:, 0:3] = (fg[:, 0:3] - c) / scale
            s["face_grid"] = fg
            if eg.size:
                eg[:, 0:3] = (eg[:, 0:3] - centre.astype(np.float32).reshape(1, 3, 1)) / scale
                s["edge_grid"] = eg
            fa[:, 11:14] = (fa[:, 11:14] - centre.astype(np.float32)) / scale
            fa[:, 9] = fa[:, 9] / (scale ** 2)      # area lives at [9] post-normalisation
            s["face_attr"] = fa
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data_dir", type=Path,
                    default=Path("brepformer/data/defeature_processed"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--label_space", default="5", choices=["27", "5"])
    ap.add_argument("--n_rotations", type=int, default=4,
                    help="Random SO(3) draws. Each is scored on the whole split, so the "
                         "spread across draws tells you whether one unlucky frame is "
                         "doing the damage.")
    ap.add_argument("--n_octahedral", type=int, default=2,
                    help="Axis-aligned 90-degree draws, as the control.")
    ap.add_argument("--no_renormalize", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
        else args.device)

    meta = args.data_dir / "metadata.json"
    data_classes = int(json.loads(meta.read_text()).get("num_classes", 27)) \
        if meta.exists() else 27

    pkls = sorted(p for p in (args.data_dir / args.split).iterdir() if p.suffix == ".pkl")
    if args.limit:
        pkls = pkls[:args.limit]
    base = []
    for pkl in pkls:
        with open(pkl, "rb") as f:
            s = pickle.load(f)
        y = np.asarray(s["face_labels"])
        if (y >= 0).sum():
            base.append((s, y))
    print(f"{len(base)} labelled models from {args.data_dir / args.split} ({device})")

    rng = np.random.default_rng(args.seed)
    frames = [("identity", np.eye(3))]
    frames += [(f"octahedral-{i}", octahedral_rotation(rng))
               for i in range(args.n_octahedral)]
    frames += [(f"so3-{i}", random_rotation(rng)) for i in range(args.n_rotations)]

    renorm = not args.no_renormalize
    rows = {ck: {"checkpoint": ck, "frames": {}} for ck in args.ckpts}

    # Frames are the OUTER loop on purpose: one rotated copy of the split is held at a
    # time. `d2_distance` is (F+1, F+1, 64) float32, so a 1000-face part is 256 MB by
    # itself and materialising every frame up front would exhaust memory on a big split.
    for name, R in frames:
        samples = base if name == "identity" \
            else [(rotate_sample(s, R, renorm), y) for s, y in base]
        print(f"\n[{name}]")
        for ck in args.ckpts:
            r = score_checkpoint(Path(ck), samples, device,
                                 args.label_space, data_classes)
            if r is None:
                continue
            rows[ck]["frames"][name] = {"macro_f1": r["macro_f1"],
                                        "weighted_f1": r["weighted_f1"],
                                        "per_class": r["per_class"]}
            print(f"  macro {r['macro_f1']:.4f}  weighted {r['weighted_f1']:.4f}  "
                  f"{Path(ck).parent.parent.name}/{Path(ck).parent.name}", flush=True)
        del samples

    results = []
    for ck in args.ckpts:
        row = rows[ck]
        print(f"\n{ck}")
        ident = row["frames"].get("identity", {}).get("macro_f1")
        so3 = [v["macro_f1"] for k, v in row["frames"].items() if k.startswith("so3")]
        octa = [v["macro_f1"] for k, v in row["frames"].items() if k.startswith("octa")]
        if ident and so3:
            row["retention_so3"] = float(np.mean(so3) / ident)
            row["retention_octahedral"] = float(np.mean(octa) / ident) if octa else None
            print(f"  -> retains {100 * row['retention_so3']:.1f}% of macro F1 under "
                  f"SO(3), {100 * (row['retention_octahedral'] or 0):.1f}% under the "
                  f"90-degree augmentation it was trained with")
        results.append(row)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
