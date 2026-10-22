#!/usr/bin/env python3
"""Audit ``face_grid`` channel 6 -- the UV validity mask -- across corpora, and fit MFTRCAD's
UV-gridding convention against their own published graphs.

This is the evidence for **Bug 8**. The short version: ``step_to_graph.compute_face_uv_grid``
grids a face's UV parameter box and writes ``mask = 1`` at every point the surface evaluated
at. A face is a *trimmed* region of a surface and that region is almost never a box, so the
grid samples points outside the face -- inside a through-hole, off the corner of a triangular
pocket floor, past the taper of a narrowing fillet -- hands them the underlying surface's xyz
and normal, and declares them valid. On every STEP-derived corpus here, channel 6 is therefore
identically 1 and carries no information at all.

MFTRCAD's published graphs do not look like that: 44.3% of their grid points are masked. So the
SurfaceEncoder learns channel 6 as a validity gate on the pretraining corpus and is handed a
constant on the real parts it is deployed on. That is an input-channel distribution shift that
lands squarely on the transfer this whole program depends on.

Three modes:

``disk``
    Read masks straight out of preprocessed pickles and/or raw graph JSONs. Costs no geometry
    kernel; run it on any corpus to see whether channel 6 is alive or dead there.

``steps``
    Re-derive grids from STEP with and without ``--mftrcad_uv`` and report how many faces the
    legacy path feeds off-face geometry to. Answers "how much does this corpus actually care".

``fit``
    The measurement that pins the convention. Runs the trim test on MFTRCAD's *own* STEP files
    at a sweep of UV paddings and scores each against their published masks on the same faces.
    ``--pad_sweep 0 0.1 0.15 0.2 0.3`` reproduces the table in ``compute_face_uv_grid``: 0.10
    matches at mean absolute per-face error 0.038, and nothing else is close.

Usage:
    # Is channel 6 alive on these corpora?
    python -m brepformer.data.uv_mask_audit disk \
        --pkl_dirs brepformer/data/defeature_processed/test \
        --graph_dirs brepformer/data/mftrcad/graphs --limit 200

    # How much off-face geometry does the legacy path feed the real parts?
    python -m brepformer.data.uv_mask_audit steps \
        --steps_dir brepformer/data/defeature/steps --limit 12

    # Fit the convention against MFTRCAD's published masks.
    python -m brepformer.data.uv_mask_audit fit \
        --steps_dir brepformer/data/mftrcad/steps \
        --graphs_dir brepformer/data/mftrcad/graphs --limit 15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _summarize(name: str, frac: np.ndarray) -> dict:
    """Print and return the five numbers that decide whether channel 6 carries information."""
    st = {
        "faces": int(frac.size),
        "masked_mean": float(frac.mean()),
        "masked_median": float(np.median(frac)),
        "masked_p90": float(np.percentile(frac, 90)),
        "frac_faces_any_masked": float((frac > 0).mean()),
        "frac_faces_over_half_masked": float((frac > 0.5).mean()),
    }
    print(f"{name:<44} {st['faces']:>7} faces  masked mean {st['masked_mean']:.3f} "
          f"median {st['masked_median']:.3f} p90 {st['masked_p90']:.3f}  "
          f"any {100 * st['frac_faces_any_masked']:>5.1f}%  "
          f">half {100 * st['frac_faces_over_half_masked']:>5.1f}%")
    if st["masked_mean"] == 0.0:
        print(f"{'':<44} ^^ channel 6 is the constant 1 here: DEAD input channel, and every "
              f"off-face UV sample is being declared valid")
    return st


def _mask_frac_from_grid(grid: np.ndarray) -> np.ndarray:
    """Per-face masked fraction from a ``(F, 7, H, W)`` face grid."""
    m = np.asarray(grid)[:, 6] > 0.5
    return 1.0 - m.reshape(m.shape[0], -1).mean(1)


def mode_disk(args) -> dict:
    import pickle

    out = {}
    for d in args.pkl_dirs or []:
        fracs = []
        for pkl in sorted(Path(d).glob("*.pkl"))[: args.limit]:
            with open(pkl, "rb") as fh:
                s = pickle.load(fh)
            fracs.append(_mask_frac_from_grid(np.asarray(s["face_grid"])[: int(s["num_faces"])]))
        if fracs:
            out[str(d)] = _summarize(f"pkl {Path(d).parent.name}/{Path(d).name}",
                                     np.concatenate(fracs))
    for d in args.graph_dirs or []:
        fracs = []
        for js in sorted(Path(d).glob("*.json"))[: args.limit]:
            try:
                _, payload = json.loads(js.read_text())
            except Exception:  # noqa: BLE001
                continue
            fracs.append(_mask_frac_from_grid(np.asarray(payload["graph_face_grid"])))
        if fracs:
            out[str(d)] = _summarize(f"json {Path(d).parent.name}/{Path(d).name}",
                                     np.concatenate(fracs))
    if not out:
        raise SystemExit("nothing measured: pass --pkl_dirs and/or --graph_dirs")
    return out


def mode_steps(args) -> dict:
    from brepformer.data.step_to_graph import compute_face_uv_grid, get_faces, read_step

    legacy, fixed = [], []
    steps = sorted(Path(args.steps_dir).glob("*.step"))[: args.limit]
    for sp in steps:
        shape = read_step(str(sp))
        if shape is None:
            continue
        faces = get_faces(shape)
        # A 2000-face part costs 200k trim queries and tells us nothing a 60-face part does not.
        if not faces or len(faces) > args.max_faces:
            continue
        legacy.append(np.asarray(
            [1.0 - float(compute_face_uv_grid(f, mftrcad_uv=False)[6].mean()) for f in faces]))
        fixed.append(np.asarray(
            [1.0 - float(compute_face_uv_grid(f, mftrcad_uv=True)[6].mean()) for f in faces]))
    if not legacy:
        raise SystemExit(f"no readable STEP under {args.steps_dir}")
    out = {"legacy": _summarize("legacy (grid the box, all valid)", np.concatenate(legacy)),
           "mftrcad_uv": _summarize("mftrcad_uv (pad 10% + trim mask)", np.concatenate(fixed))}
    f = np.concatenate(fixed)
    print(f"\nunder the legacy path this corpus feeds off-face geometry as valid on "
          f"{100 * (f > 0).mean():.1f}% of faces, and on {100 * (f > 0.5).mean():.1f}% of faces "
          f"more than half the grid is off-face")
    return out


def mode_fit(args) -> dict:
    """Fit the UV padding that reproduces MFTRCAD's published masks.

    Scored per face, not per corpus. Matching a corpus mean is easy and two conventions with
    very different per-face behaviour can share one; ``mean|diff|`` is what distinguishes them.
    """
    from OCC.Core.BRepTools import breptools
    from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
    from OCC.Core.gp import gp_Pnt2d
    from OCC.Core.TopAbs import TopAbs_OUT

    from brepformer.data.step_to_graph import get_faces, read_step

    pads = list(args.pad_sweep)
    ours: dict[float, list] = {p: [] for p in pads}
    pub: list[float] = []
    n = 0
    for js in sorted(Path(args.graphs_dir).glob("*.json")):
        sp = Path(args.steps_dir) / f"{js.stem}.step"
        if not sp.exists():
            continue
        try:
            _, payload = json.loads(js.read_text())
        except Exception:  # noqa: BLE001
            continue
        published = np.asarray(payload["graph_face_grid"])[:, 6]
        shape = read_step(str(sp))
        if shape is None:
            continue
        faces = get_faces(shape)
        # Only compare when our face traversal order and count agree with theirs; otherwise the
        # per-face pairing is meaningless and the mean would silently hide it.
        if len(faces) != published.shape[0]:
            continue
        for face, pm in zip(faces, published):
            try:
                clf = BRepTopAdaptor_FClass2d(face, 1e-7)
            except Exception:  # noqa: BLE001
                continue
            u0, u1, v0, v1 = breptools.UVBounds(face)
            du, dv = u1 - u0, v1 - v0
            for p in pads:
                a0, a1 = u0 - p * du, u1 + p * du
                b0, b1 = v0 - p * dv, v1 + p * dv
                out = 0
                for i in range(args.nu):
                    for j in range(args.nv):
                        u = a0 + (a1 - a0) * i / max(args.nu - 1, 1)
                        v = b0 + (b1 - b0) * j / max(args.nv - 1, 1)
                        try:
                            out += clf.Perform(gp_Pnt2d(u, v)) == TopAbs_OUT
                        except Exception:  # noqa: BLE001
                            pass
                ours[p].append(out / float(args.nu * args.nv))
            pub.append(1.0 - float(pm.mean()))
        n += 1
        if n >= args.limit:
            break
    if not pub:
        raise SystemExit("no model matched by id AND face count between graphs_dir and steps_dir")

    B = np.asarray(pub)
    print(f"\nmatched {n} models, {B.size} faces; MFTRCAD published masked mean {B.mean():.3f}\n")
    print(f"{'pad':>6}{'masked':>10}{'mean|diff|':>12}{'corr':>8}")
    rows = {}
    best = None
    for p in pads:
        A = np.asarray(ours[p])
        md = float(np.abs(A - B).mean())
        corr = float(np.corrcoef(A, B)[0, 1]) if A.std() > 0 else float("nan")
        rows[str(p)] = {"masked_mean": float(A.mean()), "mean_abs_diff": md, "corr": corr}
        print(f"{p:>6.2f}{A.mean():>10.3f}{md:>12.3f}{corr:>8.3f}")
        if best is None or md < rows[str(best)]["mean_abs_diff"]:
            best = p
    print(f"\nbest pad {best:.2f} (mean|diff| {rows[str(best)]['mean_abs_diff']:.3f}) -- this is "
          f"the value MFTRCAD_UV_PAD is set to in step_to_graph.py")
    return {"published_masked_mean": float(B.mean()), "faces": int(B.size), "models": n,
            "by_pad": rows, "best_pad": best}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    d = sub.add_parser("disk", help="Mask statistics from preprocessed pickles / graph JSONs")
    d.add_argument("--pkl_dirs", type=Path, nargs="*")
    d.add_argument("--graph_dirs", type=Path, nargs="*")
    d.add_argument("--limit", type=int, default=200)

    s = sub.add_parser("steps", help="Legacy vs mftrcad_uv, re-derived from STEP")
    s.add_argument("--steps_dir", type=Path, required=True)
    s.add_argument("--limit", type=int, default=12)
    s.add_argument("--max_faces", type=int, default=400)

    f = sub.add_parser("fit", help="Fit the UV padding against MFTRCAD's published masks")
    f.add_argument("--steps_dir", type=Path, default=Path("brepformer/data/mftrcad/steps"))
    f.add_argument("--graphs_dir", type=Path, default=Path("brepformer/data/mftrcad/graphs"))
    f.add_argument("--limit", type=int, default=15)
    f.add_argument("--pad_sweep", type=float, nargs="+",
                   default=[0.0, 0.1, 0.15, 0.2, 0.3])
    f.add_argument("--nu", type=int, default=10)
    f.add_argument("--nv", type=int, default=10)

    # On each subparser rather than the top level, so `... disk --out X` works: argparse only
    # accepts a parent-level option BEFORE the subcommand, which is not where anyone types it.
    for p_ in (d, s, f):
        p_.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    res = {"disk": mode_disk, "steps": mode_steps, "fit": mode_fit}[args.mode](args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
