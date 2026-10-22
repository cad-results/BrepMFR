"""Curate the hard_samples corpus for training quality, not file count.

Motivation
----------
A raw GrabCAD scrape is a bad training set even when every file is valid CAD.
Measured on this corpus (399 files / 199 slugs):

* 60% of files came from 38 slugs. ``pipe-clamp-14`` alone contributed **44**
  files — parametric size variants of one clamp. ``butterfly-valves-wafer-type-1``
  contributed 21, a bearing-unit family 18, ``wire-rope-clip-din-741`` 15.
* 25 groups were byte-identical duplicates.

Training on that teaches the model those few product families. Worse, a random
train/val split leaks size-variants of the same part across the split and
inflates validation F1, so the run *looks* better while generalising worse.

Filename dedupe cannot fix this: the pipe-clamp variants have distinct names
and distinct bytes. They are distinguishable only geometrically. So this script
computes a topology/geometry signature per file and groups on that.

Signatures
----------
``exact``  — same face/edge/solid counts, same surface-type histogram, and
             matching volume+area to a tolerance. Genuinely redundant
             geometry; keep one.
``family`` — same face/edge/solid counts and same surface-type histogram, but
             different size. This is the parametric-variant case (one clamp in
             12 sizes). They are not redundant, but they are highly correlated,
             so keep at most ``--max_per_family``.

Also flags, for training suitability:

* multi-body assemblies (``n_solids > 1``) — MFTRCAD is single-body machined
  parts, and the teacher model's confidence is measurably worst on assemblies,
  so their pseudo-labels are the least trustworthy;
* face counts outside a usable band — tiny parts carry almost no signal, huge
  ones OOM'd during Phase 3 inference.

Nothing is deleted by default: the script reports, and only moves files when
given ``--apply``.

    # report only
    python brepformer/data/curate_corpus.py --step_dir brepformer/data/hard_samples

    # act on it
    python brepformer/data/curate_corpus.py --step_dir brepformer/data/hard_samples \\
        --apply --reject_dir brepformer/data/hard_samples_rejected
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import shutil
from pathlib import Path

from OCC.Core.BRep import BRep_Tool  # noqa: F401  (imported for OCC init side effects)
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GProp import GProp_GProps
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer

STEP_EXTS = {".step", ".stp"}


def surface_type_histogram(shape) -> tuple[int, ...]:
    """Count faces by OCC surface type. Rotation/translation invariant."""
    hist = collections.Counter()
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        try:
            hist[int(BRepAdaptor_Surface(exp.Current()).GetType())] += 1
        except Exception:  # noqa: BLE001
            hist[-1] += 1
        exp.Next()
    # Fixed-width tuple so signatures compare cleanly; OCC has ~11 surface types.
    return tuple(hist.get(i, 0) for i in range(-1, 12))


def count(shape, kind) -> int:
    n = 0
    exp = TopExp_Explorer(shape, kind)
    while exp.More():
        n += 1
        exp.Next()
    return n


def signature(step_path: Path) -> dict | None:
    """Topology + size fingerprint for one STEP file."""
    reader = STEPControl_Reader()
    try:
        if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
            return None
        reader.TransferRoots()
        shape = reader.OneShape()
        if shape is None or shape.IsNull():
            return None
    except Exception:  # noqa: BLE001
        return None

    n_faces = count(shape, TopAbs_FACE)
    if n_faces == 0:
        return None

    props = GProp_GProps()
    try:
        brepgprop.VolumeProperties(shape, props)
        volume = abs(props.Mass())
    except Exception:  # noqa: BLE001
        volume = 0.0
    aprops = GProp_GProps()
    try:
        brepgprop.SurfaceProperties(shape, aprops)
        area = abs(aprops.Mass())
    except Exception:  # noqa: BLE001
        area = 0.0

    box = Bnd_Box()
    try:
        brepbndlib.Add(shape, box)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        dims = sorted([abs(xmax - xmin), abs(ymax - ymin), abs(zmax - zmin)])
    except Exception:  # noqa: BLE001
        dims = [0.0, 0.0, 0.0]

    return {
        "path": str(step_path),
        "name": step_path.name,
        "slug": step_path.name.split("__")[0],
        "n_faces": n_faces,
        "n_edges": count(shape, TopAbs_EDGE),
        "n_solids": count(shape, TopAbs_SOLID),
        "surf_hist": surface_type_histogram(shape),
        "volume": volume,
        "area": area,
        "dims": dims,
        "bytes": step_path.stat().st_size,
    }


def topology_key(s: dict) -> tuple:
    """Scale-invariant: same topology and same mix of surface types."""
    return (s["n_faces"], s["n_edges"], s["n_solids"], s["surf_hist"])


def exact_key(s: dict, tol: float = 1e-4) -> tuple:
    """Topology plus size, so genuinely identical geometry collides."""
    def q(v: float) -> int:
        # Quantise on a log scale so the tolerance is relative, not absolute —
        # CAD parts in this corpus span millimetres to metres.
        return 0 if v <= 0 else int(round(math.log(v) / tol))
    return topology_key(s) + (q(s["volume"]), q(s["area"]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step_dir", type=Path, default=Path("brepformer/data/hard_samples"))
    p.add_argument("--recursive", action="store_true", default=True)
    p.add_argument("--max_per_family", type=int, default=2,
                   help="Keep at most this many parametric variants per topology family")
    p.add_argument("--max_per_slug", type=int, default=3,
                   help="Keep at most this many files per GrabCAD slug. Catches "
                        "product families whose sizes differ topologically and so "
                        "evade --max_per_family (e.g. an M3-M12 screw library).")
    p.add_argument("--min_faces", type=int, default=12,
                   help="Below this a part carries too little signal to be worth a sample")
    p.add_argument("--max_faces", type=int, default=1500,
                   help="Above this Phase 3 inference OOM'd; also far outside MFTRCAD")
    p.add_argument("--drop_assemblies", action="store_true",
                   help="Reject multi-solid files outright rather than just flagging them")
    p.add_argument("--apply", action="store_true", help="Actually move rejected files")
    p.add_argument("--reject_dir", type=Path,
                   default=Path("brepformer/data/hard_samples_rejected"))
    p.add_argument("--report", type=Path, default=Path("/tmp/curate_report.json"))
    args = p.parse_args()

    files = sorted(
        f for f in (args.step_dir.rglob("*") if args.recursive else args.step_dir.iterdir())
        if f.is_file() and f.suffix.lower() in STEP_EXTS
    )
    # Never touch files already triaged into yes/maybe/no — those are decisions.
    triaged = {"yes", "maybe", "no"}
    files = [f for f in files if not (triaged & set(p.name for p in f.parents))]
    print(f"Fingerprinting {len(files)} STEP files (this reads every file, be patient)...")

    sigs: list[dict] = []
    unreadable: list[str] = []
    for i, f in enumerate(files, 1):
        s = signature(f)
        if s is None:
            unreadable.append(f.name)
        else:
            sigs.append(s)
        if i % 25 == 0:
            print(f"  {i}/{len(files)}")

    # --- group -------------------------------------------------------------
    by_exact: dict[tuple, list[dict]] = collections.defaultdict(list)
    for s in sigs:
        by_exact[exact_key(s)].append(s)
    by_topo: dict[tuple, list[dict]] = collections.defaultdict(list)
    for s in sigs:
        by_topo[topology_key(s)].append(s)

    reject: dict[str, str] = {}  # path -> reason

    # Prefer the smallest file among identical geometry (fastest to preprocess).
    for grp in by_exact.values():
        if len(grp) > 1:
            for s in sorted(grp, key=lambda x: x["bytes"])[1:]:
                reject[s["path"]] = "exact_duplicate"

    # Cap parametric families. Keep the most topologically "central" members by
    # picking spread-out sizes rather than the first N alphabetically.
    for grp in by_topo.values():
        alive = [s for s in grp if s["path"] not in reject]
        if len(alive) <= args.max_per_family:
            continue
        alive.sort(key=lambda x: x["volume"])
        keep_idx = {0, len(alive) - 1}
        while len(keep_idx) < args.max_per_family:
            keep_idx.add(len(alive) // 2 + len(keep_idx))
        for i, s in enumerate(alive):
            if i not in keep_idx:
                reject[s["path"]] = "family_variant"

    # Slug-level cap. Topology grouping alone misses product families whose
    # sizes differ topologically — e.g. a "M3-M12 socket head cap screw"
    # library, where each diameter has a different face count and so lands in
    # its own topology family despite being the same part in 7 sizes. The slug
    # is a strong proxy for "one uploader's parametric family", so cap it too.
    by_slug: dict[str, list[dict]] = collections.defaultdict(list)
    for s in sigs:
        if s["path"] not in reject:
            by_slug[s["slug"]].append(s)
    for grp in by_slug.values():
        if len(grp) <= args.max_per_slug:
            continue
        # Keep the most geometrically distinct members: spread over face count.
        grp.sort(key=lambda x: x["n_faces"])
        keep_idx = {0, len(grp) - 1}
        step = max(len(grp) // max(args.max_per_slug, 1), 1)
        i = 0
        while len(keep_idx) < args.max_per_slug and i < len(grp):
            keep_idx.add(i)
            i += step
        for i, s in enumerate(grp):
            if i not in keep_idx:
                reject[s["path"]] = "slug_overrepresented"

    for s in sigs:
        if s["path"] in reject:
            continue
        if s["n_faces"] < args.min_faces:
            reject[s["path"]] = "too_few_faces"
        elif s["n_faces"] > args.max_faces:
            reject[s["path"]] = "too_many_faces"
        elif args.drop_assemblies and s["n_solids"] > 1:
            reject[s["path"]] = "assembly"

    kept = [s for s in sigs if s["path"] not in reject]

    # --- report ------------------------------------------------------------
    reasons = collections.Counter(reject.values())
    print("\n" + "=" * 64)
    print(f"readable:      {len(sigs)}   unreadable: {len(unreadable)}")
    print(f"KEEP:          {len(kept)}")
    print(f"REJECT:        {len(reject)}")
    for r, n in reasons.most_common():
        print(f"   {r:20s} {n}")
    print()
    multi = sum(1 for s in kept if s["n_solids"] > 1)
    print(f"of kept: {multi} multi-body assemblies "
          f"({100 * multi / max(len(kept), 1):.0f}%), "
          f"{len(kept) - multi} single-body")
    print(f"distinct slugs kept: {len({s['slug'] for s in kept})}")
    fam = collections.Counter(topology_key(s) for s in kept)
    print(f"largest remaining family: {max(fam.values()) if fam else 0} files")

    worst = collections.Counter(s["slug"] for s in kept).most_common(8)
    print("\nmost-represented slugs after curation:")
    for slug, n in worst:
        print(f"   {n:3d}  {slug[:64]}")

    args.report.write_text(json.dumps({
        "kept": [s["name"] for s in kept],
        "rejected": {Path(k).name: v for k, v in reject.items()},
        "unreadable": unreadable,
        "n_kept": len(kept), "n_rejected": len(reject),
    }, indent=2))
    print(f"\nreport: {args.report}")

    if not args.apply:
        print("\n(dry run — pass --apply to move rejected files to --reject_dir)")
        return

    args.reject_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for path, reason in reject.items():
        src = Path(path)
        if not src.exists():
            continue
        dst_dir = args.reject_dir / reason
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if dst.exists():
            continue
        shutil.move(str(src), str(dst))
        moved += 1
    print(f"moved {moved} files into {args.reject_dir}/<reason>/")


if __name__ == "__main__":
    main()
