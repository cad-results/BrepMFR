#!/usr/bin/env python3
"""Measure exact B-Rep geometry per label class over the hand-labelled defeature corpus.

This exists so the vision prompt's decision logic is derived from the ground truth rather
than from plausible-sounding rules. Written by hand, those rules are wrong in ways that
are invisible until measured: the first version of the prompt asserted "narrow + curved =
fillet", which is also true of a bore wall, and it turned three ground-truth holes into
confident fillets. And ``geometric_rules.py`` -- also hand-written -- was measured to
overwrite 15% of correct mftrcad labels and 100% of ``chamfer_2``.

So: read every labelled model, extract the geometry the kernel actually knows for every
face (surface type, radius, sweep, neighbour surfaces, per-edge normal deviation and
convexity), and report per class what those numbers really look like. The output feeds
``vision_label._stats_block``, which states the measured ranges in the prompt, and it
doubles as a check on how separable the classes are at all: a fact whose distribution is
identical across classes is not worth a line in the prompt.

Percentiles rather than means, because these distributions are not symmetric -- a class
whose radius ranges over three orders of magnitude has a meaningless mean.

Usage:
    python -m brepformer.data.measure_class_geometry --limit 200            # quick look
    python -m brepformer.data.measure_class_geometry --workers 24           # full corpus
    python -m brepformer.data.measure_class_geometry --label_space 27 \\
        --steps brepformer/data/mftrcad/steps --labels brepformer/data/mftrcad/labels
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from brepformer.data.classes import CLASS_NAMES, DEFEATURE_CLASS_NAMES

#: Numeric facts worth summarising. Anything absent for a face (a plane has no radius) is
#: simply skipped, so the per-fact sample counts differ and are reported.
NUMERIC = ["area_fraction_of_part", "radius_vs_part_extent", "u_sweep_degrees",
           "coaxial_group_sweep_degrees", "half_angle_degrees",
           "n_neighbours", "min_edge_normal_deviation_degrees",
           "max_edge_normal_deviation_degrees", "n_tangential_edges",
           "n_convex_edges", "n_concave_edges", "n_faces_in_part",
           # Added after the first round of measurement; each is a whole-part or
           # graph-level quantity that no single-face fact can stand in for.
           "n_inner_wires", "n_wires", "accessibility", "hull_depth_vs_extent",
           "concave_component_size", "concave_component_area_fraction",
           "strict_component_size", "strict_component_area_fraction"]

CATEGORICAL = ["surface_type", "concave_surface", "closed_in_u", "on_convex_hull",
               "is_saddle", "mean_curvature_sign", "gaussian_curvature_sign",
               "strict_component_surfaces"]

DEFAULT_STATS = (Path(__file__).resolve().parent / "class_geometry_stats.json")


def _one_model(job) -> Optional[Dict[int, List[Dict]]]:
    """Geometry of every labelled face of one model, grouped by its ground-truth class."""
    step_path, label_path = job
    try:
        from brepformer.face_geometry import face_geometry
        labels = json.loads(Path(label_path).read_text())["cls"]
        geo = face_geometry(str(step_path), with_topology=True)
    except Exception as e:  # noqa: BLE001 - one unreadable STEP must not kill the sweep
        return {"__error__": f"{Path(step_path).name}: {type(e).__name__}: {e}"}

    out: Dict[int, List[Dict]] = defaultdict(list)
    for fi, rec in geo.items():
        cls = labels.get(str(fi))
        if cls is None:
            continue
        rec.pop("edges", None)
        out[int(cls)].append(rec)
    return out


def summarise(by_class: Dict[int, List[Dict]], names: List[str]) -> Dict:
    stats: Dict = {"classes": {}}
    for cls, recs in sorted(by_class.items()):
        entry: Dict = {"name": names[cls] if cls < len(names) else str(cls),
                       "n_faces": len(recs), "numeric": {}, "categorical": {}}
        for key in NUMERIC:
            vals = [r[key] for r in recs if isinstance(r.get(key), (int, float))]
            if not vals:
                continue
            a = np.asarray(vals, dtype=float)
            entry["numeric"][key] = {
                "n": int(a.size),
                "p10": round(float(np.percentile(a, 10)), 3),
                "p50": round(float(np.percentile(a, 50)), 3),
                "p90": round(float(np.percentile(a, 90)), 3),
            }
        for key in CATEGORICAL:
            c = Counter(str(r.get(key)) for r in recs if r.get(key) is not None)
            tot = sum(c.values()) or 1
            entry["categorical"][key] = {k: round(v / tot, 3)
                                         for k, v in c.most_common(6)}
        # Neighbour surface composition is a strong cue that no single scalar captures:
        # a chamfer sits between two planes, a fillet between whatever it blends.
        nb = Counter()
        for r in recs:
            for s in r.get("neighbour_surfaces") or []:
                nb[s] += 1
        entry["neighbour_surface_share"] = {
            k: round(v / (len(recs) or 1), 3) for k, v in nb.most_common(6)}
        stats["classes"][str(cls)] = entry
    return stats


#: Facts to condition on, with the bin edges used for the numeric ones. Bins are set at
#: the values the classes actually separate on (a full 360-degree bore, a 90-degree blend)
#: rather than at round numbers, so a bin boundary does not split one class in half.
CONDITIONS: List = [
    ("surface_type", None),
    ("concave_surface", None),
    ("coaxial_group_sweep_degrees", [0, 50, 100, 200, 340, 361, 1e9]),
    ("n_tangential_edges", [0, 1, 2, 3, 5, 1e9]),
    ("half_angle_degrees", [0, 20, 40, 50, 65, 1e9]),
    ("radius_vs_part_extent", [0, 0.01, 0.03, 0.1, 0.3, 1e9]),
    ("n_concave_edges", [0, 1, 2, 3, 5, 1e9]),
    ("area_fraction_of_part", [0, 0.001, 0.005, 0.02, 0.1, 1e9]),
    ("on_convex_hull", None),
    ("mean_curvature_sign", None),
    ("is_saddle", None),
    ("strict_component_surfaces", None),
    # Openness bins are set where enclosure changes in kind: 1.0 is a face open to
    # everything (stock), the low end is a face reachable from almost nowhere (a bore or a
    # blind-hole bottom), and the middle is a pocket floor or a counterbore seat.
    ("accessibility", [0, 0.2, 0.5, 0.8, 0.999, 1e9]),
    ("hull_depth_vs_extent", [-1e9, 0.005, 0.03, 0.1, 0.3, 1e9]),
    ("n_inner_wires", [0, 1, 2, 4, 1e9]),
    ("strict_component_size", [0, 2, 3, 5, 9, 1e9]),
    ("strict_component_area_fraction", [0, 0.01, 0.05, 0.2, 0.6, 1e9]),
]


def _bin_label(v: float, edges: List[float]) -> str:
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            lo, hi = edges[i], edges[i + 1]
            if hi >= 1e9:
                return f">={lo:g}"
            if hi - lo <= 1 and float(lo).is_integer():
                return f"{lo:g}"
            return f"{lo:g}-{hi:g}"
    return "?"


def posteriors(by_class: Dict[int, List[Dict]], names: List[str]) -> Dict:
    """P(class | fact) for each fact value, measured on the labelled corpus.

    This is the part the prompt should actually contain. Per-class *marginals* -- "65% of
    chamfers are cones" -- read as if they answer the question but do not: the model is
    shown a cone and needs P(chamfer | cone), and because `random` and `cut` are an order
    of magnitude larger classes, a fact that is characteristic of chamfers can still be
    dominated by them. Inverting the conditional is also the only way to catch a rule that
    is simply false. Two of the rules written by hand from the 40-model sample -- "cone
    means chamfer", "chamfers have no tangential edges" -- did not survive this over the
    full corpus, and the prompt would have asserted both as fact.
    """
    tot = {c: len(v) for c, v in by_class.items()}
    n_all = sum(tot.values()) or 1
    out: Dict = {"prior": {str(c): round(n / n_all, 4) for c, n in sorted(tot.items())},
                 "facts": {}}
    for key, edges in CONDITIONS:
        counts: Dict[str, Counter] = defaultdict(Counter)
        for cls, recs in by_class.items():
            for r in recs:
                v = r.get(key)
                if v is None:
                    continue
                if edges is None:
                    label = str(v)
                else:
                    if not isinstance(v, (int, float)):
                        continue
                    label = _bin_label(float(v), edges)
                counts[label][cls] += 1
        table = {}
        for label, c in counts.items():
            n = sum(c.values())
            if n < 30:      # too few faces to state a share that means anything
                continue
            table[label] = {"n": n, "p": {
                names[cls] if cls < len(names) else str(cls): round(v / n, 3)
                for cls, v in c.most_common()}}
        # Most informative bins first, so the prompt can be truncated without losing the
        # decisive ones: a bin whose top class dominates is what actually settles a face.
        out["facts"][key] = dict(sorted(
            table.items(), key=lambda kv: -max(kv[1]["p"].values())))

    # Single facts leave real ambiguity: a cylinder is a hole, a fillet or a cut wall
    # depending on how far it sweeps, and a plane is `random` or `cut` depending on
    # whether it is concave. Conditioning on the pair resolves exactly those cases, which
    # is where most of the remaining error lives.
    for a, b in (("surface_type", "coaxial_group_sweep_degrees"),
                 ("surface_type", "concave_surface"),
                 ("surface_type", "n_tangential_edges"),
                 # Shape x enclosure. This is the pair that should settle the confusions
                 # shape alone cannot: a plane open to everything is stock, the same plane
                 # reachable from a third of the sphere is a pocket floor; a cylinder swept
                 # 90 degrees is a fillet outside and a bore corner inside.
                 ("surface_type", "accessibility"),
                 ("surface_type", "on_convex_hull"),
                 ("surface_type", "n_inner_wires"),
                 ("strict_component_surfaces", "accessibility")):
        edges_b = dict(CONDITIONS)[b]
        counts: Dict[str, Counter] = defaultdict(Counter)
        for cls, recs in by_class.items():
            for r in recs:
                va, vb = r.get(a), r.get(b)
                if va is None or vb is None:
                    continue
                if edges_b is not None and not isinstance(vb, (int, float)):
                    continue
                lb = _bin_label(float(vb), edges_b) if edges_b else str(vb)
                counts[f"{va} & {b}={lb}"][cls] += 1
        table = {}
        for label, c in counts.items():
            n = sum(c.values())
            if n < 50:
                continue
            table[label] = {"n": n, "p": {
                names[cls] if cls < len(names) else str(cls): round(v / n, 3)
                for cls, v in c.most_common(3)}}
        out["facts"][f"{a} x {b}"] = dict(sorted(
            table.items(), key=lambda kv: -kv[1]["n"]))
    return out


def as_markdown(stats: Dict) -> str:
    rows = ["| class | faces | surface types | area frac p10/50/90 | u sweep p10/50/90 "
            "| tangential edges p50 | convex/concave p50 | neighbours p50 |",
            "|---|---|---|---|---|---|---|---|"]
    for cls, e in stats["classes"].items():
        n = e["numeric"]

        def trio(k):
            v = n.get(k)
            return f"{v['p10']}/{v['p50']}/{v['p90']}" if v else "—"

        st = ", ".join(f"{k} {int(100 * v)}%"
                       for k, v in list(e["categorical"].get("surface_type", {}).items())[:3])
        rows.append(
            f"| {cls} {e['name']} | {e['n_faces']} | {st} | "
            f"{trio('area_fraction_of_part')} | {trio('u_sweep_degrees')} | "
            f"{n.get('n_tangential_edges', {}).get('p50', '—')} | "
            f"{n.get('n_convex_edges', {}).get('p50', '—')}/"
            f"{n.get('n_concave_edges', {}).get('p50', '—')} | "
            f"{n.get('n_neighbours', {}).get('p50', '—')} |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="brepformer/data/defeature/steps")
    ap.add_argument("--labels", default="brepformer/data/defeature/labels")
    ap.add_argument("--label_space", choices=["5", "27"], default="5")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    ap.add_argument("--out", type=Path, default=DEFAULT_STATS)
    ap.add_argument("--dump_faces", type=Path,
                    default=Path(__file__).resolve().parent / "class_geometry_faces.jsonl.gz",
                    help="Cache the raw per-face records here; pass '' to skip")
    args = ap.parse_args()

    names = DEFEATURE_CLASS_NAMES if args.label_space == "5" else CLASS_NAMES
    steps = Path(args.steps)
    labels = Path(args.labels)
    jobs = []
    for lp in sorted(labels.glob("*.json")):
        for ext in (".step", ".stp", ".STEP"):
            sp = steps / (lp.stem + ext)
            if sp.exists():
                jobs.append((sp, lp))
                break
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(jobs)} labelled models, {args.workers} workers", flush=True)

    by_class: Dict[int, List[Dict]] = defaultdict(list)
    errors = []
    done = 0
    with mp.Pool(args.workers) as pool:
        for res in pool.imap_unordered(_one_model, jobs, chunksize=1):
            done += 1
            if res and "__error__" in res:
                errors.append(res["__error__"])
            elif res:
                for cls, recs in res.items():
                    by_class[cls].extend(recs)
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)} models, "
                      f"{sum(len(v) for v in by_class.values())} faces, "
                      f"{len(errors)} failed", flush=True)

    # Cache the raw per-face records. Extraction over the full corpus costs minutes of
    # OCC work; every future question ("what about torus faces specifically?") is then a
    # second of arithmetic on this file instead of a re-run.
    if args.dump_faces:
        with gzip.open(args.dump_faces, "wt") as fh:
            for cls, recs in by_class.items():
                for r in recs:
                    fh.write(json.dumps({"cls": cls, **r}) + "\n")
        print(f"wrote per-face records to {args.dump_faces}")

    stats = summarise(by_class, names)
    stats["posteriors"] = posteriors(by_class, names)
    stats["meta"] = {"models": len(jobs), "models_failed": len(errors),
                     "label_space": args.label_space, "steps": str(steps),
                     "faces": sum(len(v) for v in by_class.values())}
    args.out.write_text(json.dumps(stats, indent=2))
    print(f"\n{stats['meta']}\nwrote {args.out}\n")
    print(as_markdown(stats))
    print("\nP(class | fact) — the tables the prompt quotes:")
    for fact, table in stats["posteriors"]["facts"].items():
        print(f"\n  {fact}:")
        for label, e in list(table.items())[:8]:
            top = ", ".join(f"{k} {int(100 * v)}%" for k, v in list(e["p"].items())[:3])
            print(f"    {label:<44} n={e['n']:<6} {top}")
    if errors:
        print(f"\n{len(errors)} models failed, first 5:")
        for e in errors[:5]:
            print("  " + e)


if __name__ == "__main__":
    main()
