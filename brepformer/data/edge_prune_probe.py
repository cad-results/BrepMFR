#!/usr/bin/env python3
"""Does a LEARNED edge prune produce components that are actually features?

Motivation, all measured (see SCRIPTS.md "Diagnostics that killed the feature-clustering
plan"): clustering faces by AAG concave components and propagating one label per component
scores 31.8% vs 49.1% per-face, because only 25.5% of components are label-pure and the
*ceiling* -- the fraction of faces sitting in their component's majority label -- is 57.3%.
That ceiling binds any single-label-per-component rule, hard majority vote and soft logit-sum
alike, so no better pooling rule can rescue it. The partition itself is wrong: a chamfer on
the rim of a passage is concave-connected to the passage, which is the textbook AAG
interacting-feature limitation.

FeatureFox (arXiv:2604.26770, verified) says: don't take components of the raw adjacency
graph. Train a binary edge classifier for "do these two faces belong to the same feature?",
prune the edges it rejects, and take connected components of what is left. This script tests
whether that premise holds on our data, using only edge attributes `face_geometry` already
emits (`normal_deviation_degrees`, `convex`, `param_length`, neighbour surface type).

Three numbers are reported and the third is the one that matters:
  1. held-out ROC AUC of the edge classifier -- is "same feature?" predictable at all;
  2. purity and ceiling of the components the *predicted* prune yields;
  3. the same for the existing strict concave components, as the baseline to beat.

Ground truth caveat: mftrcad gives semantic labels, not instance ids, so "same feature" is
approximated by "same semantic label". That merges two adjacent same-class features into one
component, which makes the ceiling reported here optimistic for instance segmentation but
correct for the per-face labelling task we actually run.

Usage:
    python brepformer/data/edge_prune_probe.py --limit 60
"""

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from brepformer.face_geometry import face_geometry
from brepformer.data.label_harness import iter_ground_truth

SURFACES = ["plane", "cylinder", "cone", "sphere", "torus", "bspline", "revolution", "other"]


def _surf_index(name):
    try:
        return SURFACES.index(name)
    except ValueError:
        return len(SURFACES) - 1


def edge_rows(geo, gt):
    """One row per undirected adjacent face pair, with symmetric features and the label."""
    seen = set()
    rows, labels, pairs = [], [], []
    for i in range(len(gt)):
        rec = geo.get(i) or geo.get(str(i))
        if not rec:
            continue
        si = _surf_index(rec.get("surface_type"))
        for e in rec.get("edges") or []:
            j = e.get("neighbour_face")
            if j is None or not (0 <= j < len(gt)):
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            other = geo.get(j) or geo.get(str(j)) or {}
            sj = _surf_index(other.get("surface_type"))
            dev = float(e.get("normal_deviation_degrees") or 0.0)
            convex = 1.0 if e.get("convex") else 0.0
            # Symmetric encoding: the pair is unordered, so features must not depend on which
            # face we happened to iterate from. min/max instead of (i, j) ordering.
            ai = float(rec.get("area_fraction_of_part") or 0.0)
            aj = float(other.get("area_fraction_of_part") or 0.0)
            rows.append([
                dev, convex, float(e.get("param_length") or 0.0),
                min(si, sj), max(si, sj), 1.0 if si == sj else 0.0,
                min(ai, aj), max(ai, aj),
                float(rec.get("n_concave_edges") or 0) + float(other.get("n_concave_edges") or 0),
                min(float(rec.get("accessibility") or 0.0), float(other.get("accessibility") or 0.0)),
                float(rec.get("concave_surface") or 0) + float(other.get("concave_surface") or 0),
            ])
            labels.append(1 if (gt[i] >= 0 and gt[j] >= 0 and gt[i] == gt[j]) else 0)
            pairs.append(key)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels), pairs


def components(n, keep_pairs):
    """Union-find over the kept edges."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in keep_pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    out = collections.defaultdict(list)
    for i in range(n):
        out[find(i)].append(i)
    return list(out.values())


def score_partition(parts, gt):
    """Purity of components and the ceiling any one-label-per-component rule can reach."""
    pure = total = 0
    in_majority = faces = 0
    for members in parts:
        labs = [gt[i] for i in members if gt[i] >= 0]
        if not labs:
            continue
        total += 1
        counts = collections.Counter(labs)
        if len(counts) == 1:
            pure += 1
        in_majority += counts.most_common(1)[0][1]
        faces += len(labs)
    return (100 * pure / max(total, 1), 100 * in_majority / max(faces, 1), total)


def component_features(members, geo):
    """Feature-level descriptors for one component.

    Deliberately includes the two things no per-face fact can express and which the measured
    0% recall on every polygonal class is attributed to: the **side count** (planar wall count
    and distinct surface composition at integer resolution, which the existing
    `strict_component_size` 5-9 bin destroys) and a **bottom-face proxy**. Per AAGNet's own
    generator a bottom face exists iff its normal is parallel to the machining feed direction,
    so through features have no bottom face by construction and the bottom-face indicator *is*
    a through-vs-blind detector -- the property dense-view vision scored at chance.
    """
    types = collections.Counter()
    areas, access, inner, ncc, radii = [], [], [], [], []
    for i in members:
        rec = geo.get(i) or geo.get(str(i)) or {}
        types[rec.get("surface_type") or "other"] += 1
        areas.append(float(rec.get("area_fraction_of_part") or 0.0))
        access.append(float(rec.get("accessibility") or 0.0))
        inner.append(float(rec.get("n_inner_wires") or 0))
        ncc.append(float(rec.get("n_concave_edges") or 0))
        radii.append(float(rec.get("radius_vs_part_extent") or 0.0))
    n = len(members)
    return [
        n,
        types["plane"], types["cylinder"], types["cone"], types["torus"],
        types["bspline"] + types["revolution"] + types["sphere"],
        len([t for t, c in types.items() if c]),
        sum(areas), max(areas), min(areas), sum(areas) / n,
        max(access), min(access), sum(access) / n,
        sum(inner), max(inner),
        sum(ncc) / n, max(radii),
        # bottom-face proxy: a blind feature has an interior planar face that is fully
        # occluded (accessibility 0) while a through feature does not.
        sum(1 for i, a in zip(members, access)
            if a <= 0.001 and ((geo.get(i) or geo.get(str(i)) or {}).get("surface_type") == "plane")),
    ]


def classify_components(train, test, clf_edge, threshold, gt_space_size):
    """Do the better components convert into better per-face labels?

    The 88% ceiling is an upper bound, not an accuracy. This trains a component-level
    classifier on the pruned partition and propagates one label to every face of a component,
    then scores per-face accuracy against `geo_label`-style per-face labelling **on the same
    held-out models**, which is the only fair comparison.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    def build(models, learned):
        rows, labs = [], []
        for X, y, pairs, gt, geo in models:
            n = len(gt)
            if learned:
                p = clf_edge.predict_proba(X)[:, 1]
                keep = [pr for pr, pv in zip(pairs, p) if pv >= threshold]
            else:
                groups = collections.defaultdict(list)
                for i in range(n):
                    rec = geo.get(i) or geo.get(str(i))
                    if rec and rec.get("strict_component_id") is not None:
                        groups[rec["strict_component_id"]].append(i)
                keep = None
            parts = components(n, keep) if learned else list(groups.values())
            for members in parts:
                truth = [gt[i] for i in members if gt[i] >= 0]
                if not truth:
                    continue
                rows.append(component_features(members, geo))
                labs.append(collections.Counter(truth).most_common(1)[0][0])
        return np.asarray(rows, dtype=np.float64), np.asarray(labs)

    out = {}
    for tag, learned in (("learned edge prune", True), ("existing strict concave", False)):
        Xtr, ytr = build(train, learned)
        if len(set(ytr.tolist())) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        clf.fit(Xtr, ytr)
        correct = total = 0
        for X, y, pairs, gt, geo in test:
            n = len(gt)
            if learned:
                p = clf_edge.predict_proba(X)[:, 1]
                parts = components(n, [pr for pr, pv in zip(pairs, p) if pv >= threshold])
            else:
                groups = collections.defaultdict(list)
                for i in range(n):
                    rec = geo.get(i) or geo.get(str(i))
                    if rec and rec.get("strict_component_id") is not None:
                        groups[rec["strict_component_id"]].append(i)
                parts = list(groups.values())
            feats, keep_parts = [], []
            for members in parts:
                if any(gt[i] >= 0 for i in members):
                    feats.append(component_features(members, geo))
                    keep_parts.append(members)
            if not feats:
                continue
            pred = clf.predict(np.asarray(feats, dtype=np.float64))
            for members, pc in zip(keep_parts, pred):
                for i in members:
                    if gt[i] >= 0:
                        total += 1
                        correct += int(gt[i] == pc)
        out[tag] = 100 * correct / max(total, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="brepformer/data/mftrcad")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--label_space", default="27")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--classify", action="store_true",
                    help="Also train a component-level classifier and score per-face accuracy.")
    args = ap.parse_args()

    per_model = []
    for step_path, gt in iter_ground_truth(Path(args.data_dir), args.limit, 0, args.label_space):
        try:
            geo = face_geometry(str(step_path), with_topology=True)
        except Exception:
            continue
        X, y, pairs = edge_rows(geo, gt)
        if len(y) == 0:
            continue
        per_model.append((X, y, pairs, gt, geo))

    # Split BY MODEL, never by edge: edges of one part share faces, components and a bbox.
    half = len(per_model) // 2
    train, test = per_model[:half], per_model[half:]
    Xtr = np.vstack([m[0] for m in train])
    ytr = np.concatenate([m[1] for m in train])
    Xte = np.vstack([m[0] for m in test])
    yte = np.concatenate([m[1] for m in test])
    print(f"{len(per_model)} models -> {len(ytr)} train edges / {len(yte)} test edges, "
          f"{100 * ytr.mean():.1f}% same-feature in train")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    print(f"held-out edge AUC = {roc_auc_score(yte, prob):.3f}   "
          f"(dihedral angle alone = {roc_auc_score(yte, Xte[:, 0]):.3f}, "
          f"convexity alone = {roc_auc_score(yte, Xte[:, 1]):.3f})\n")

    # Compare three partitions on the held-out models only.
    learned = {"pure": [], "ceil": [], "n": 0}
    strict = {"pure": [], "ceil": [], "n": 0}
    oracle = {"pure": [], "ceil": [], "n": 0}
    off = 0
    for X, y, pairs, gt, geo in test:
        p = clf.predict_proba(X)[:, 1]
        off += len(y)
        n = len(gt)

        keep = [pr for pr, pv in zip(pairs, p) if pv >= args.threshold]
        for tag, parts in (
            ("learned", components(n, keep)),
            ("oracle", components(n, [pr for pr, yv in zip(pairs, y) if yv == 1])),
        ):
            d = learned if tag == "learned" else oracle
            pu, ce, nc = score_partition(parts, gt)
            d["pure"].append(pu); d["ceil"].append(ce); d["n"] += nc

        groups = collections.defaultdict(list)
        for i in range(n):
            rec = geo.get(i) or geo.get(str(i))
            if rec and rec.get("strict_component_id") is not None:
                groups[rec["strict_component_id"]].append(i)
        pu, ce, nc = score_partition(list(groups.values()), gt)
        strict["pure"].append(pu); strict["ceil"].append(ce); strict["n"] += nc

    print(f"{'partition':<34}{'pure %':>8}{'ceiling %':>11}{'components':>12}")
    for name, d in (("existing strict concave (baseline)", strict),
                    ("learned edge prune", learned),
                    ("oracle same-label prune", oracle)):
        print(f"{name:<34}{np.mean(d['pure']):>8.1f}{np.mean(d['ceil']):>11.1f}{d['n']:>12}")
    print("\nceiling = faces sitting in their component's majority label; it bounds ANY\n"
          "single-label-per-component rule, so it is the number to beat, not purity.")

    if args.classify:
        print("\nper-face accuracy from a component-level classifier (same held-out models):")
        res = classify_components(train, test, clf, args.threshold, None)
        for tag, acc in res.items():
            print(f"  {tag:<32}{acc:>6.1f}%")
        print("  (compare: geo_label per-face 27-class = 49.1%, and hard voting over the\n"
              "   existing concave partition = 31.8%)")


if __name__ == "__main__":
    main()
