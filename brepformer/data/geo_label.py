#!/usr/bin/env python3
"""Label faces from exact B-Rep geometry alone, using the *measured* evidence table.

Why this exists
---------------
`measure_class_geometry.py` already measured P(class | geometric fact) over 12,000 mftrcad
models and 305,785 labelled faces, for 24 facts including surface type, accessibility,
edge convexity counts, tangential-edge counts and AAG component composition. The result
sits in `class_geometry_stats*.json` and **nothing has ever used it to label a face** --
`vision_label.py` pastes a few rows into a prompt, and that is all.

That is the largest unexploited asset in the labelling pipeline, for three reasons:

1. **It is measured, not guessed.** Every hand-written geometric rule in this repo was
   wrong when finally checked: `geometric_rules.py` corrupted 15% of correct labels, and
   its thresholds were off by more than an order of magnitude (`chamfer_2` has a median
   area fraction of 0.158 against a 0.10 cap). This table has no thresholds to guess --
   the bins and their class posteriors both come from the corpus.

2. **It can see what the network cannot.** The model's `face_attr` reads an index layout
   the data never had, so surface type collapses to plane/not-plane and edge convexity is
   identically zero (see README "Feature Representation"). Convexity is the single best
   fillet-vs-chamfer discriminator and it is dead in the network. Here it is a first-class
   fact.

3. **It is independent of the teacher.** The current synthetic-label pipeline is
   teacher -> confidence mask -> train, so every pseudo-label inherits the teacher's
   errors and there is no second opinion anywhere in the chain. Geometry is a genuinely
   different view, which is what makes agreement between the two worth something.

How the facts are combined
--------------------------
The table stores P(c | fact), so the likelihood ratio of a fact is P(c | fact) / P(c) and
evidence combines additively in log space:

    log score(c) = log P(c) + (1/k**alpha) * sum_f log( P(c | bin_f(face)) / P(c) )

with `alpha = 0` -- a plain naive-Bayes sum. This file was written with `alpha = 1` (average
the evidence, so that correlated facts cannot vote k times) and that was wrong: the average
also rescales every *informative* fact by k/(k+1) whenever an uninformative one is added, so
it cost more than half the macro-recall (0.307 -> 0.128) and made fact selection halt after
three facts. Correlation is a calibration problem, not a ranking problem, and it is handled
downstream by `--temperature`, which cannot reorder anything. See DEFAULT_ALPHA.

Two attempts to pick a good subset of facts also lost to simply using all of them, the
greedy one only on the models it was not chosen on -- see DEFAULT_FACTS. `--select_facts`
survives as a diagnostic that reports both halves of the split.

Output is `infer.py`-compatible (`face_preds` + `face_probs`), so `label_harness.py`
scores this exactly like a checkpoint, including `--mode select` risk-coverage. That is
deliberate: a labeller that cannot be measured by the existing harness should not be
trusted, and this one makes no claim that has not been through it.

Usage
-----
    # label one part
    python -m brepformer.data.geo_label --step model.step --out /tmp/pred.json

    # batch, then score against ground truth with the harness
    python -m brepformer.data.geo_label --step_dir /tmp/mftrcad_select_sample \\
        --out_dir /tmp/geo_preds
    python -m brepformer.data.label_harness --mode select \\
        --predictions /tmp/geo_preds --limit 60 --label_space 27

    # diagnostics: which facts carry the decision, and how to pool them
    python -m brepformer.data.geo_label --select_facts --limit 120
    python -m brepformer.data.geo_label --sweep_alpha --limit 40
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from brepformer.data.classes import CLASS_NAMES, DEFEATURE_CLASS_NAMES
from brepformer.data.measure_class_geometry import CONDITIONS, _bin_label

#: `None` means "every fact in the table". That is the measured answer, not laziness, and it
#: is the opposite of what this file was first written to do.
#:
#: Two hand-built alternatives were tried and both lost on models the choice was not made
#: on (120 models, `--select_facts`, macro-recall on the held-out half):
#:
#:     all 24 facts                       0.3121   <- generalises best
#:     greedy-selected 7 facts            0.2877   (0.3092 on the selection half)
#:     9 hand-picked "complementary"      0.2520
#:
#: The greedy subset beat all-24 on the half it was chosen on and lost on the other half,
#: which is the signature of fitting the sample: with ~1400 faces spread over 27 classes, a
#: fact's apparent gain is mostly which rare class it happened to catch. Excluding a fact
#: is a claim that needs more evidence than including one, because a weak fact under
#: alpha=0 pooling costs almost nothing while a missing fact costs everything it knew.
#:
#: So `--select_facts` is kept as a *diagnostic* -- it shows which facts carry the decision
#: and reports both halves so the overfit stays visible -- and not as a configuration step.
DEFAULT_FACTS = None

#: The two sets that lost, kept so `--select_facts` and `--sweep_alpha` can re-derive the
#: comparison above instead of asking anyone to take it on trust.
HAND_PICKED_FACTS = [
    "surface_type x accessibility",
    "surface_type x concave_surface",
    "strict_component_surfaces x accessibility",
    "n_tangential_edges",
    "n_concave_edges",
    "area_fraction_of_part",
    "strict_component_size",
    "n_inner_wires",
    "on_convex_hull",
]
GREEDY_FACTS_120 = [
    "surface_type x n_tangential_edges",
    "surface_type x accessibility",
    "area_fraction_of_part",
    "hull_depth_vs_extent",
    "surface_type x coaxial_group_sweep_degrees",
    "mean_curvature_sign",
    "n_inner_wires",
]

#: How k facts are pooled: `scale = 1 / k**alpha`.
#:
#:   alpha = 0  sum      -- textbook naive Bayes. Treats the facts as independent, which
#:                          they are not, so confidence saturates at 0/1.
#:   alpha = 1  mean     -- log-opinion pool. Immune to correlation, but *monotonically
#:                          diluting*: an uninformative fact contributes a ~0 log-ratio and
#:                          still rescales every informative one by k/(k+1), so adding facts
#:                          can only ever weaken the decision.
#:
#: The dilution is not hypothetical. Measured on 881 faces / 40 models (`--sweep_alpha`),
#: macro-recall over all 24 facts: 0.307 at alpha=0, 0.242 at 0.5, 0.128 at 1.0 -- pooling
#: as a mean throws away well over half the signal, and it also made greedy selection halt
#: after three facts because a fourth could only dilute. alpha=0 wins on macro-recall at
#: every fact count; alpha=0.25 edges it on plain accuracy (0.539 vs 0.499) by favouring the
#: big classes, which is not the trade this pipeline wants -- the rare classes (`chamfer`,
#: `round`) are the ones it is trying to label.
#:
#: The cost of alpha=0 is real but lands somewhere harmless: summing correlated evidence
#: saturates the *confidence*, so these posteriors rank well and are not calibrated. Fix
#: that with `--temperature`, fitted against `label_harness --mode select`, rather than by
#: pooling -- calibration is a monotone rescale and cannot damage the ranking, while pooling
#: as a mean demonstrably does.
DEFAULT_ALPHA = 0.0

#: Pooling exponent for the *confidence*, as opposed to the decision. Keyed by class count.
#:
#: The decision and the gate are two different questions asked of the same evidence -- "which
#: class" and "how much should this be trusted" -- and nothing requires one number to answer
#: both. Measured on 1412 faces / 60 mftrcad models, decide-alpha x gate-alpha:
#:
#:   5-class    decide 0, gate 0  ->  acc .7224  macro .7476  AURC .2345
#:              decide 1, gate 1  ->  acc .7167  macro .5212  AURC .2039
#:              decide 0, gate 1  ->  acc .7224  macro .7476  AURC .1851   <- dominates both
#:  27-class    decide 0, gate 0.5 -> acc .4894  macro .2998  AURC .2169   <- best
#:
#: Those are **AURC**, not AUGRC -- this file originally called the harness metric AUGRC when
#: it divided by the accepted count. Re-checked 2026-09-02 against the corrected AUGRC and the
#: choice survives: the production decoupled config (decide 0, gate 1) scores AUGRC .1069,
#: better than every row of the joint `--sweep_calibration` sweep, whose own best is .1112 at
#: alpha=1.0/temp=0.25 and which gives up macro recall (.7370 vs .7476) to get there.
#:
#: Decoupling is free: the gate never changes a label, so it cannot cost accuracy, and it
#: bought a fifth off the 5-class AUGRC. The direction differs by label space because
#: `cut` (21 fine classes) accumulates agreeing facts that a summed pool reads as certainty
#: -- normalising by fact count is what stops confident-and-wrong `cut` from crowding the
#: top of the ranking. In the 27-class space the classes are narrow enough that heavy
#: normalising discards real evidence, so the best gate sits near the middle.
GATE_ALPHA = {5: 1.0, 27: 0.5}

#: Floor for a class a bin never observed. The tables drop bins with n<30 and the cross
#: tables keep only the top 3 classes, so a missing entry means "rare here", not
#: "impossible" -- a hard zero would let one sparse bin veto a class outright.
EPS = 0.002


def all_facts(table: dict) -> list:
    """Every fact with at least one surviving bin -- the measured default. See DEFAULT_FACTS."""
    return [f for f in table["facts"] if table["facts"][f]]


def _complete_row(row: dict, prior: dict) -> dict:
    """Give the classes a truncated table row omits their share of the leftover mass.

    The single-fact tables list every class, but the 7 cross tables keep only
    `most_common(3)`, so a row like ``plane & accessibility=0.5-0.8`` sums to 0.71 and says
    nothing about the other 24 classes. Flooring them at a constant is wrong in a way that
    matters: it makes the floor a *likelihood ratio* of EPS/prior, which is a large penalty
    for a common class (stock, 0.348) and almost none for a rare one (chamfer, 0.0056) --
    exactly backwards, and it is why the cross-heavy DEFAULT_FACTS scored 0.074 macro-recall
    against 0.229 for three single facts.

    Spreading the residual proportional to the prior instead says the honest thing: within
    the classes this row did not resolve, believe the corpus. Their likelihood ratio then
    lands near the residual fraction for all of them, so a truncated row contributes
    evidence about its top 3 and stays neutral elsewhere.
    """
    listed = sum(row.values())
    residual = 1.0 - listed
    if residual <= 1e-6:
        return dict(row)
    rest = {c: p for c, p in prior.items() if c not in row}
    z = sum(rest.values())
    out = dict(row)
    for c, p in rest.items():
        out[c] = residual * (p / z) if z > 0 else EPS
    return out


def load_table(path: Path) -> dict:
    """Load the measured evidence table and index it by class id rather than name."""
    raw = json.loads(path.read_text())
    names = CLASS_NAMES if len(raw["posteriors"]["prior"]) > 8 else DEFEATURE_CLASS_NAMES
    name_to_id = {n: i for i, n in enumerate(names)}

    prior = {int(k): v for k, v in raw["posteriors"]["prior"].items()}

    facts = {}
    for fact, bins in raw["posteriors"]["facts"].items():
        facts[fact] = {
            b: _complete_row(
                {name_to_id[n]: p for n, p in rec["p"].items() if n in name_to_id},
                prior)
            for b, rec in bins.items()
        }
    # From `names`, not from the prior: a class absent from the corpus would silently
    # shorten every posterior and shift every class id past it.
    return {"prior": prior, "facts": facts, "names": names,
            "n_classes": len(names)}


def face_facts(rec: dict) -> dict:
    """Bin one face's raw geometry record into the same labels the table was built with.

    Reuses `CONDITIONS` and `_bin_label` from the measuring script rather than
    reimplementing the edges: a bin boundary that differs by so much as a rounding rule
    silently looks up the wrong row, and the result would still be a plausible-looking
    posterior.
    """
    out = {}
    for key, edges in CONDITIONS:
        v = rec.get(key)
        if v is None:
            continue
        if edges is None:
            out[key] = str(v)
        elif isinstance(v, (int, float)):
            out[key] = _bin_label(float(v), edges)
    # The cross facts are keyed "a x b" with bin label "<a value> & b=<b bin>", matching
    # `posteriors()`.
    for a, b in (("surface_type", "coaxial_group_sweep_degrees"),
                 ("surface_type", "concave_surface"),
                 ("surface_type", "n_tangential_edges"),
                 ("surface_type", "accessibility"),
                 ("surface_type", "on_convex_hull"),
                 ("surface_type", "n_inner_wires"),
                 ("strict_component_surfaces", "accessibility")):
        va, vb = rec.get(a), out.get(b)
        if va is None or vb is None:
            continue
        out[f"{a} x {b}"] = f"{va} & {b}={vb}"
    return out


def posterior(binned: dict, table: dict, facts: list, temperature: float = 1.0,
              alpha: float = DEFAULT_ALPHA) -> list:
    """P(class | facts) for one face as a list over class ids.

    Returns the prior unchanged when no fact in `facts` applies to this face, which is the
    honest answer: an unlabellable face should look maximally uncertain, not confidently
    like whatever the last fact said.
    """
    prior, ftab, n = table["prior"], table["facts"], table["n_classes"]
    logs = [math.log(max(prior.get(c, EPS), EPS)) for c in range(n)]

    used = 0
    acc = [0.0] * n
    for fact in facts:
        bins = ftab.get(fact)
        if not bins:
            continue
        row = bins.get(binned.get(fact))
        if not row:
            continue
        used += 1
        for c in range(n):
            p_c_given_f = row.get(c, EPS)
            acc[c] += math.log(max(p_c_given_f, EPS) / max(prior.get(c, EPS), EPS))

    if used:
        # alpha interpolates between naive Bayes (0, sum) and a log-opinion pool (1, mean).
        # See DEFAULT_ALPHA: which end wins is measured, not argued.
        scale = 1.0 / (used ** alpha * max(temperature, 1e-6))
        logs = [lg + scale * a for lg, a in zip(logs, acc)]

    m = max(logs)
    exp = [math.exp(x - m) for x in logs]
    s = sum(exp)
    return [e / s for e in exp]


def label_step(step_path: Path, table: dict, facts: list, temperature: float = 1.0,
               alpha: float = DEFAULT_ALPHA, gate_alpha: float | None = None) -> dict:
    """Label every face of one STEP file. Output matches `infer.py`'s JSON schema.

    Emits `face_probs` from the decision pooling -- so the distribution's argmax really is
    the reported label -- and `face_confidence` from the gate pooling, which rates that
    decision without being able to change it. See GATE_ALPHA.
    """
    from brepformer.face_geometry import face_geometry

    geo = face_geometry(str(step_path), with_topology=True)
    n_faces = (max(int(k) for k in geo) + 1) if geo else 0
    names = table["names"]
    if gate_alpha is None:
        gate_alpha = GATE_ALPHA.get(table["n_classes"], alpha)

    probs, preds, conf = [], [], []
    for i in range(n_faces):
        rec = geo.get(i) or geo.get(str(i))
        if rec is None:
            # No geometry for this face: say so with a flat posterior rather than letting
            # it inherit whatever the prior happens to favour.
            p = [1.0 / table["n_classes"]] * table["n_classes"]
            c = p[0]
        else:
            binned = face_facts(rec)
            p = posterior(binned, table, facts, temperature, alpha)
            pred = max(range(len(p)), key=p.__getitem__)
            g = (p if gate_alpha == alpha
                 else posterior(binned, table, facts, temperature, gate_alpha))
            c = g[pred]
        probs.append([round(x, 5) for x in p])
        preds.append(int(max(range(len(p)), key=p.__getitem__)))
        conf.append(round(c, 5))

    return {
        "model_id": step_path.stem,
        "num_faces": n_faces,
        "face_preds": preds,
        "face_class_names": [names[c] if c < len(names) else str(c) for c in preds],
        "face_probs": probs,
        "face_confidence": conf,
        "source": "geo_label",
        "geo_label": {"alpha": alpha, "gate_alpha": gate_alpha,
                      "temperature": temperature, "n_facts": len(facts)},
    }


# ============================================================
# Measured fact selection
# ============================================================

def _eval_facts(samples: list, table: dict, facts: list, temperature: float,
                alpha: float = DEFAULT_ALPHA, full: bool = False):
    """Macro-recall of a fact subset.

    Macro, not accuracy: `stock` is 35% of all faces, so accuracy is maximised by a
    labeller that says `stock` and nothing else -- which is exactly the degenerate solution
    this whole file exists to beat.
    """
    import collections
    per = collections.defaultdict(lambda: [0, 0])
    hits = 0
    for binned, truth in samples:
        p = posterior(binned, table, facts, temperature, alpha)
        pred = max(range(len(p)), key=p.__getitem__)
        per[truth][0] += 1
        per[truth][1] += pred == truth
        hits += pred == truth
    if not per:
        return (0.0, 0.0, {}) if full else 0.0
    macro = sum(k / n for n, k in per.values()) / len(per)
    if full:
        return macro, hits / len(samples), dict(per)
    return macro


def collect_samples(data_dir: Path, limit: int, seed: int, label_space: str) -> list:
    """(binned facts, true class, model index) for every labelled face of `limit` models."""
    from brepformer.data.label_harness import iter_ground_truth
    from brepformer.face_geometry import face_geometry

    out = []
    for m, (step_path, gt) in enumerate(
            iter_ground_truth(data_dir, limit, seed, label_space)):
        try:
            geo = face_geometry(str(step_path), with_topology=True)
        except Exception:  # noqa: BLE001
            continue
        for i, truth in enumerate(gt):
            if truth < 0:
                continue
            rec = geo.get(i) or geo.get(str(i))
            if rec is None:
                continue
            out.append((face_facts(rec), int(truth), m))
    return out


def split_by_model(samples: list, frac: float = 0.5) -> tuple:
    """Split faces into two sets along *model* boundaries.

    By model, not by face: faces of one part share a component, an accessibility field and
    a bounding box, so a face-level split leaks its part's answer into the other half and
    every selected fact would look better than it is.
    """
    models = sorted({m for _, _, m in samples})
    cut = models[int(len(models) * frac)] if models else 0
    a = [(f, t) for f, t, m in samples if m < cut]
    b = [(f, t) for f, t, m in samples if m >= cut]
    return a, b


def select_facts(samples: list, table: dict, temperature: float,
                 alpha: float = DEFAULT_ALPHA) -> list:
    """Greedy forward selection over the available facts, scored by macro-recall.

    Forward selection specifically: correlated facts are the failure mode here, and adding
    facts one at a time and keeping only those that *improve* the score is what detects a
    fact whose information is already present. Ranking facts individually and taking the
    top k would keep three views of the same surface and drop enclosure entirely.
    """
    candidates = [f for f in table["facts"] if table["facts"][f]]
    chosen: list = []
    best = _eval_facts(samples, table, chosen, temperature, alpha)
    print(f"prior only: macro-recall {best:.4f}")
    while True:
        gains = []
        for f in candidates:
            if f in chosen:
                continue
            s = _eval_facts(samples, table, chosen + [f], temperature, alpha)
            gains.append((s, f))
        if not gains:
            break
        gains.sort(reverse=True)
        s, f = gains[0]
        if s <= best + 1e-4:
            print(f"stop: best remaining addition {f} would give {s:.4f} "
                  f"(no better than {best:.4f})")
            break
        chosen.append(f)
        best = s
        print(f"  + {f:<44} macro-recall {best:.4f}")
    return chosen


def sweep_calibration(samples: list, table: dict, facts: list) -> None:
    """Grid alpha x temperature against accuracy *and* AUGRC, which disagree.

    They have to be reported together. alpha=0 maximises what the labeller gets right, but
    it also saturates confidence -- every face with many agreeing facts lands at 1.000, so
    the threshold has nothing left to sort by, and the 5-class table's top-decile precision
    (69.6%) came out *below* its own overall accuracy (72.6%). A labeller whose most
    confident faces are its worse ones is useless for auto-accept no matter how accurate it
    is on average, because auto-accept only ever reads the top of that ranking.

    Note that temperature is *not* a no-op here even though it is a monotone rescale of one
    face's logits: faces differ in how many facts applied, so dividing the summed evidence
    changes how faces compare *to each other*, which is exactly the ordering AUGRC scores.
    """
    from brepformer.data.label_harness import compute_augrc

    print(f"{'':>6}{'':>7}  {'accuracy':>9}  {'macro':>7}  {'AUGRC':>7}  {'base':>7}")
    print(f"{'alpha':>6}{'temp':>7}  {'(argmax)':>9}  {'recall':>7}  {'(rank)':>7}  {'AUGRC':>7}")
    best = None
    for alpha in (0.0, 0.25, 0.5, 1.0):
        for temp in (0.25, 1.0, 4.0, 16.0):
            rows, per = [], {}
            for binned, truth in samples:
                p = posterior(binned, table, facts, temp, alpha)
                pred = max(range(len(p)), key=p.__getitem__)
                rows.append((p[pred], pred == truth))
                d = per.setdefault(truth, [0, 0])
                d[0] += 1
                d[1] += pred == truth
            acc = sum(1 for _, ok in rows if ok) / len(rows)
            macro = sum(k / n for n, k in per.values()) / len(per)
            augrc = compute_augrc(rows)
            # AUGRC's uninformative baseline is HALF the error rate, not the error rate: a
            # random ranking has AUROC_f = 0.5, so the closed form collapses to (1-acc)/2.
            # The error rate is the *AURC* baseline. See label_harness.compute_risk_coverage.
            base = (1 - acc) / 2
            flag = ""
            if best is None or augrc < best[0]:
                best, flag = (augrc, alpha, temp), " <- best AUGRC"
            print(f"{alpha:>6.2f}{temp:>7.2f}  {acc:>9.4f}  {macro:>7.4f}  "
                  f"{augrc:>7.4f}  {base:>7.4f}{flag}")
    print(f"\nbest AUGRC {best[0]:.4f} at alpha={best[1]}, temperature={best[2]}.")
    print("`base AUGRC` is what uninformative confidence would score at that accuracy, so a "
          "row is only useful when AUGRC sits well below it.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", type=Path,
                    default=Path("brepformer/data/class_geometry_stats_27.json"),
                    help="Measured evidence table from measure_class_geometry.py.")
    ap.add_argument("--step", type=Path, default=None)
    ap.add_argument("--step_dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--facts", nargs="*", default=None,
                    help="Facts to combine. Default: every fact in the table, which "
                         "measured better than any subset tried (see DEFAULT_FACTS).")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help=">1 flattens the posterior, <1 sharpens it. Fit it with "
                         "label_harness --mode select, do not guess it.")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help="Pooling exponent: 0 = naive-Bayes sum, 1 = mean. See "
                         "DEFAULT_ALPHA.")
    ap.add_argument("--gate_alpha", type=float, default=None,
                    help="Pooling exponent for the confidence only; cannot change a label. "
                         "Default: per label space, see GATE_ALPHA.")
    ap.add_argument("--select_facts", action="store_true",
                    help="Greedy forward selection on mftrcad ground truth, then exit.")
    ap.add_argument("--sweep_calibration", action="store_true",
                    help="Grid alpha x temperature against accuracy and AUGRC, then exit.")
    ap.add_argument("--sweep_alpha", action="store_true",
                    help="Score every alpha on both fact sets, then exit. Run this before "
                         "trusting DEFAULT_ALPHA on a new stats table.")
    ap.add_argument("--data_dir", type=Path, default=Path("brepformer/data/mftrcad"))
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label_space", default="27", choices=["27", "5"])
    args = ap.parse_args()

    table = load_table(args.stats)
    facts = args.facts if args.facts is not None else all_facts(table)

    if args.sweep_calibration:
        raw = collect_samples(args.data_dir, args.limit, args.seed, args.label_space)
        print(f"{len(raw)} labelled faces from {args.limit} models, "
              f"{len(facts)} facts, label_space={args.label_space}\n")
        sweep_calibration([(f, t) for f, t, _ in raw], table, facts)
        return

    if args.sweep_alpha:
        raw = collect_samples(args.data_dir, args.limit, args.seed, args.label_space)
        samples = [(f, t) for f, t, _ in raw]
        every = all_facts(table)
        print(f"{len(samples)} labelled faces from {args.limit} models\n")
        print(f"{'alpha':>6}  {'hand-picked 9':>22}  {'all facts (default)':>22}")
        print(f"{'':>6}  {'macro    acc':>22}  {'macro    acc':>22}")
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            md, ad, _ = _eval_facts(samples, table, HAND_PICKED_FACTS, args.temperature, a,
                                    full=True)
            ma, aa, _ = _eval_facts(samples, table, every, args.temperature, a,
                                    full=True)
            print(f"{a:>6.2f}  {md:>7.4f} {ad:>7.4f}          "
                  f"{ma:>7.4f} {aa:>7.4f}")
        return

    if args.select_facts:
        raw = collect_samples(args.data_dir, args.limit, args.seed, args.label_space)
        train, held = split_by_model(raw)
        print(f"{len(raw)} labelled faces from {args.limit} models -> "
              f"select on {len(train)}, score on {len(held)} held-out "
              f"(alpha={args.alpha})\n")
        chosen = select_facts(train, table, args.temperature, args.alpha)
        print("\nselected:")
        for f in chosen:
            print(f"    {f!r},")
        print(f"\n{'fact set':>16}  {'selection half':>16}  {'HELD-OUT half':>16}")
        for tag, fs in (("selected now", chosen),
                        ("greedy (recorded)", GREEDY_FACTS_120),
                        ("hand-picked 9", HAND_PICKED_FACTS),
                        ("all facts", all_facts(table))):
            mt = _eval_facts(train, table, fs, args.temperature, args.alpha)
            mh, ah, _ = _eval_facts(held, table, fs, args.temperature, args.alpha,
                                    full=True)
            print(f"{tag:>16}  {mt:>16.4f}  {mh:>10.4f} (acc {ah:.3f})")
        print("\nmacro-recall. A selected set that wins on the selection half and loses on "
              "the held-out half was fitted to the sample, not to the geometry.")
        return

    if args.step:
        res = label_step(args.step, table, facts, args.temperature, args.alpha,
                         args.gate_alpha)
        text = json.dumps(res, indent=2)
        if args.out:
            args.out.write_text(text)
            print(f"wrote {args.out}")
        else:
            print(text[:2000])
        return

    if not (args.step_dir and args.out_dir):
        raise SystemExit("need --step or (--step_dir and --out_dir), or --select_facts")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    steps = sorted(p for p in args.step_dir.iterdir()
                   if p.suffix.lower() in {".step", ".stp"})
    ok = bad = 0
    for i, sp in enumerate(steps, 1):
        try:
            res = label_step(sp, table, facts, args.temperature, args.alpha,
                             args.gate_alpha)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(steps)}] {sp.stem}: {type(e).__name__}: {e}")
            bad += 1
            continue
        (args.out_dir / f"{sp.stem}.json").write_text(json.dumps(res))
        ok += 1
        if i % 20 == 0:
            print(f"  {i}/{len(steps)}", flush=True)
    print(f"wrote {ok} predictions to {args.out_dir} ({bad} failed)")


if __name__ == "__main__":
    main()
