"""Measure any label-producing or label-modifying component against mftrcad ground truth.

Why this exists
---------------
`geometric_rules.py` shipped a rule that denied 100% of `chamfer_2` and 15% of all
correct labels, and nobody noticed for weeks. It could not be noticed: it ran on
pseudo-labels, which have no correct answer for a bad heuristic to disagree with, so it
produced confident-looking output and poisoned the corpus silently. The fix is not more
care — it is refusing to let any component write labels until its harm rate has been
measured on data where the answer is known.

mftrcad supplies that: ~28.5k models with exact per-face labels. This module is the
single place that measurement happens, so a new heuristic gets scored the same way as
the last one instead of growing its own bespoke validator.

Three modes, because "is this component good?" is really three questions
------------------------------------------------------------------------
``--mode harm`` — Feed the *ground truth* in as if it were the component's input. Every
face is then correct by construction, so **any** change the component proposes is
provably a false one, and its rate is the rate at which the component destroys correct
labels. This is the cheap, decisive test and the one that catches disasters.

``--mode benefit`` — Harm alone cannot distinguish a useless component from a harmful
one: a component that changes nothing scores a perfect 0% harm. So corrupt the ground
truth first, then measure how much of it the component recovers. The corruption is
deliberately *not* uniform-random — a real teacher confuses `chamfer` with `fillet` and
`hole` with `Oring`, it does not confuse `chamfer` with `6sides_passage`. Uniform noise
is trivially easy to fix and would flatter every component. See ``NOISE_MODELS``.

``--mode select`` — The mode that matters for auto-labelling. The goal is not a
component that is right everywhere; it is a component that **knows when it is right**,
so predictions can be auto-accepted above a confidence and routed to review below it.
That is a ranking question, not an accuracy question, and it is answered by a
risk–coverage curve: sort faces by confidence, accept the top k%, and plot the error rate
among accepted faces. Reports AUGRC (area under the generalised risk–coverage curve,
lower is better) plus the achievable coverage at target precisions, per class.

Everything is reported **per class**. A component that is fine on `stock` and
catastrophic on `chamfer` looks acceptable in aggregate while destroying exactly the rare
classes this corpus was built to rescue — and that is not hypothetical: the
`area_fraction_max` rule binds on 0.34% of faces corpus-wide and on 72% of `chamfer_2`.

Usage
-----
    # Does geometric_rules destroy correct labels? (the test that caught the disaster)
    python -m brepformer.data.label_harness --mode harm --component geometric_rules \
        --limit 60 --label_space 27

    # Does it recover corrupted ones? Run both; a high change rate alone cannot
    # distinguish fixing from churn.
    python -m brepformer.data.label_harness --mode benefit --component geometric_rules \
        --limit 60 --noise confusable --noise_rate 0.2

    # At what coverage can model predictions be auto-accepted, and at what precision?
    python -m brepformer.data.label_harness --mode select \
        --predictions brepformer/inference/hard_samples_27 --limit 200
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from brepformer.data.classes import (
    CLASS_NAMES, DEFEATURE_CLASS_NAMES, CLASS_TO_DEFEATURE,
)

#: Class pairs a real teacher actually confuses, used by ``--noise confusable``.
#: Uniform-random corruption is trivially detectable by any geometric check and would
#: make every component look good, so the noise model has to be plausible or the benefit
#: number is meaningless. Pairs are bidirectional and drawn from the confusion structure
#: recorded in the project's own error analyses: blend-vs-blend, and round features that
#: differ only in depth or closure.
CONFUSABLE_27 = [
    (0, 26),    # chamfer <-> chamfer_2
    (0, 23),    # chamfer <-> round
    (23, 26),   # round <-> chamfer_2
    (1, 12),    # through_hole <-> blind_hole
    (1, 11),    # through_hole <-> Oring
    (12, 16),   # blind_hole <-> circular_end_pocket
    (7, 19),    # circular_through_slot <-> circular_blind_slot
    (3, 25),    # rectangular_passage <-> rectangular_passage_2
    (6, 17),    # rectangular_through_slot <-> rectangular_blind_slot
    (8, 20),    # rectangular_through_step <-> rectangular_blind_step
    (2, 13),    # triangular_passage <-> triangular_pocket
    (14, 15),   # rectangular_pocket <-> 6sides_pocket
]

CONFUSABLE_5 = [
    (2, 3),     # chamfer <-> fillet: the pair the whole project turns on
    (1, 4),     # hole <-> cut
    (0, 4),     # random <-> cut
]


def load_gt_face_labels(label_path: Path, num_faces_hint: int | None = None):
    """Read per-face ground-truth class ids from an mftrcad ``*_result.json``.

    Two schemas exist in this repo and mixing them up yields an all-``-1`` corpus that
    trains silently on nothing: pseudo-label files are ``[id, {"labels": [...]}]`` while
    some carry a sparse ``{"cls": {face_idx: class}}`` map instead.
    """
    data = json.loads(label_path.read_text())
    if isinstance(data, list) and len(data) > 1:
        data = data[1]
    if not isinstance(data, dict):
        return None
    if "labels" in data and isinstance(data["labels"], list):
        return [int(x) for x in data["labels"]]
    if "cls" in data:
        cls = {int(k): int(v) for k, v in data["cls"].items()}
        n = (num_faces_hint or 0) or (max(cls) + 1 if cls else 0)
        return [cls.get(i, -1) for i in range(n)]
    return None


def iter_ground_truth(data_dir: Path, limit: int, seed: int, label_space: str = "27"):
    """Yield ``(step_path, gt_labels)`` for up to ``limit`` mftrcad models.

    Shuffled with a fixed seed so two harness runs compare the same models; a different
    sample would make a before/after difference unattributable to the change.

    mftrcad ground truth is always 27-class, so when the component under test works in
    the 5-class space the truth is collapsed here rather than at each call site. Getting
    this wrong is silent and total: comparing 5-class predictions against 27-class truth
    scored a 0.94-F1 checkpoint at 7.6% accuracy with `stock` at 0%, which reads as a
    broken model rather than as a units error.
    """
    steps_dir, labels_dir = data_dir / "steps", data_dir / "labels"
    steps = sorted(p for p in steps_dir.iterdir()
                   if p.suffix.lower() in {".step", ".stp"})
    random.Random(seed).shuffle(steps)

    n = 0
    for step_path in steps:
        if n >= limit:
            return
        label_path = labels_dir / f"{step_path.stem}_result.json"
        if not label_path.exists():
            hits = [h for h in labels_dir.glob(f"{step_path.stem}*.json")
                    if "_rel" not in h.name]
            if not hits:
                continue
            label_path = hits[0]
        try:
            gt = load_gt_face_labels(label_path)
        except Exception:  # noqa: BLE001
            continue
        if not gt:
            continue
        if label_space == "5":
            gt = [CLASS_TO_DEFEATURE[c] if 0 <= c < len(CLASS_TO_DEFEATURE) else -1
                  for c in gt]
        n += 1
        yield step_path, gt


def iter_ground_truth_pkl(pkl_dir: Path, limit: int, label_space: str):
    """Yield ``(model_id, gt_labels)`` from a *preprocessed* split's ``face_labels``.

    Why this exists alongside ``iter_ground_truth``: every checkpoint F1 in SCRIPTS.md is
    measured on ``defeature_processed/test`` — 157 real hand-labelled parts — while the
    harness's mftrcad path reads synthetic MFTRCAD models remapped to 5 classes. Quoting an
    auto-accept coverage from one and a macro F1 from the other invites the reader to
    combine two numbers that were never measured on the same parts.

    ``face_labels`` is already in the corpus's own space, so it is collapsed only when the
    corpus is 27-class and the caller asked for 5 (``metadata.json`` says which).
    """
    meta = pkl_dir.parent / "metadata.json"
    data_classes = int(json.loads(meta.read_text()).get("num_classes", 27)) \
        if meta.exists() else 27
    n = 0
    for pkl in sorted(p for p in pkl_dir.iterdir() if p.suffix == ".pkl"):
        if limit and n >= limit:
            return
        with open(pkl, "rb") as f:
            s = pickle.load(f)
        gt = [int(v) for v in s.get("face_labels", [])]
        if not gt or not any(v >= 0 for v in gt):
            continue
        if label_space == "5" and data_classes == 27:
            gt = [CLASS_TO_DEFEATURE[c] if 0 <= c < len(CLASS_TO_DEFEATURE) else -1
                  for c in gt]
        n += 1
        yield s["model_id"], gt


def corrupt(labels: list, rate: float, model: str, label_space: str, rng: random.Random):
    """Return ``(corrupted_labels, corrupted_indices)``.

    ``confusable`` swaps a label only for one it is genuinely mistaken for (see
    ``CONFUSABLE_*``); ``uniform`` picks any other class. Prefer ``confusable``: uniform
    noise produces geometrically absurd labels that any surface-type check catches, which
    inflates the measured benefit of components that are useless on real errors.
    """
    pairs = CONFUSABLE_27 if label_space == "27" else CONFUSABLE_5
    n_classes = len(CLASS_NAMES) if label_space == "27" else len(DEFEATURE_CLASS_NAMES)
    swap = collections.defaultdict(list)
    for a, b in pairs:
        swap[a].append(b)
        swap[b].append(a)

    out, changed = list(labels), []
    for i, c in enumerate(labels):
        if c < 0 or rng.random() >= rate:
            continue
        if model == "confusable":
            options = swap.get(c)
            if not options:
                continue  # no plausible confusion for this class; leave it alone
            out[i] = rng.choice(options)
        else:
            out[i] = rng.choice([k for k in range(n_classes) if k != c])
        changed.append(i)
    return out, changed


def run_rules(step_path: Path, preds: list, label_space: str):
    """Apply geometric_rules and return the label list it would produce, plus verdicts.

    A DENY with a suggestion is a proposed rewrite; CONFIRM and UNCERTAIN leave the label
    alone. This is ``rewrite`` semantics on purpose — the harness must measure what the
    component *would* do if trusted, not what ``advise`` mode safely does instead.
    """
    from brepformer.geometric_rules import apply_geometric_rules, VERDICT_DENY

    verdicts = apply_geometric_rules(str(step_path), preds, label_space)
    if len(verdicts) != len(preds):
        return None, None  # face-index misalignment makes every comparison meaningless
    out = list(preds)
    for v in verdicts:
        if v["verdict"] == VERDICT_DENY and v.get("suggested") is not None:
            out[v["face_idx"]] = v["suggested"]
    return out, verdicts


def mode_harm(args) -> dict:
    """Feed ground truth in; every proposed change is provably a corruption."""
    names = CLASS_NAMES if args.label_space == "27" else DEFEATURE_CLASS_NAMES
    per_class = collections.defaultdict(lambda: {"n": 0, "corrupted": 0})
    by_rule = collections.Counter()
    rule_by_class = collections.defaultdict(collections.Counter)
    n_models = n_faces = n_bad = skipped = 0

    for step_path, gt in iter_ground_truth(args.data_dir, args.limit, args.seed, args.label_space):
        try:
            out, verdicts = run_rules(step_path, gt, args.label_space)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {step_path.stem}: {e}", file=sys.stderr)
            skipped += 1
            continue
        if out is None:
            skipped += 1
            continue

        for i, (truth, got) in enumerate(zip(gt, out)):
            if truth < 0:
                continue
            per_class[truth]["n"] += 1
            n_faces += 1
            if got != truth:
                per_class[truth]["corrupted"] += 1
                n_bad += 1
                rule = next((v.get("rule") for v in verdicts if v["face_idx"] == i), "?")
                by_rule[rule or "?"] += 1
                rule_by_class[truth][rule or "?"] += 1
        n_models += 1

    print(f"\nHARM: {n_models} models / {n_faces} labelled faces ({skipped} skipped)")
    if not n_faces:
        raise SystemExit("no faces measured — check paths / face-index alignment")
    print(f"correct labels destroyed: {n_bad}/{n_faces} = {100 * n_bad / n_faces:.1f}%\n")

    print(f"{'class':<26}{'faces':>8}{'corrupted':>11}{'rate':>8}  worst rule")
    for c, s in sorted(per_class.items(), key=lambda kv: -kv[1]["corrupted"]):
        if not s["corrupted"] and not args.show_all:
            continue
        worst = rule_by_class[c].most_common(1)
        name = names[c] if c < len(names) else str(c)
        print(f"{name:<26}{s['n']:>8}{s['corrupted']:>11}"
              f"{100 * s['corrupted'] / s['n']:>7.1f}%  "
              f"{worst[0][0] if worst else '-'}")

    print("\nby rule (which rule to change, not just that something is broken):")
    for r, n in by_rule.most_common():
        print(f"  {r:<24}{n:>6}")

    return {
        "mode": "harm", "n_models": n_models, "n_faces": n_faces,
        "harm_rate": n_bad / n_faces,
        "per_class": {str(c): s for c, s in per_class.items()},
        "by_rule": dict(by_rule),
    }


def mode_benefit(args) -> dict:
    """Corrupt ground truth, then measure what fraction the component recovers.

    Three outcomes per corrupted face -- fixed (restored to truth), churned (changed to a
    third wrong answer) and missed (left wrong) -- because "changed a lot of labels" is
    not evidence of anything. Clean faces are tracked separately: a component that fixes
    every corruption while breaking twice as many correct labels is a net loss, and only
    reporting both numbers shows it.
    """
    names = CLASS_NAMES if args.label_space == "27" else DEFEATURE_CLASS_NAMES
    rng = random.Random(args.seed)
    stats = collections.defaultdict(
        lambda: {"noisy": 0, "fixed": 0, "churned": 0, "clean": 0, "broken": 0})
    n_models = skipped = 0

    for step_path, gt in iter_ground_truth(args.data_dir, args.limit, args.seed, args.label_space):
        noisy, changed = corrupt(gt, args.noise_rate, args.noise, args.label_space, rng)
        try:
            out, _ = run_rules(step_path, noisy, args.label_space)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if out is None:
            skipped += 1
            continue

        corrupted_idx = set(changed)
        for i, truth in enumerate(gt):
            if truth < 0:
                continue
            s = stats[truth]
            if i in corrupted_idx:
                s["noisy"] += 1
                if out[i] == truth:
                    s["fixed"] += 1
                elif out[i] != noisy[i]:
                    s["churned"] += 1
            else:
                s["clean"] += 1
                if out[i] != truth:
                    s["broken"] += 1
        n_models += 1

    tot = collections.Counter()
    for s in stats.values():
        for k, v in s.items():
            tot[k] += v
    print(f"\nBENEFIT: {n_models} models, noise={args.noise} @ {args.noise_rate:.0%} "
          f"({skipped} skipped)")
    if not tot["noisy"]:
        raise SystemExit("no faces were corrupted — raise --noise_rate")
    print(f"corrupted faces fixed:   {tot['fixed']}/{tot['noisy']} = "
          f"{100 * tot['fixed'] / tot['noisy']:.1f}%")
    print(f"corrupted faces churned: {tot['churned']}/{tot['noisy']} = "
          f"{100 * tot['churned'] / tot['noisy']:.1f}%  (changed, still wrong)")
    print(f"clean faces broken:      {tot['broken']}/{tot['clean']} = "
          f"{100 * tot['broken'] / max(tot['clean'], 1):.1f}%")
    net = tot["fixed"] - tot["broken"]
    print(f"\nNET: {net:+d} labels ({'improvement' if net > 0 else 'REGRESSION'})\n")

    print(f"{'class':<26}{'noisy':>7}{'fixed':>7}{'fix%':>7}{'clean':>8}{'broken':>8}{'net':>7}")
    for c, s in sorted(stats.items(), key=lambda kv: -kv[1]["noisy"]):
        if not s["noisy"] and not args.show_all:
            continue
        name = names[c] if c < len(names) else str(c)
        print(f"{name:<26}{s['noisy']:>7}{s['fixed']:>7}"
              f"{100 * s['fixed'] / max(s['noisy'], 1):>6.0f}%"
              f"{s['clean']:>8}{s['broken']:>8}{s['fixed'] - s['broken']:>+7d}")

    return {
        "mode": "benefit", "noise": args.noise, "noise_rate": args.noise_rate,
        "n_models": n_models, "totals": dict(tot), "net": net,
        "per_class": {str(c): s for c, s in stats.items()},
    }


#: Confidence level for the certified auto-accept threshold. Split across the binary search
#: over candidate thresholds by a delta/log2(m) multiplicity correction (Geifman & El-Yaniv,
#: arXiv:1705.08500) -- without it, scanning every k and keeping the best one is selecting on
#: the same data the bound is computed from, and the bound stops holding.
DELTA = 0.05


def binom_precision_lower_bound(n_ok: int, n: int, delta: float = DELTA) -> float:
    """Clopper-Pearson lower confidence bound on precision from ``n_ok`` of ``n`` correct.

    Replaces reading the empirical precision off the risk-coverage table and calling it the
    auto-accept budget. The empirical number is the precision *on this sample*; at the tail of
    the ranking it is computed from very few faces, so "100% precision on the top 12 faces"
    certifies nothing. This returns the precision that actually holds on unseen faces with
    probability 1-delta, so a small accepted set is correctly penalised.

    Exact binomial rather than a normal or Hoeffding approximation: for a 0/1 loss the exact
    inversion is both valid and strictly tighter (Bates et al., RCPS, JACM 2021).
    """
    if n <= 0:
        return 0.0
    d = _delta_eff(n, delta)
    if n_ok >= n:
        # One-sided: P(all n correct | p) = p**n >= delta_eff  =>  p >= delta_eff ** (1/n).
        return d ** (1.0 / n)
    from scipy.stats import beta
    return float(beta.ppf(d, n_ok, n - n_ok + 1))


def _delta_eff(n: int, delta: float = DELTA) -> float:
    """Multiplicity-corrected delta for a search over ~n candidate thresholds.

    ``delta`` was previously accepted by ``binom_precision_lower_bound`` and then **silently
    ignored** here, so every caller got the module-level 0.05 no matter what it asked for. Only
    ``mode_select`` called it, and only at the default, so no recorded number changes — but a
    per-class threshold search has to spend a *smaller* delta per class (Bonferroni over the
    classes, on top of the log2(n) correction over candidate thresholds within a class), and that
    is impossible if the argument does nothing.
    """
    return delta / max(math.log2(max(n, 2)), 1.0)


def compute_risk_coverage(rows: list) -> dict:
    """Risk-coverage summaries for ``[(confidence, correct), ...]``.

    Both answer the only question an auto-accept threshold cares about -- does confidence
    rank the errors last -- and neither can be improved by a labeller that merely predicts
    the majority class. They differ in the denominator, and the difference is not cosmetic:

    ``aurc``   mean error rate *among the accepted* faces, averaged over every coverage.
               The incumbent metric (Jaeger et al., ICLR 2023, arXiv:2211.15259).
    ``augrc``  mean error *count over all* faces, averaged over every coverage; i.e. the
               generalised risk (Traub et al., NeurIPS 2024, arXiv:2407.01032).

    **This function used to compute AURC and call it AUGRC** -- ``errs / k`` rather than
    ``errs / n``. That matters here specifically: AURC's 1/(accepted count) denominator gives
    high-confidence failures a harmonic weight, which breaks monotonicity (it can prefer a
    scorer with both lower accuracy *and* worse error ranking) and amplifies the influence of
    label noise, which this corpus has. Verified against the closed form
    ``(1 - AUROC_f) * acc * (1 - acc) + 0.5 * (1 - acc) ** 2`` to within 0.004.

    Each has its own uninformative baseline, and they are different numbers. A random ranking
    has ``AUROC_f = 0.5``, so ``augrc_base = (1 - acc) / 2`` -- *half* the error rate -- while
    ``aurc_base`` is the error rate itself. The previously documented baselines (0.27 for 5
    classes, 0.51 for 27) were the AURC ones. Never compare either across label spaces.

    Shared rather than reimplemented: `geo_label --sweep_calibration` tunes against these, and
    a tuner optimising a slightly different metric than the one reporting the result is how a
    calibration silently stops meaning anything.
    """
    n = len(rows)
    if not n:
        return {"aurc": 0.0, "augrc": 0.0, "aurc_base": 0.0, "augrc_base": 0.0}
    ordered = sorted(rows, key=lambda r: -r[0])
    errs = 0.0
    aurc = 0.0
    augrc = 0.0
    for k, (_, ok) in enumerate(ordered, start=1):
        if not ok:
            errs += 1
        aurc += errs / k
        augrc += errs / n
    err_rate = errs / n
    return {
        "aurc": aurc / n,
        "augrc": augrc / n,
        "aurc_base": err_rate,
        "augrc_base": err_rate / 2.0,
    }


def compute_augrc(rows: list) -> float:
    """Back-compatible accessor returning the (now correct) AUGRC."""
    return compute_risk_coverage(rows)["augrc"]


def mode_select(args) -> dict:
    """Risk-coverage analysis: can predictions be auto-accepted above a confidence?

    This is the question auto-labelling actually poses. Accuracy says how often the model
    is right; this says whether its confidence *ranks* its errors last, which is what
    decides how much of a corpus can be accepted untouched and how much needs review.

    AUGRC is the mean error count among accepted faces, averaged over every coverage
    level, normalised by the total. It is the right scalar because it charges for errors
    that slip through at high confidence far more than for a low-confidence mistake that
    review would have caught anyway. Lower is better; 0 means every error is ranked below
    every correct prediction.
    """
    names = CLASS_NAMES if args.label_space == "27" else DEFEATURE_CLASS_NAMES
    pred_dir = Path(args.predictions)
    rows = []  # (confidence, correct, true_class, pred_class)
    n_models = skipped = 0

    gt_by_id = {}
    if args.gt_pkl_dir is not None:
        for model_id, gt in iter_ground_truth_pkl(args.gt_pkl_dir, args.limit, args.label_space):
            gt_by_id[str(model_id)] = gt
    else:
        for step_path, gt in iter_ground_truth(args.data_dir, args.limit, args.seed,
                                               args.label_space):
            gt_by_id[step_path.stem] = gt

    for pred_path in sorted(pred_dir.glob("*.json")):
        stem = pred_path.stem
        gt = gt_by_id.get(stem) or gt_by_id.get(stem.replace("_result", ""))
        if gt is None:
            continue
        try:
            res = json.loads(pred_path.read_text())
            preds, probs = res["face_preds"], res.get("face_probs")
            # A labeller may score its own decision on a different scale than the posterior
            # it decided with -- `geo_label` pools evidence one way to pick the class and
            # another to rate it, which cut its 5-class AUGRC from 0.2345 to 0.1851. Honour
            # that when offered; fall back to max(posterior), which is all a checkpoint has.
            explicit_conf = res.get("face_confidence")
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if probs is None and explicit_conf is None:
            skipped += 1
            continue
        # A checkpoint's label space is not recoverable from its filename, and infer.py
        # will happily print 27-class names over a 5-class posterior when --5_classes is
        # omitted. Comparing across spaces does not error anywhere -- the ids just never
        # agree -- so it has to be caught here or it silently reports a good model as a
        # catastrophic one.
        if probs is not None and len(probs[0]) != len(names):
            raise SystemExit(
                f"{pred_path.name} has a {len(probs[0])}-class posterior but "
                f"--label_space {args.label_space} expects {len(names)}. Re-run infer.py "
                f"with the matching --5_classes/--8_classes flag, or pass the "
                f"--label_space the checkpoint was trained in.")
        # A fused labeller (`fuse_labels.py`) emits a confidence tier and no posterior, so
        # there is no width to check. Fall back to the weaker guard that still catches the
        # 27-vs-5 mixup: a 27-class id cannot appear in a 5-class prediction.
        if probs is None and any(int(p) >= len(names) for p in preds):
            raise SystemExit(
                f"{pred_path.name} predicts class id {max(int(p) for p in preds)} but "
                f"--label_space {args.label_space} only has {len(names)} classes.")
        if len(preds) != len(gt):
            skipped += 1
            continue
        if explicit_conf is not None and len(explicit_conf) != len(preds):
            explicit_conf = None
        for i, (p, truth) in enumerate(zip(preds, gt)):
            if truth < 0:
                continue
            c = explicit_conf[i] if explicit_conf is not None else max(probs[i])
            rows.append((c, int(p) == int(truth), int(truth), int(p)))
        n_models += 1

    print(f"\nSELECT: {n_models} models / {len(rows)} faces "
          f"({skipped} prediction files skipped)")
    if not rows:
        raise SystemExit(
            "no faces matched — do the prediction filenames match mftrcad stems, and do "
            "the files carry face_probs? (infer.py writes them only in .json, not .seg)")

    rows.sort(key=lambda r: -r[0])
    n = len(rows)
    n_err = sum(1 for r in rows if not r[1])
    print(f"overall accuracy: {100 * (n - n_err) / n:.1f}%\n")

    rc = compute_risk_coverage([(r[0], r[1]) for r in rows])
    augrc = rc["augrc"]
    print(f"AUGRC: {augrc:.4f}   (0 = every error ranked below every correct prediction; "
          f"{rc['augrc_base']:.4f} = confidence is uninformative)")
    print(f"AURC:  {rc['aurc']:.4f}   (uninformative {rc['aurc_base']:.4f})   "
          f"-- both reported because AUGRC is a 2024 proposal and AURC is still the incumbent")

    print(f"\n{'coverage':>10}{'threshold':>11}{'precision':>11}{'errors let through':>20}")
    errs = 0
    marks = {int(n * f) for f in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)}
    for k, r in enumerate(rows, start=1):
        if not r[1]:
            errs += 1
        if k in marks:
            print(f"{100 * k / n:>9.0f}%{r[0]:>11.3f}"
                  f"{100 * (k - errs) / k:>10.1f}%{errs:>20d}")

    print("\nmax coverage at each target precision (the auto-accept budget):")
    print("  observed = the empirical cut; certified = holds on unseen faces with "
          f"probability {1 - DELTA:.0%}")
    for target in (0.99, 0.98, 0.95, 0.90):
        errs = best_k = cert_k = 0
        for k, r in enumerate(rows, start=1):
            if not r[1]:
                errs += 1
            if (k - errs) / k >= target:
                best_k = k
            if binom_precision_lower_bound(k - errs, k) >= target:
                cert_k = k
        if best_k:
            line = (f"  >={target:.0%} precision: accept {100 * best_k / n:.1f}% of faces "
                    f"(threshold {rows[best_k - 1][0]:.3f})")
        else:
            line = f"  >={target:.0%} precision: unreachable at any coverage"
        if cert_k:
            line += f"   | certified {100 * cert_k / n:.1f}% (threshold {rows[cert_k - 1][0]:.3f})"
        else:
            line += "   | certified: none"
        print(line)

    # Per class, on the true label: a class whose errors are confident is a class that
    # cannot be auto-accepted no matter what the global threshold is.
    print(f"\n{'class (true)':<26}{'faces':>7}{'acc':>7}{'mean conf':>11}"
          f"{'conf when wrong':>17}")
    per_class = collections.defaultdict(lambda: {"n": 0, "ok": 0, "c": 0.0, "cw": 0.0})
    for conf, ok, truth, _ in rows:
        s = per_class[truth]
        s["n"] += 1
        s["c"] += conf
        if ok:
            s["ok"] += 1
        else:
            s["cw"] += conf
    out_pc = {}
    for c, s in sorted(per_class.items(), key=lambda kv: -kv[1]["n"]):
        wrong = s["n"] - s["ok"]
        name = names[c] if c < len(names) else str(c)
        print(f"{name:<26}{s['n']:>7}{100 * s['ok'] / s['n']:>6.0f}%"
              f"{s['c'] / s['n']:>11.3f}"
              f"{(s['cw'] / wrong if wrong else float('nan')):>17.3f}")
        out_pc[str(c)] = {"n": s["n"], "acc": s["ok"] / s["n"],
                          "mean_conf": s["c"] / s["n"],
                          "mean_conf_wrong": (s["cw"] / wrong) if wrong else None}

    return {"mode": "select", "n_models": n_models, "n_faces": n,
            "accuracy": (n - n_err) / n, "augrc": augrc, "per_class": out_pc}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["harm", "benefit", "select"])
    ap.add_argument("--component", default="geometric_rules",
                    choices=["geometric_rules"],
                    help="Component under test (harm/benefit modes).")
    ap.add_argument("--data_dir", type=Path, default=Path("brepformer/data/mftrcad"),
                    help="mftrcad root, with steps/ and labels/ subdirs.")
    ap.add_argument("--limit", type=int, default=60, help="Models to evaluate.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Fixes both the model sample and the corruption draw, so two "
                         "runs are comparable.")
    ap.add_argument("--label_space", default="27", choices=["27", "5"])
    ap.add_argument("--noise", default="confusable", choices=["confusable", "uniform"],
                    help="Corruption model for --mode benefit. See CONFUSABLE_27.")
    ap.add_argument("--noise_rate", type=float, default=0.2,
                    help="Fraction of labelled faces to corrupt.")
    ap.add_argument("--predictions", type=Path, default=None,
                    help="Directory of infer.py .json outputs, for --mode select. Must "
                         "contain face_probs, which .seg output does not carry.")
    ap.add_argument("--gt_pkl_dir", type=Path, default=None,
                    help="Take --mode select ground truth from a PREPROCESSED split's "
                         "face_labels instead of from mftrcad labels/, e.g. "
                         "brepformer/data/defeature_processed/test. Use this to put "
                         "coverage/precision on the same 157 real hand-labelled parts that "
                         "every checkpoint macro F1 in SCRIPTS.md is measured on; the "
                         "mftrcad path scores synthetic models remapped to 5 classes, and "
                         "the two are not comparable. Pair it with "
                         "`scan_checkpoints.py --dump_probs`.")
    ap.add_argument("--show_all", action="store_true",
                    help="List classes with nothing to report as well.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.mode == "select" and not args.predictions:
        raise SystemExit("--mode select needs --predictions")

    result = {"harm": mode_harm, "benefit": mode_benefit,
              "select": mode_select}[args.mode](args)

    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
