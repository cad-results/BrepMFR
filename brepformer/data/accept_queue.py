#!/usr/bin/env python3
"""Per-class certified auto-accept thresholds, and the review queue that falls out of them.

This is the operational half of the risk-coverage analysis in ``label_harness --mode select``.
That reports *whether* confidence ranks errors last; this decides, per face, **accept or review**,
and it is the tool the labelling workflow runs on: a face above its class's threshold is taken as
correct without being looked at, and everything else goes into a queue ordered so the most
consequential faces are seen first. Nothing is masked — a reviewed face gets a decision, and an
unreviewed face gets a label with a certified precision attached to it.

**Why per class, given that it accepts FEWER faces.** Measured on the 157 real hand-labelled parts,
per-class thresholds certify 72.9% of faces at 95% precision where a single global threshold
certifies 80.0% — the 7.1 points go to the Bonferroni split across classes and to the weaker bound
each class gets from its own smaller sample. Raw coverage is the wrong comparison, because the
global number is a **pooled** guarantee: at its threshold of 0.917, ``cut`` and ``hole`` (55% of
faces) hold at 95%+ and carry the average, while ``random`` (94.2% certified), ``fillet`` (91.7%)
and ``chamfer`` (83.9%) all sit *under* target inside the accepted set. So the global cut
auto-accepts roughly one wrong chamfer in nine — in the class where a wrong auto-accepted label is
most damaging, since chamfer/fillet confusion is this model's characteristic failure. Per-class
trades those 7.1 points for a guarantee that holds per class, which is the only kind a review
workflow can act on. ``calibrate`` prints both tables so the trade is visible, not assumed.

For the incumbent, no threshold certifies 95% for ``chamfer`` at any confidence, so the rule is
that **every predicted-chamfer face is reviewed** — 474 faces, 3.5% of the corpus.

**Why the threshold is certified rather than read off the table.** Scanning candidate thresholds and
keeping the best one is selecting on the same data the precision is computed from, so the empirical
number does not hold on unseen faces — and at the tail of the ranking it is computed from very few
faces, where "100% precision on the top 12" certifies nothing. The bound is Clopper-Pearson,
inverted exactly (Bates et al., RCPS, JACM 2021), with delta split two ways: by ``log2(n)`` over the
candidate thresholds inside a class, and by Bonferroni over the classes. Splitting only the first
way is the usual mistake and it makes a per-class table look better than a global one purely
because it searched more.

The thresholds are keyed on the **predicted** class, not the true one, because that is the only
thing available at apply time. The per-class accuracy tables elsewhere in this repo are keyed on the
true class; the two are different numbers and the predicted-class one is the one a threshold can act
on.

Usage:
    # 1. Dump per-face posteriors for the deployed checkpoint.
    python -m brepformer.scan_checkpoints --ckpt_dirs results/trial1_finetune_ss \
        --data_dir brepformer/data/defeature_processed --split test --label_space 5 \
        --dump_probs /tmp/probs_incumbent

    # 2. Calibrate against ground truth, and see what per-class buys over global.
    python -m brepformer.data.accept_queue calibrate --probs_dir /tmp/probs_incumbent \
        --gt_pkl_dir brepformer/data/defeature_processed/test --label_space 5 \
        --target_precision 0.95 --out results/accept_thresholds_5class.json

    # 3. Apply to an unlabelled corpus: writes accept/review decisions and a queue.
    python -m brepformer.data.accept_queue apply --probs_dir /tmp/probs_newparts \
        --thresholds results/accept_thresholds_5class.json --out_dir /tmp/queue
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from brepformer.data.classes import CLASS_NAMES, DEFEATURE_CLASS_NAMES
from brepformer.data.label_harness import (
    DELTA, binom_precision_lower_bound, iter_ground_truth_pkl,
)


def load_rows(probs_dir: Path, gt_by_id: dict | None):
    """Yield ``(model_id, face_index, pred, confidence, truth_or_None)`` per supervised face.

    ``confidence`` is ``max(posterior)``. A labeller that scores its decision on a different scale
    than the posterior it decided with may supply ``face_confidence``; that is honoured, because
    `geo_label` does exactly this and it cut its AUGRC from 0.2345 to 0.1851.
    """
    n_files = skipped = 0
    for path in sorted(probs_dir.glob("*.json")):
        try:
            res = json.loads(path.read_text())
            preds = res["face_preds"]
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        probs = res.get("face_probs")
        conf = res.get("face_confidence")
        if probs is None and conf is None:
            skipped += 1
            continue
        model_id = str(res.get("model_id", path.stem))
        gt = None
        if gt_by_id is not None:
            gt = gt_by_id.get(model_id) or gt_by_id.get(path.stem) \
                or gt_by_id.get(path.stem.replace("_result", ""))
            if gt is None or len(gt) != len(preds):
                skipped += 1
                continue
        n_files += 1
        for i, p in enumerate(preds):
            c = conf[i] if conf is not None else max(probs[i])
            truth = None if gt is None else int(gt[i])
            if truth is not None and truth < 0:
                continue  # unsupervised face: cannot calibrate on it
            yield model_id, i, int(p), float(c), truth
    if not n_files:
        raise SystemExit(
            f"no usable prediction files in {probs_dir} ({skipped} skipped). Do they carry "
            f"face_probs, and do their model_ids match the ground-truth split?")
    print(f"read {n_files} prediction files ({skipped} skipped)")


def certify(rows: list, target: float, delta: float) -> tuple[float | None, dict]:
    """Highest-coverage confidence cut whose *certified* precision still clears ``target``.

    ``rows`` is ``[(confidence, correct), ...]``. Returns ``(threshold, stats)``, or
    ``(None, stats)`` if no cut certifies — which is a real answer and must not be rounded down to
    "accept everything above 0.9".
    """
    if not rows:
        return None, {"n": 0, "accepted": 0}
    ordered = sorted(rows, key=lambda r: -r[0])
    n = len(ordered)
    errs = 0
    best = None
    for k, (c, ok) in enumerate(ordered, start=1):
        if not ok:
            errs += 1
        # Ties matter: accepting at >= c admits every face with that confidence, so a threshold
        # is only valid at a boundary where the next face is strictly less confident.
        if k < n and ordered[k][0] == c:
            continue
        if binom_precision_lower_bound(k - errs, k, delta) >= target:
            best = (c, k, k - errs)
    if best is None:
        return None, {"n": n, "accepted": 0, "coverage": 0.0,
                      "observed_precision": None, "certified_precision": None}
    c, k, ok = best
    return c, {"n": n, "accepted": k, "coverage": k / n,
               "observed_precision": ok / k,
               "certified_precision": binom_precision_lower_bound(ok, k, delta)}


def mode_calibrate(args) -> dict:
    names = CLASS_NAMES if args.label_space == "27" else DEFEATURE_CLASS_NAMES
    gt_by_id = {str(mid): gt for mid, gt in
                iter_ground_truth_pkl(args.gt_pkl_dir, args.limit, args.label_space)}
    if not gt_by_id:
        raise SystemExit(f"no supervised ground truth under {args.gt_pkl_dir}")
    rows = list(load_rows(args.probs_dir, gt_by_id))
    if not rows:
        raise SystemExit("no supervised faces matched")

    by_pred: dict[int, list] = {}
    for _, _, pred, c, truth in rows:
        by_pred.setdefault(pred, []).append((c, pred == truth))

    # Bonferroni over the classes actually searched, on top of the log2(n) correction that
    # binom_precision_lower_bound applies over candidate thresholds within a class.
    n_cls = max(len(by_pred), 1)
    delta_pc = args.delta / n_cls

    print(f"\ncalibrating at target precision {args.target_precision:.0%}, "
          f"delta {args.delta} split over {n_cls} classes -> {delta_pc:.4f} each\n")
    hdr = (f"{'predicted class':<20}{'faces':>7}{'acc':>7}{'threshold':>11}"
           f"{'accept':>9}{'observed':>10}{'certified':>11}")
    print(hdr)
    thresholds, per_class = {}, {}
    tot_acc = 0
    for pred, cls_rows in sorted(by_pred.items(), key=lambda kv: -len(kv[1])):
        thr, st = certify(cls_rows, args.target_precision, delta_pc)
        name = names[pred] if pred < len(names) else str(pred)
        acc = sum(1 for _, ok in cls_rows if ok) / len(cls_rows)
        if thr is None:
            print(f"{name:<20}{len(cls_rows):>7}{100*acc:>6.0f}%{'none':>11}"
                  f"{'0.0%':>9}{'-':>10}{'-':>11}")
        else:
            print(f"{name:<20}{len(cls_rows):>7}{100*acc:>6.0f}%{thr:>11.3f}"
                  f"{100*st['coverage']:>8.1f}%{100*st['observed_precision']:>9.1f}%"
                  f"{100*st['certified_precision']:>10.1f}%")
            thresholds[str(pred)] = thr
            tot_acc += st["accepted"]
        per_class[str(pred)] = {"name": name, "n": len(cls_rows), "accuracy": acc,
                               "threshold": thr, **st}

    # The comparison that justifies the whole file: one global cut on the same faces, same target,
    # same total delta -- so the only difference is per-class vs shared.
    g_thr, g_st = certify([(c, p == t) for _, _, p, c, t in rows],
                          args.target_precision, args.delta)
    n = len(rows)
    print(f"\nper-class:  accept {100 * tot_acc / n:.1f}% of {n} faces "
          f"at certified >={args.target_precision:.0%}")
    global_breakdown = {}
    if g_thr is None:
        print("global:     no single threshold certifies at this target")
    else:
        print(f"global:     accept {100 * g_st['coverage']:.1f}% at threshold {g_thr:.3f}")
        d = 100 * (tot_acc / n - g_st["coverage"])
        print(f"difference: {d:+.1f} points of coverage at the same certified precision")

        # The comparison that actually decides between the two, because raw coverage does not.
        # A global threshold certifies the *pool*: 95% of accepted faces are right on average.
        # It says nothing about any one class, and a rare badly-calibrated class can sit far
        # below the target inside the accepted set while the pooled number holds. Per-class
        # thresholds certify each class separately, which is a strictly stronger promise -- so if
        # this table shows a class under target, the coverage that global "wins" is coverage it
        # was not entitled to.
        print(f"\nprecision per class INSIDE the global accepted set (threshold {g_thr:.3f}):")
        print(f"{'predicted class':<20}{'accepted':>10}{'observed':>10}{'certified':>11}"
              f"{'vs target':>11}")
        for pred, cls_rows in sorted(by_pred.items(), key=lambda kv: -len(kv[1])):
            acc_rows = [ok for c, ok in cls_rows if c >= g_thr]
            name = names[pred] if pred < len(names) else str(pred)
            if not acc_rows:
                print(f"{name:<20}{0:>10}{'-':>10}{'-':>11}{'-':>11}")
                global_breakdown[str(pred)] = {"name": name, "accepted": 0}
                continue
            k, ok = len(acc_rows), sum(acc_rows)
            # Same total delta as the global search, no per-class split: this is a report on one
            # already-chosen threshold, not a search, so there is no multiplicity to pay for.
            cert = binom_precision_lower_bound(ok, k, args.delta)
            flag = "OK" if cert >= args.target_precision else "UNDER"
            print(f"{name:<20}{k:>10}{100*ok/k:>9.1f}%{100*cert:>10.1f}%{flag:>11}")
            global_breakdown[str(pred)] = {"name": name, "accepted": k,
                                           "observed_precision": ok / k,
                                           "certified_precision": cert,
                                           "meets_target": cert >= args.target_precision}
    print(f"\nreview queue: {n - tot_acc} of {n} faces ({100 * (n - tot_acc) / n:.1f}%) "
          f"under per-class thresholds")

    out = {"label_space": args.label_space, "target_precision": args.target_precision,
           "delta": args.delta, "delta_per_class": delta_pc,
           "thresholds": thresholds, "per_class": per_class,
           "n_faces": n, "accepted": tot_acc, "coverage": tot_acc / n,
           "global_threshold": g_thr,
           "global_coverage": g_st["coverage"] if g_thr is not None else 0.0,
           "global_per_class": global_breakdown,
           "class_names": list(names)}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return out


def mode_apply(args) -> dict:
    cal = json.loads(args.thresholds.read_text())
    names = cal.get("class_names") or DEFEATURE_CLASS_NAMES
    thr = {int(k): float(v) for k, v in cal["thresholds"].items()}
    rows = list(load_rows(args.probs_dir, None))

    by_model: dict[str, dict] = {}
    n_acc = 0
    for model_id, i, pred, c, _ in rows:
        # A class with no certified threshold is not accepted at any confidence. Falling back to
        # the global cut here would quietly re-admit exactly the class calibration refused.
        ok = pred in thr and c >= thr[pred]
        n_acc += ok
        m = by_model.setdefault(model_id, {"accept": [], "review": []})
        m["accept" if ok else "review"].append(
            {"face": i, "pred": pred, "name": names[pred] if pred < len(names) else str(pred),
             "confidence": round(c, 4)})

    n = len(rows)
    print(f"\naccepted {n_acc}/{n} faces ({100 * n_acc / n:.1f}%) at certified "
          f">={cal['target_precision']:.0%}; {n - n_acc} queued for review")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for model_id, m in by_model.items():
        # Least confident first: the queue is worked from the top, so the faces most likely to be
        # wrong should be the ones a reviewer sees while attention is cheapest.
        m["review"].sort(key=lambda r: r["confidence"])
        (args.out_dir / f"{model_id}.json").write_text(json.dumps(
            {"model_id": model_id, "thresholds": cal["thresholds"],
             "n_accepted": len(m["accept"]), "n_review": len(m["review"]), **m}, indent=2))

    # Models with the most review work first, so a reviewer with limited time picks up the parts
    # where a single opened model yields the most decisions.
    order = sorted(by_model.items(), key=lambda kv: -len(kv[1]["review"]))
    index = [{"model_id": mid, "n_review": len(m["review"]),
              "n_accepted": len(m["accept"])} for mid, m in order]
    (args.out_dir / "queue_index.json").write_text(json.dumps(
        {"n_models": len(by_model), "n_faces": n, "n_accepted": n_acc,
         "target_precision": cal["target_precision"], "models": index}, indent=2))
    print(f"wrote {len(by_model)} per-model files + queue_index.json to {args.out_dir}")
    if index:
        print(f"most review needed: " + ", ".join(
            f"{r['model_id']} ({r['n_review']})" for r in index[:5]))
    return {"n_faces": n, "accepted": n_acc, "n_models": len(by_model)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("calibrate", help="Fit per-class thresholds against ground truth")
    c.add_argument("--probs_dir", type=Path, required=True,
                   help="Output of scan_checkpoints --dump_probs (or infer.py .json)")
    c.add_argument("--gt_pkl_dir", type=Path, required=True,
                   help="Preprocessed split whose face_labels are exact, e.g. "
                        "brepformer/data/defeature_processed/test")
    c.add_argument("--label_space", default="5", choices=["5", "27"])
    c.add_argument("--target_precision", type=float, default=0.95)
    c.add_argument("--delta", type=float, default=DELTA,
                   help="Total failure probability, split across classes by Bonferroni and "
                        "across candidate thresholds within a class by log2(n).")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--out", type=Path, default=None)

    a = sub.add_parser("apply", help="Split an unlabelled corpus into accept / review")
    a.add_argument("--probs_dir", type=Path, required=True)
    a.add_argument("--thresholds", type=Path, required=True,
                   help="JSON written by `calibrate`")
    a.add_argument("--out_dir", type=Path, required=True)

    args = ap.parse_args()
    if args.mode == "calibrate":
        mode_calibrate(args)
    else:
        mode_apply(args)


if __name__ == "__main__":
    main()
