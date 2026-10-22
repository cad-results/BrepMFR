#!/usr/bin/env python3
"""Per-class face-segmentation report for a trained checkpoint on preprocessed data.

Answers "why does class X not work" with evidence rather than inference: gives
per-class precision/recall/F1, what each class gets confused *with*, how confident
the model is when it does predict the class, and the mean face area per class (the
usual reason a small feature is indistinguishable from plain stock).

Runs on the preprocessed pickles, so features are exactly what training saw.

Usage:
    python -m brepformer.eval_per_class \
        --checkpoint "results/trial5/best-epoch=39-val/f1=0.8832.ckpt" \
        --data_dir brepformer/data/mftrcad_processed --split test --limit 400
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from brepformer.data.classes import CLASS_NAMES, DEFEATURE_CLASS_NAMES, CLASS_TO_DEFEATURE
from brepformer.infer import load_model, prepare_batch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", type=Path,
                    default=Path("brepformer/data/mftrcad_processed"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--label_space", default="27", choices=["27", "5"])
    ap.add_argument("--focus", type=int, default=None,
                    help="Class id to break down by face area. A feature class can "
                         "score well overall and still be dead in deployment if the "
                         "training corpus only ever shows it at one scale, so recall "
                         "as a function of relative area is the number that predicts "
                         "transfer to real parts.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model = load_model(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    n_out = model.config.num_face_classes
    print(f"checkpoint head: {n_out} face classes; device {device}")

    names = CLASS_NAMES if args.label_space == "27" else DEFEATURE_CLASS_NAMES

    # Predictions and targets live in independent label spaces: a 27-class checkpoint
    # scored on defeature_processed needs its PREDICTIONS collapsed, but the targets are
    # already 5-class. Remapping both silently folded fillet+chamfer+cut into one bucket
    # and renamed random->chamfer, so read the target space from the corpus metadata
    # rather than assuming it matches the head.
    meta_path = args.data_dir / "metadata.json"
    data_classes = 27
    if meta_path.exists():
        data_classes = int(json.loads(meta_path.read_text()).get("num_classes", 27))
    remap_pred = args.label_space == "5" and n_out == 27
    remap_target = args.label_space == "5" and data_classes == 27
    if remap_pred != remap_target:
        print(f"remapping 27->5: predictions={remap_pred} targets={remap_target} "
              f"(head {n_out}-class, corpus {data_classes}-class)")

    split_dir = args.data_dir / args.split
    pkls = sorted(p for p in split_dir.iterdir() if p.suffix == ".pkl")[:args.limit]
    print(f"evaluating {len(pkls)} models from {split_dir}")

    tp = collections.Counter()
    fp = collections.Counter()
    fn = collections.Counter()
    confusion = collections.Counter()
    pred_conf = collections.defaultdict(list)     # per predicted class
    true_conf = collections.defaultdict(list)     # confidence assigned to true class
    area_by_class = collections.defaultdict(list)
    focus_pts = []           # (area, hit) for --focus
    n_faces = 0

    for pkl in pkls:
        with open(pkl, "rb") as f:
            sample = pickle.load(f)
        y = np.asarray(sample["face_labels"])
        if (y >= 0).sum() == 0:
            continue
        try:
            batch = prepare_batch(sample, device)
            with torch.no_grad():
                out = model(batch)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {pkl.name}: {e}", file=sys.stderr)
            continue
        nf = int(sample["num_faces"])
        logits = out["face_logits"][0, :nf]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred = probs.argmax(axis=-1)

        fa = np.asarray(sample["face_attr"])
        areas = fa[:, 9] if fa.shape[1] > 9 else np.zeros(nf)

        for i in range(min(nf, len(y))):
            t = int(y[i])
            if t < 0:
                continue
            p = int(pred[i])
            if remap_target:
                t = CLASS_TO_DEFEATURE[t] if 0 <= t < len(CLASS_TO_DEFEATURE) else -1
                if t < 0:
                    continue
            if remap_pred:
                p = CLASS_TO_DEFEATURE[p] if 0 <= p < len(CLASS_TO_DEFEATURE) else -1
            n_faces += 1
            area_by_class[t].append(float(areas[i]) if i < len(areas) else 0.0)
            true_conf[t].append(float(probs[i].max()))
            pred_conf[p].append(float(probs[i].max()))
            confusion[(t, p)] += 1
            if args.focus is not None and t == args.focus:
                focus_pts.append((float(areas[i]) if i < len(areas) else 0.0, t == p))
            if t == p:
                tp[t] += 1
            else:
                fn[t] += 1
                fp[p] += 1

    print(f"\n{n_faces} labelled faces\n")
    print(f"{'class':<26}{'n':>7}{'prec':>8}{'rec':>8}{'F1':>8}"
          f"{'#pred':>8}{'meanArea':>10}{'conf@true':>10}")
    order = sorted(set(list(tp) + list(fn)), key=lambda c: -(tp[c] + fn[c]))
    rows = {}
    for c in order:
        n = tp[c] + fn[c]
        npred = tp[c] + fp[c]
        prec = tp[c] / npred if npred else 0.0
        rec = tp[c] / n if n else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        ma = float(np.mean(area_by_class[c])) if area_by_class[c] else 0.0
        ct = float(np.mean(true_conf[c])) if true_conf[c] else 0.0
        rows[names[c]] = {"n": n, "precision": prec, "recall": rec, "f1": f1,
                          "n_predicted": npred, "mean_area": ma, "mean_conf": ct}
        print(f"{names[c]:<26}{n:>7}{prec:>8.3f}{rec:>8.3f}{f1:>8.3f}"
              f"{npred:>8}{ma:>10.5f}{ct:>10.3f}")

    print("\nwhere each weak class actually goes (true -> predicted, top 3):")
    for c in order:
        n = tp[c] + fn[c]
        if n == 0 or tp[c] / n > 0.5:
            continue
        dests = [(p, k) for (t, p), k in confusion.items() if t == c and p != c]
        dests.sort(key=lambda x: -x[1])
        s = ", ".join(f"{names[p]} {100*k/n:.0f}%" for p, k in dests[:3])
        print(f"  {names[c]:<24} n={n:<6} -> {s}")

    if args.focus is not None and focus_pts:
        focus_pts.sort(key=lambda x: x[0])
        print(f"\n'{names[args.focus]}' recall vs relative face area "
              f"(n={len(focus_pts)}, deciles):")
        k = max(len(focus_pts) // 10, 1)
        for b in range(0, len(focus_pts), k):
            chunk = focus_pts[b:b + k]
            if not chunk:
                continue
            rec = sum(h for _, h in chunk) / len(chunk)
            print(f"  area {chunk[0][0]:9.5f}..{chunk[-1][0]:9.5f}  "
                  f"n={len(chunk):<5} recall {100*rec:5.1f}%")

    if args.out:
        args.out.write_text(json.dumps({
            "checkpoint": args.checkpoint, "split": args.split,
            "label_space": args.label_space, "n_faces": n_faces,
            "per_class": rows,
            "confusion": {f"{names[t]}->{names[p]}": k
                          for (t, p), k in confusion.items()},
        }, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
