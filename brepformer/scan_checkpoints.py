#!/usr/bin/env python3
"""Rank every saved checkpoint of a run by FACE-level F1 on a held-out split.

``ModelCheckpoint`` monitors ``val/f1``, which is the model-level multi-label F1 —
not per-face segmentation quality. The two rank epochs differently, so the file named
``best-*`` is not necessarily the best segmenter, and the top-k pruning kept whichever
epochs won on the wrong metric. Every surviving checkpoint is therefore a candidate,
and re-scoring them costs one forward pass over the test split each: no training.

The test samples are loaded and prepared **once** and reused across checkpoints, which
is the whole point of a dedicated script — 60+ invocations of ``eval_per_class.py``
would re-read and re-collate the split every time.

Reports macro F1 (every class counts equally — the metric that notices a dead rare
class) alongside weighted F1 (dominated by ``cut``/``hole``), because a run can improve
one while losing the other.

Usage:
    python -m brepformer.scan_checkpoints \
        --ckpt_dirs results/trial1_ss1500 results/trial1_ss \
        --data_dir brepformer/data/defeature_processed --split test --label_space 5
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from brepformer.data.classes import CLASS_NAMES, DEFEATURE_CLASS_NAMES, CLASS_TO_DEFEATURE
from brepformer.infer import load_model, prepare_batch


def _to_defeature(probs: np.ndarray) -> np.ndarray:
    """Collapse a 27-class posterior into the 5 defeature classes by SUMMING mass.

    Not by remapping the argmax. The two differ: 27-class mass can be split across several
    ids that map to one defeature class, so argmax-then-remap understates the confidence the
    model actually has in that class — which is both the wrong decision rule and precisely
    the quantity a risk-coverage curve ranks on. 27-class ids with no defeature counterpart
    (``CLASS_TO_DEFEATURE[c] < 0``) are dropped, then the row is renormalized, so the result
    is a proper distribution over the 5 classes rather than one that sums to less than 1.
    """
    merged = np.zeros((probs.shape[0], len(DEFEATURE_CLASS_NAMES)), dtype=np.float64)
    for c in range(probs.shape[1]):
        d = CLASS_TO_DEFEATURE[c] if c < len(CLASS_TO_DEFEATURE) else -1
        if d >= 0:
            merged[:, d] += probs[:, c]
    return merged / np.clip(merged.sum(axis=1, keepdims=True), 1e-12, None)


def score_checkpoint(ckpt: Path, samples: list, device, label_space: str,
                     data_classes: int, dump_probs: Path | None = None,
                     collapse: str = "argmax_remap") -> dict | None:
    """Per-class + macro/weighted face F1 for one checkpoint. None if unloadable.

    ``dump_probs`` additionally writes one ``{model_id}.json`` per model with
    ``face_preds`` / ``face_probs``, the format ``label_harness.py --mode select`` consumes.
    Dumping from here rather than from ``infer.py`` means the risk-coverage curve is
    computed over *exactly* the pickles every checkpoint F1 in this repo was measured on,
    so coverage/precision and macro F1 are finally on one footing.

    ``collapse`` selects how a 27-class head is scored in the 5-class space, and the two
    settings answer different questions:

    - ``argmax_remap`` takes the 27-class argmax and maps it down. This is bug 1's pattern and
      it *understates* the model, because 27-class mass split across several ids that map to
      one defeature class loses to a single id that does not. It is the default anyway,
      because every 27-class-head-at-``--label_space 5`` number in SCRIPTS.md was produced
      this way and silently changing it would make the tables incomparable.
    - ``marginalize`` sums the posterior into each defeature class first and then takes the
      argmax — the decision rule ``infer.py`` actually deploys after the bug-1 fix. Use this
      for any *new* claim about how a 27-class model performs on 5-class parts.
    """
    try:
        model = load_model(str(ckpt)).to(device).eval()
    except Exception as e:  # noqa: BLE001
        print(f"  {ckpt.name}: unloadable ({type(e).__name__}: {e})", file=sys.stderr)
        return None

    n_out = model.config.num_face_classes
    names = CLASS_NAMES if label_space == "27" else DEFEATURE_CLASS_NAMES
    remap_pred = label_space == "5" and n_out == 27
    remap_target = label_space == "5" and data_classes == 27

    if dump_probs is not None:
        dump_probs.mkdir(parents=True, exist_ok=True)

    tp, fp, fn = collections.Counter(), collections.Counter(), collections.Counter()
    for sample, y in samples:
        try:
            batch = prepare_batch(sample, device)
            with torch.no_grad():
                out = model(batch)
        except Exception:  # noqa: BLE001
            continue
        nf = int(sample["num_faces"])
        logits = out["face_logits"][0, :nf]
        pred = logits.argmax(dim=-1).cpu().numpy()

        # Marginalizing needs the posterior, so compute it once and let both the scoring loop
        # and --dump_probs read the same array rather than softmaxing twice.
        marginalized = None
        if remap_pred and collapse == "marginalize":
            marginalized = _to_defeature(torch.softmax(logits.float(), dim=-1).cpu().numpy())

        if dump_probs is not None:
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            if remap_pred:
                # Dumps are ALWAYS marginalized regardless of --collapse: a risk-coverage
                # curve ranks on confidence, and argmax-then-remap does not produce one.
                probs = marginalized if marginalized is not None else _to_defeature(probs)
            dump_preds = probs.argmax(axis=1)
            (dump_probs / f"{sample['model_id']}.json").write_text(json.dumps({
                "model_id": sample["model_id"],
                "checkpoint": str(ckpt),
                "label_space": label_space,
                "face_preds": [int(p) for p in dump_preds],
                "face_probs": [[round(float(v), 6) for v in row] for row in probs],
            }))

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
                p = (int(marginalized[i].argmax()) if marginalized is not None
                     else (CLASS_TO_DEFEATURE[p] if 0 <= p < len(CLASS_TO_DEFEATURE) else -1))
            if t == p:
                tp[t] += 1
            else:
                fn[t] += 1
                fp[p] += 1

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    per_class, f1s, weights = {}, [], []
    for c in sorted(set(list(tp) + list(fn))):
        n = tp[c] + fn[c]
        if n == 0:
            continue
        npred = tp[c] + fp[c]
        prec = tp[c] / npred if npred else 0.0
        rec = tp[c] / n
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[names[c]] = {"n": n, "precision": prec, "recall": rec, "f1": f1}
        f1s.append(f1)
        weights.append(n)
    if not f1s:
        return None
    total = sum(weights)
    return {
        "checkpoint": str(ckpt),
        "head_classes": n_out,
        "collapse": collapse if remap_pred else None,
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(sum(f * w for f, w in zip(f1s, weights)) / total),
        "min_class_f1": float(min(f1s)),
        "per_class": per_class,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt_dirs", type=Path, nargs="+", required=True,
                    help="Run directories to scan; searched recursively for *.ckpt "
                         "because lightning writes 'best-epoch=NN-val/f1=X.ckpt', and "
                         "the '/' makes each checkpoint a DIRECTORY containing the file.")
    ap.add_argument("--data_dir", type=Path,
                    default=Path("brepformer/data/defeature_processed"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap the number of test models. Leave unset for the full "
                         "split — a subsample reorders near-tied checkpoints.")
    ap.add_argument("--label_space", default="5", choices=["27", "5"])
    ap.add_argument("--mask_policy", default="asis", choices=["asis", "const", "zero"],
                    help="Overwrite face_grid channel 6 (the UV validity mask) with a constant "
                         "before scoring. Bug 8 candidate fix C. Scoring an 'asis'-trained "
                         "checkpoint under 'const' measures how much the model leans on a "
                         "channel that is dead on real parts; it is not a fix on its own.")
    ap.add_argument("--collapse", default="argmax_remap",
                    choices=["argmax_remap", "marginalize"],
                    help="How a 27-class head is scored at --label_space 5. Default reproduces "
                         "every such number already in SCRIPTS.md (argmax in 27-space, then map "
                         "down) even though that is bug 1's pattern and understates the model. "
                         "Use 'marginalize' -- sum the posterior into the 5 classes, then argmax, "
                         "the rule infer.py deploys -- for any NEW cross-label-space claim. "
                         "Ignored for a 5-class head or --label_space 27.")
    ap.add_argument("--skip_last", action="store_true",
                    help="Skip last.ckpt / last-vN.ckpt (final epoch, not selected "
                         "for anything). They are included by default: the metric "
                         "that pruned the others was the wrong one, so the final "
                         "epoch is as plausible a winner as any 'best-*'.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dump_probs", type=Path, default=None,
                    help="Write per-model face_preds/face_probs JSONs here, for "
                         "`label_harness.py --mode select`. Pass exactly ONE checkpoint "
                         "when using this, or later ones silently overwrite earlier ones. "
                         "Note for a 27-class head scored at --label_space 5: the dumped "
                         "face_preds are argmax of the SUMMED 5-class mass, while the F1 "
                         "above is argmax-then-remap (kept unchanged so every number in "
                         "SCRIPTS.md stays comparable), so the two can disagree on a few "
                         "faces where 27-class mass splits within one defeature class.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="Force CPU to score checkpoints while a training run owns the "
                         "GPU: this is one forward pass per model, so it is slow but it "
                         "cannot OOM the trainer.")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    meta_path = args.data_dir / "metadata.json"
    data_classes = 27
    if meta_path.exists():
        data_classes = int(json.loads(meta_path.read_text()).get("num_classes", 27))

    ckpts: list[Path] = []
    for d in args.ckpt_dirs:
        found = sorted(p for p in d.rglob("*.ckpt") if p.is_file())
        if args.skip_last:
            found = [p for p in found if not p.name.startswith("last")]
        print(f"{d}: {len(found)} checkpoints")
        ckpts.extend(found)
    if not ckpts:
        raise SystemExit("no checkpoints found")

    split_dir = args.data_dir / args.split
    pkls = sorted(p for p in split_dir.iterdir() if p.suffix == ".pkl")
    if args.limit:
        pkls = pkls[:args.limit]
    samples = []
    for pkl in pkls:
        with open(pkl, "rb") as f:
            s = pickle.load(f)
        if args.mask_policy != "asis":
            # Applied here, where the pickles are read, because scan_checkpoints bypasses
            # PreprocessedDataset entirely and goes straight to prepare_batch.
            s["face_grid"] = np.asarray(s["face_grid"]).copy()
            s["face_grid"][:, 6, :, :] = 1.0 if args.mask_policy == "const" else 0.0
        y = np.asarray(s["face_labels"])
        if (y >= 0).sum():
            samples.append((s, y))
    print(f"loaded {len(samples)} labelled models from {split_dir}\n"
          f"scoring {len(ckpts)} checkpoints on {args.label_space}-class face F1 "
          f"({device})\n")

    results = []
    t0 = time.time()
    for i, ck in enumerate(ckpts, 1):
        r = score_checkpoint(ck, samples, device, args.label_space, data_classes,
                             dump_probs=args.dump_probs, collapse=args.collapse)
        if r is None:
            continue
        results.append(r)
        rate = (time.time() - t0) / i
        print(f"  [{i}/{len(ckpts)}] macro {r['macro_f1']:.4f}  "
              f"weighted {r['weighted_f1']:.4f}  minclass {r['min_class_f1']:.4f}  "
              f"{ck.parent.parent.name}/{ck.parent.name}  "
              f"(eta {rate * (len(ckpts) - i) / 60:.0f}m)", flush=True)

    results.sort(key=lambda r: -r["macro_f1"])
    class_order = list(results[0]["per_class"].keys())
    print(f"\n{'rank':<5}{'macro':>8}{'weight':>8}{'minCls':>8}  "
          + "".join(f"{c[:9]:>10}" for c in class_order) + "  run/epoch")
    for rank, r in enumerate(results[:25], 1):
        cells = "".join(f"{r['per_class'].get(c, {}).get('f1', 0.0):>10.3f}"
                        for c in class_order)
        p = Path(r["checkpoint"])
        print(f"{rank:<5}{r['macro_f1']:>8.4f}{r['weighted_f1']:>8.4f}"
              f"{r['min_class_f1']:>8.4f}  {cells}  "
              f"{p.parent.parent.name}/{p.parent.name}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out} ({len(results)} checkpoints)")


if __name__ == "__main__":
    main()
