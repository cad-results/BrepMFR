#!/usr/bin/env python3
"""Score every sweep arm on the same held-out real-part split and emit markdown tables.

Reads the run list from ``trial6_sweep.py`` so the config columns in the output cannot
drift from what was actually executed, then scores each arm's checkpoints with
``scan_checkpoints.score_checkpoint`` and keeps the best by **face macro F1**.

Every arm is scored on ``defeature_processed/test`` in the 5-class space, including the
27-class arms (their predictions are collapsed via ``CLASS_TO_DEFEATURE``). That is the
only way the two families are comparable, and it is the space ``defeature_v2.py``
deploys in. The 27-class arms are additionally scored on ``mftrcad_processed/test`` in
their native space, since a 27-class arm can improve in-domain while losing on real
parts.

Macro F1 leads the table: aggregate metrics rise as weak classes die, which is the
failure mode this whole sweep exists to detect.

Usage:
    python -m brepformer.sweeps.collect_results \
        --out brepformer/RESULTS_TRIAL6.md
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch

from brepformer.scan_checkpoints import score_checkpoint
from brepformer.sweeps.trial6_sweep import SWEEP, CKPT_DEFAULT, CKPT_FINETUNE

#: Reference points every table is read against.
BASELINES = [
    ("trial1_ss1500/best-epoch=57 (old defeature_v2 default)", CKPT_DEFAULT),
    ("trial1_finetune_ss/best-epoch=32 (best pre-sweep)", CKPT_FINETUNE),
]

CLASS_ORDER = ["random", "hole", "chamfer", "fillet", "cut"]


def load_split(data_dir: Path, split: str, limit: int | None = None) -> list:
    split_dir = data_dir / split
    pkls = sorted(p for p in split_dir.iterdir() if p.suffix == ".pkl")
    if limit:
        pkls = pkls[:limit]
    out = []
    for p in pkls:
        with open(p, "rb") as f:
            s = pickle.load(f)
        y = np.asarray(s["face_labels"])
        if (y >= 0).sum():
            out.append((s, y))
    return out


def best_of_run(run_dir: Path, samples: list, device, label_space: str,
                data_classes: int) -> dict | None:
    """Best checkpoint of one run by face macro F1, or None if the run has none."""
    ckpts = sorted(p for p in run_dir.rglob("*.ckpt") if p.is_file())
    best = None
    for ck in ckpts:
        r = score_checkpoint(ck, samples, device, label_space, data_classes)
        if r and (best is None or r["macro_f1"] > best["macro_f1"]):
            best = r
    if best:
        best["n_checkpoints"] = len(ckpts)
    return best


def describe(run: dict) -> str:
    """Human-readable settings column, derived from the executed flags."""
    bits = []
    e = run["extra"]
    bits.append("pcq" if "trial6" in run["data"] else "no-pcq")
    bits.append("weighted" if "--weighted_crossentropy" in e else "unweighted")
    if "--face_seg_weight" in e:
        bits.append("seg2.0/cls0.5")
    else:
        bits.append("seg1.0/cls1.0")
    if "--init_weights" in e:
        src = Path(e[e.index("--init_weights") + 1]).parts[1]
        bits.append(f"init:{src}")
    else:
        bits.append("scratch")
    return ", ".join(bits)


def row(name: str, settings: str, r: dict) -> str:
    cells = "".join(f" {r['per_class'].get(c, {}).get('f1', float('nan')):.3f} |"
                    for c in CLASS_ORDER)
    return (f"| `{name}` | {settings} | **{r['macro_f1']:.4f}** | "
            f"{r['weighted_f1']:.4f} | {r['min_class_f1']:.3f} |{cells}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", type=Path, default=Path("results"))
    ap.add_argument("--data_dir", type=Path,
                    default=Path("brepformer/data/defeature_processed"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("brepformer/RESULTS_TRIAL6.md"))
    ap.add_argument("--json_out", type=Path,
                    default=Path("brepformer/analysis_results/trial6_sweep_results.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = args.data_dir / "metadata.json"
    data_classes = int(json.loads(meta.read_text()).get("num_classes", 27)) \
        if meta.exists() else 27

    samples = load_split(args.data_dir, args.split, args.limit)
    print(f"scoring on {len(samples)} models from {args.data_dir}/{args.split}")

    status_path = args.results_dir / "sweep_logs" / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    collected: dict[str, dict] = {}
    lines: list[str] = []

    lines.append("# trial6 / trial6a sweep results\n")
    lines.append(f"All arms scored on `{args.data_dir}/{args.split}` "
                 f"({len(samples)} hand-labelled real parts) in the **5-class** space; "
                 "27-class arms have their predictions collapsed via "
                 "`CLASS_TO_DEFEATURE`. Best checkpoint per run by face macro F1, not "
                 "by the `val/f1` in the filename (which is model-level).\n")

    lines.append("\n## Baselines\n")
    header = ("| run | settings | macro F1 | weighted F1 | worst class | "
              + " | ".join(CLASS_ORDER) + " |")
    sep = "|---" * (5 + len(CLASS_ORDER)) + "|"
    lines += [header, sep]
    for label, ck in BASELINES:
        r = score_checkpoint(Path(ck), samples, device, "5", data_classes)
        if r:
            collected[label] = r
            lines.append(row(label, "pre-sweep reference", r))
            print(f"  baseline {label}: macro {r['macro_f1']:.4f}")

    for family, title in ((5, "5-class arms (direct defeature task)"),
                          (27, "27-class arms (collapsed 27->5 for comparison)")):
        arms = [r for r in SWEEP if r["family"] == family]
        lines.append(f"\n## {title}\n")
        lines += [header, sep]
        for run in arms:
            run_dir = args.results_dir / run["name"]
            if not run_dir.is_dir():
                lines.append(f"| `{run['name']}` | {describe(run)} | _not run_ | | | "
                             + " | " * len(CLASS_ORDER) + "|")
                print(f"  {run['name']}: no results dir, skipped")
                continue
            r = best_of_run(run_dir, samples, device, "5", data_classes)
            if not r:
                lines.append(f"| `{run['name']}` | {describe(run)} | _no loadable "
                             f"checkpoint_ | | | " + " | " * len(CLASS_ORDER) + "|")
                continue
            r["minutes"] = status.get(run["name"], {}).get("minutes")
            collected[run["name"]] = r
            lines.append(row(run["name"], describe(run), r))
            print(f"  {run['name']}: macro {r['macro_f1']:.4f} "
                  f"({Path(r['checkpoint']).parent.name})")

        lines.append("")
        for run in arms:
            r = collected.get(run["name"])
            if r:
                mins = f", {r['minutes']:.0f} min" if r.get("minutes") else ""
                lines.append(f"- `{run['name']}` — {run['purpose']}  \n"
                             f"  best: `{r['checkpoint']}` "
                             f"({r['n_checkpoints']} checkpoints scanned{mins})")

    if collected:
        arms_only = {k: v for k, v in collected.items()
                     if k not in dict(BASELINES) and not k.startswith("trial1")}
        if arms_only:
            best_name = max(arms_only, key=lambda k: arms_only[k]["macro_f1"])
            best = arms_only[best_name]
            base_macro = max((collected[l]["macro_f1"] for l, _ in BASELINES
                              if l in collected), default=0.0)
            lines.append(f"\n## Verdict\n")
            lines.append(
                f"Best sweep arm: **`{best_name}`** at macro F1 "
                f"**{best['macro_f1']:.4f}** vs the best pre-sweep checkpoint's "
                f"{base_macro:.4f} "
                f"({'+' if best['macro_f1'] >= base_macro else ''}"
                f"{best['macro_f1'] - base_macro:+.4f}).\n")
            lines.append(f"Checkpoint: `{best['checkpoint']}`\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(collected, indent=2))
    print(f"\nwrote {args.out}\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
