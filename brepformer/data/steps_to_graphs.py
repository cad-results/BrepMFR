#!/usr/bin/env python3
"""Batch-convert a directory of STEP files to the graph JSONs ``preprocess.py`` reads.

``preprocess.py`` wants ``<data_dir>/graphs/<model_id>.json`` alongside
``<data_dir>/labels/<model_id>_result.json``; this produces the former for any corpus
of STEP files. Previously this only existed as an inline ``python -c`` blob in
SCRIPTS.md, which meant every new corpus re-typed it and quietly dropped the error
handling.

Resumable: existing outputs are skipped, so a run killed by an OCC segfault can be
restarted and will pick up where it stopped. That matters because the OCC reader
segfaults (not raises) on a handful of real-world STEP files, taking the whole process
down — ``--tolerate_crashes`` re-execs a fresh worker per remaining file so one bad
part costs one file rather than the rest of the corpus.

Usage:
    python -m brepformer.data.steps_to_graphs \
        --steps_dir brepformer/data/suitable_samples_pcq/steps \
        --out_dir   brepformer/data/suitable_samples_pcq/graphs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def convert_one(step_file: Path, out_path: Path, mftrcad_uv: bool = False,
                uv_pad: float = 0.10) -> bool:
    """Convert a single STEP to graph JSON. Returns True on success."""
    from brepformer.data.step_to_graph import step_to_graph

    data = step_to_graph(str(step_file), mftrcad_uv=mftrcad_uv, uv_pad=uv_pad)
    if data is None:
        return False
    payload = [
        step_file.stem,
        {
            "graph": {
                "edges": data["edge_index"].tolist(),
                "num_nodes": data["num_nodes"],
            },
            "graph_face_attr": data["face_attr"].tolist(),
            "graph_face_grid": data["face_grid"].tolist(),
            "graph_edge_attr": data["edge_attr"].tolist(),
            "graph_edge_grid": data["edge_grid"].tolist(),
        },
    ]
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(out_path)  # atomic, so an interrupted run leaves no half-file
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--uv_pad", type=float, default=0.10,
                    help="UV padding fraction per side, only used with --mftrcad_uv. 0.10 is "
                         "the fitted MFTRCAD value and the only one that reproduces their "
                         "corpus. Pass 0.0 to get trim masking over the unpadded box, which "
                         "isolates the masking half of Bug 8's fix from the sampling-domain "
                         "half; that corpus is a diagnostic, not a candidate for deployment.")
    ap.add_argument("--mftrcad_uv", action="store_true",
                    help="Emit UV grids under MFTRCAD's published convention: pad the face "
                         "UV box by 10%% per side, then mask samples outside the trim loops. "
                         "Every STEP-derived corpus here was built without either half, so "
                         "face_grid channel 6 is the constant 1 (vs 44.3%% of MFTRCAD's grid "
                         "points masked) and the xyz/normal channels are sampled over a "
                         "narrower UV domain than the pretraining corpus -- an input "
                         "mismatch on the exact corpus the model is deployed on. The 10%% is "
                         "fitted against MFTRCAD's own masks; see compute_face_uv_grid. Off "
                         "by default because flipping it changes the pickles and would make "
                         "new numbers incomparable with SCRIPTS.md.")
    ap.add_argument("--tolerate_crashes", action="store_true",
                    help="Run each conversion in a subprocess so an OCC segfault "
                         "costs one file instead of aborting the run. ~50ms/file "
                         "slower from interpreter startup; worth it on real-world "
                         "STEP files, unnecessary on a corpus known to be clean.")
    ap.add_argument("--single", type=Path, default=None,
                    help=argparse.SUPPRESS)  # internal: subprocess worker mode
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Worker mode for --tolerate_crashes: convert one file, exit non-zero on failure.
    if args.single is not None:
        out = args.out_dir / f"{args.single.stem}.json"
        raise SystemExit(0 if convert_one(args.single, out, args.mftrcad_uv, args.uv_pad) else 1)

    steps = sorted(p for p in args.steps_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in {".step", ".stp"})
    todo = [p for p in steps if not (args.out_dir / f"{p.stem}.json").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(steps)} STEP files, {len(steps) - len(todo)} already converted, "
          f"{len(todo)} to do")

    ok = failed = 0
    failures: list[str] = []
    for i, step_file in enumerate(todo, 1):
        out = args.out_dir / f"{step_file.stem}.json"
        try:
            if args.tolerate_crashes:
                r = subprocess.run(
                    [sys.executable, "-m", "brepformer.data.steps_to_graphs",
                     "--steps_dir", str(args.steps_dir), "--out_dir", str(args.out_dir),
                     "--single", str(step_file)]
                    # The re-exec is a *new process*, so any flag not forwarded here is
                    # silently dropped -- and --mftrcad_uv changes what the pickles contain,
                    # so dropping it would produce a corpus that looks converted and is not.
                    + (["--mftrcad_uv"] if args.mftrcad_uv else [])
                    + ["--uv_pad", str(args.uv_pad)],
                    capture_output=True, text=True, timeout=600,
                )
                success = r.returncode == 0 and out.exists()
                err = (r.stderr or "").strip().splitlines()[-1:] or ["crashed"]
                if not success:
                    failures.append(f"{step_file.name}: {err[0][:120]}")
            else:
                success = convert_one(step_file, out, args.mftrcad_uv, args.uv_pad)
                if not success:
                    failures.append(f"{step_file.name}: step_to_graph returned None")
        except Exception as e:  # noqa: BLE001
            success = False
            failures.append(f"{step_file.name}: {type(e).__name__}: {e}")
        ok += success
        failed += not success
        if i % 25 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] ok={ok} failed={failed}", flush=True)

    print(f"\nconverted {ok}, failed {failed} -> {args.out_dir}")
    if failures:
        print("failures:")
        for f in failures[:40]:
            print(f"  {f}")
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more")


if __name__ == "__main__":
    main()
