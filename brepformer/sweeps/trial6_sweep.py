#!/usr/bin/env python3
"""The trial6 / trial6a hyperparameter sweep: one definition, three consumers.

Why a definition file instead of a shell script: the sweep's commands, the SCRIPTS.md
documentation and the results table all have to describe the same 12 runs, and keeping
three hand-written copies in sync is how a table ends up claiming a run used settings it
did not. ``--emit`` prints the commands for SCRIPTS.md, ``--run`` executes them, and
``collect_results.py`` reads the same table to label its rows.

The design answers three confounded questions that earlier runs could not separate:

1. **Does the pcq pseudo-labelled data help?**  Every treatment has a ``*ctl*`` twin
   trained without it, so a gain can be attributed to the data rather than to the
   hyperparameters that changed alongside it.
2. **Does class weighting or loss rebalancing help?**  A 2x2 factorial over
   ``--weighted_crossentropy`` and ``--face_seg_weight/--model_cls_weight``, since half
   the loss budget currently goes to a model-level head that is never deployed.
3. **Is the fine-tune advantage real?**  ``trial1_finetune_ss`` beat
   ``trial1_ss1500`` at face F1, but only the fine-tune run had a working LR schedule --
   the from-scratch baseline never finished ``warmup_steps=5000`` at batch 32 and so
   never exceeded 45% of its nominal LR, with its ReduceLROnPlateau writes overwritten
   every batch. At batch 2 warmup completes in ~5 epochs, so the from-scratch arms here
   are the first properly-scheduled ones, and the ``init_*`` arms warm-start from both
   existing 5-class checkpoints to test the same thing from the other side.

**What the completed family-5 run changed about how to use this file (2026-09-04).** All 11
arms finished, ~72 GPU-hours, and *nothing* beat the checkpoint the warm starts were seeded
from. Two consequences are now enforced here rather than left as advice:

- **From-scratch arms are opt-in.** Every one of the six from-scratch family-5 arms collapsed
  to a constant face prediction (face macro F1 0.089-0.198, i.e. ~1/5). Five of the seven
  family-27 arms are also from-scratch, on ~12x the samples per epoch, so running them
  unguarded spends tens of GPU-hours reproducing a known collapse. They need
  ``--include_scratch`` now.
- **``--attr_layout`` is pinned explicitly.** The model's default decoding changed on
  2026-09-04, and a checkpoint trained under one layout scores ~50 points worse under the
  other. Without pinning, re-running an arm today would silently not reproduce the recorded
  number. The default here is ``categorical`` — the layout the recorded numbers were produced
  under — so a re-run reproduces; pass ``--attr_layout onehot`` to sweep the *fixed* model.

Usage:
    python -m brepformer.sweeps.trial6_sweep --emit          # commands for SCRIPTS.md
    python -m brepformer.sweeps.trial6_sweep --run --family 5    # run the 5-class arms
    python -m brepformer.sweeps.trial6_sweep --run --family 27   # run the 27-class arms
    python -m brepformer.sweeps.trial6_sweep --run --family 27 --include_scratch
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

#: Existing checkpoints used as warm starts. Both are 5-class, architecturally
#: identical, and differ only in how they were trained -- see README "Face Segmentation
#: on Real Parts".
#:
#: NOTE: ``CKPT_DEFAULT`` is a historical name -- it was defeature_v2.py's default when this
#: sweep was written. It is no longer: defeature_v2 now ships ``CKPT_FINETUNE``, which scores
#: 0.8886 against this one's 0.8564 on ``defeature_processed/test``. The name is kept so the
#: recorded arm names (``t6_init_def*``) still mean what the results table says they mean.
CKPT_DEFAULT = "results/trial1_ss1500/best-epoch=57-val/f1=0.9415.ckpt"
CKPT_FINETUNE = "results/trial1_finetune_ss/best-epoch=32-val/f1=0.9354.ckpt"
CKPT_TRIAL5 = "results/trial5/best-epoch=39-val/f1=0.8832.ckpt"

#: Shared settings. batch 2 x accum 16 = effective batch 32, matching both baselines,
#: while keeping peak memory survivable at --max_faces 1000 (the descriptor tensors are
#: 2*(n+1)^2*64 float32, so one 1000-face sample is ~513 MB and the collator pads to the
#: batch maximum).
COMMON = [
    "--max_faces", "1000",
    "--batch_size", "2",
    "--accumulate_grad_batches", "16",
    "--learning_rate", "0.001",
    "--max_epochs", "200",
    "--seed", "42",
    "--output_dir", "results",
    "--face_segmentation",
    # Both monitors are moved off their historical defaults, identically for every arm,
    # so the comparison stays clean. train/loss_epoch decayed the LR on *training* loss
    # and so kept cutting it while the model overfit; val/loss on a pseudo-labelled
    # corpus mixes ground truth with teacher noise, so a run could be stopped by the
    # noise rather than by the task. val/face_f1 is the axis defeature_v2 deploys on.
    "--lr_monitor", "val/face_f1",
    "--early_stop_monitor", "val/face_f1",
    "--early_stop_patience", "25",
]

DATA_5_PCQ = "brepformer/data/trial6_processed"          # defeature + pcq5
DATA_5_CTL = "brepformer/data/defeature_processed"       # defeature only
DATA_27_PCQ = "brepformer/data/trial6a_processed"        # mftrcad + pcq27
DATA_27_CTL = "brepformer/data/mftrcad_processed"        # mftrcad only

#: The two loss cells that get crossed with every initialisation. The full 2x2 over
#: (weighted x loss balance) is only run from scratch; for the warm starts just the two
#: corners are run, because a warm start that loses at both corners will not win in
#: between and each arm is a full training run.
LOSS_CELLS = {
    "": [],
    "_w": ["--weighted_crossentropy"],
    "_f": ["--face_seg_weight", "2.0", "--model_cls_weight", "0.5"],
    "_wf": ["--weighted_crossentropy",
            "--face_seg_weight", "2.0", "--model_cls_weight", "0.5"],
}

# name, family, data_dir, n_classes, extra flags, one-line purpose
SWEEP: list[dict] = [
    # ---- 5-class family: 2x2 factorial on (weighted x loss balance) ----
    dict(scratch=True, name="t6_base", family=5, data=DATA_5_PCQ, extra=[],
         purpose="pcq data, baseline loss - the straight comparison to t6ctl_base"),
    dict(scratch=True, name="t6_w", family=5, data=DATA_5_PCQ, extra=["--weighted_crossentropy"],
         purpose="pcq data + inverse-frequency class weights"),
    dict(scratch=True, name="t6_f", family=5, data=DATA_5_PCQ,
         extra=["--face_seg_weight", "2.0", "--model_cls_weight", "0.5"],
         purpose="pcq data + loss shifted 4:1 toward face segmentation"),
    dict(scratch=True, name="t6_wf", family=5, data=DATA_5_PCQ,
         extra=["--weighted_crossentropy",
                "--face_seg_weight", "2.0", "--model_cls_weight", "0.5"],
         purpose="pcq data + both loss changes (interaction cell)"),
    # ---- 5-class family: warm starts from BOTH existing 5-class checkpoints, each
    # crossed with the two loss corners. This is the "checkpoint issue": the default
    # that ships in defeature_v2.py and the one that actually measures best are
    # different models, so neither is assumed. ----
    dict(name="t6_init_def", family=5, data=DATA_5_PCQ,
         extra=["--init_weights", CKPT_DEFAULT],
         purpose="continue the CURRENT defeature_v2 default (0.8564) on the pcq corpus"),
    dict(name="t6_init_def_wf", family=5, data=DATA_5_PCQ,
         extra=["--init_weights", CKPT_DEFAULT] + LOSS_CELLS["_wf"],
         purpose="same warm start + both loss changes"),
    dict(name="t6_init_ft", family=5, data=DATA_5_PCQ,
         extra=["--init_weights", CKPT_FINETUNE],
         purpose="continue the best-measured checkpoint (0.8886) on the pcq corpus"),
    dict(name="t6_init_ft_wf", family=5, data=DATA_5_PCQ,
         extra=["--init_weights", CKPT_FINETUNE] + LOSS_CELLS["_wf"],
         purpose="same warm start + both loss changes — the a-priori favourite"),
    # ---- 5-class controls: identical settings, NO pcq data ----
    dict(scratch=True, name="t6ctl_base", family=5, data=DATA_5_CTL, extra=[],
         purpose="CONTROL: first properly-warmed from-scratch 5-class run; isolates "
                 "the LR-schedule defect from the fine-tune advantage"),
    dict(scratch=True, name="t6ctl_wf", family=5, data=DATA_5_CTL, extra=LOSS_CELLS["_wf"],
         purpose="CONTROL: best loss settings without the pcq data"),
    dict(name="t6ctl_init_ft", family=5, data=DATA_5_CTL,
         extra=["--init_weights", CKPT_FINETUNE],
         purpose="CONTROL: best warm start without the pcq data — separates 'the "
                 "checkpoint was good' from 'the extra data helped'"),
    # ---- 27-class family (~10x the samples, run last) ----
    dict(scratch=True, name="t6a_base", family=27, data=DATA_27_PCQ, extra=[],
         purpose="mftrcad + pcq27, baseline loss; collapsed 27->5 at eval"),
    dict(scratch=True, name="t6a_w", family=27, data=DATA_27_PCQ, extra=["--weighted_crossentropy"],
         purpose="mftrcad + pcq27 + class weights (27 classes are severely imbalanced)"),
    dict(scratch=True, name="t6a_f", family=27, data=DATA_27_PCQ, extra=LOSS_CELLS["_f"],
         purpose="mftrcad + pcq27 + loss shifted toward face segmentation"),
    dict(scratch=True, name="t6a_wf", family=27, data=DATA_27_PCQ, extra=LOSS_CELLS["_wf"],
         purpose="mftrcad + pcq27 + both loss changes (interaction cell)"),
    # trial5 is the only 27-class checkpoint that loads, so it is the only warm start
    # available on this side; it is also the teacher that produced the pcq labels, which
    # is exactly why its arm needs a control.
    dict(name="t6a_init_t5", family=27, data=DATA_27_PCQ,
         extra=["--init_weights", CKPT_TRIAL5],
         purpose="continue trial5 on the pcq-augmented 27-class corpus"),
    dict(name="t6a_init_t5_wf", family=27, data=DATA_27_PCQ,
         extra=["--init_weights", CKPT_TRIAL5] + LOSS_CELLS["_wf"],
         purpose="same warm start + both loss changes"),
    dict(scratch=True, name="t6actl_base", family=27, data=DATA_27_CTL, extra=[],
         purpose="CONTROL: mftrcad only, same settings; isolates the pcq27 data"),
    # The control t6a_init_t5 actually needs. t6actl_base is from scratch, so it changes
    # the corpus AND throws away the init: if it collapses (and every from-scratch arm so
    # far has) it says nothing about whether t6a_init_t5's +4.7 macro on test came from the
    # pcq27 data or from 62 more hours of training. Holding --init_weights fixed and
    # removing only the pcq27 half of the corpus is the comparison that separates them.
    dict(name="t6actl_init_t5", family=27, data=DATA_27_CTL,
         extra=["--init_weights", CKPT_TRIAL5],
         purpose="CONTROL: continue trial5 on mftrcad ONLY; isolates the pcq27 data from "
                 "the extra epochs, with the init held fixed"),
]


def command(run: dict, attr_layout: str = "categorical",
            rotate_aug: str = "none") -> list[str]:
    n = run["family"]
    # --attr_layout is ALWAYS emitted, never left to the model's default. The default
    # flipped on 2026-09-04 and a checkpoint trained under one layout scores ~50 points
    # worse decoded under the other, so an unpinned re-run would silently not reproduce
    # the number in the results table. --rotate_aug is only emitted when it is on, so the
    # commands recorded in SCRIPTS.md stay byte-identical to what was actually run.
    layout = ["--attr_layout", attr_layout]
    rot = ["--rotate_aug", rotate_aug] if rotate_aug != "none" else []
    return ([sys.executable, "brepformer/train_preprocessed.py",
             "--data_dir", run["data"],
             "--num_classes", str(n), "--num_face_classes", str(n)]
            + COMMON + layout + rot + run["extra"]
            + ["--exp_name", run["name"] + suffix(attr_layout, rotate_aug)])


def suffix(attr_layout: str, rotate_aug: str) -> str:
    """Name suffix so a re-swept arm cannot overwrite the recorded one's results dir.

    ``results/t6_init_ft/`` holds a specific measurement. Re-running the same arm under the
    fixed layout is a *different* experiment and writing it to the same directory would
    destroy the number the tables cite, so it lands in ``t6_init_ft__onehot`` instead.
    """
    parts = []
    if attr_layout != "categorical":
        parts.append(attr_layout)
    if rotate_aug != "none":
        parts.append(f"rot{rotate_aug}")
    return ("__" + "_".join(parts)) if parts else ""


def select(family: int | None, include_scratch: bool) -> list[dict]:
    """Arms to act on. From-scratch arms are excluded unless asked for.

    All six from-scratch family-5 arms collapsed to a constant face prediction (macro F1
    0.089-0.198 against a 1/5 = 0.2 floor), and five of the seven family-27 arms are also
    from-scratch on ~12x the samples per epoch. Defaulting them off keeps a bare
    ``--run --family 27`` from spending tens of GPU-hours to reproduce a known collapse.
    """
    todo = [r for r in SWEEP if not family or r["family"] == family]
    if not include_scratch:
        todo = [r for r in todo if not r.get("scratch")]
    return todo


def emit(family: int | None, include_scratch: bool = True,
         attr_layout: str = "categorical", rotate_aug: str = "none") -> None:
    for run in select(family, include_scratch):
        print(f"# {run['name']}: {run['purpose']}")
        # SCRIPTS.md uses a bare `python` (the env is assumed active), and wraps at
        # flag boundaries so a reader can diff two arms line by line.
        parts = command(run, attr_layout, rotate_aug)[1:]  # drop the interpreter path
        line = "python " + " ".join(
            shlex.quote(p) if " " in p else p for p in parts)
        print(line.replace(" --", " \\\n    --"))
        print()


def launch(cmd: list[str], log: Path, out: Path, retries: int,
           resume_first: bool = False) -> tuple[int, int]:
    """Run ``cmd``, resuming from ``out/last.ckpt`` if it dies transiently. Returns (rc, attempts).

    ``t6a_init_t5_wf`` died at epoch 11 with ``RuntimeError: CUDA error: unknown error`` raised
    from inside ``torchmetrics`` — the generic WSL2 symptom of the GPU dropping out, not a fault
    in the model, since ten epochs had already completed on the same batch shapes. Because the
    driver ran the arms in a bare ``subprocess.call`` loop and this was the last arm, the failure
    left the GPU **idle for four days** before anyone looked. That is the expensive failure mode
    here, not the crash itself.

    Retries append to the log rather than truncating it, so the original traceback survives, and
    each retry adds ``--resume_from last.ckpt`` to restore optimizer and epoch state (a plain
    relaunch would silently restart from ``--init_weights`` at epoch 0 and quietly report a
    different experiment under the same name). A run that dies *before* writing ``last.ckpt`` is
    not retried: with no checkpoint the retry is a fresh start, and a crash in the first epoch is
    far more likely to be a real configuration error than a dropped GPU.
    """
    for attempt in range(1, retries + 2):
        resume = out / "last.ckpt"
        extra = []
        if attempt == 1 and resume_first and resume.exists():
            # Picking a crashed arm back up rather than paying for its epochs twice. Only
            # reached when status.json records a non-zero rc for this arm, so a completed
            # arm is never silently continued.
            extra = ["--resume_from", str(resume)]
            print(f"    previous attempt failed; resuming from {resume}", flush=True)
        elif attempt > 1:
            if not resume.exists():
                print(f"    no {resume.name}; not retrying (a crash before the first "
                      f"checkpoint is a config error, not a dropped GPU)", flush=True)
                return rc, attempt - 1
            extra = ["--resume_from", str(resume)]
            print(f"    retry {attempt - 1}/{retries}: resuming from {resume}", flush=True)
        # Append whenever this invocation continues earlier work, which includes the
        # resume-a-crashed-arm path on attempt 1. Keying only on `attempt > 1` truncated the
        # log that held the crash traceback -- the exact thing the append was added to
        # protect -- the first time that path was used.
        with open(log, "a" if (attempt > 1 or extra) else "w") as fh:
            fh.write(("\n\n=== retry %d ===\n" % (attempt - 1) if attempt > 1 else "")
                     + " ".join(cmd + extra) + "\n\n")
            fh.flush()
            rc = subprocess.call(cmd + extra, cwd=REPO, stdout=fh,
                                 stderr=subprocess.STDOUT)
        if rc == 0:
            return rc, attempt
    return rc, retries + 1


def run_all(family: int | None, dry: bool, skip_existing: bool,
            only: list[str] | None = None, include_scratch: bool = False,
            attr_layout: str = "categorical", rotate_aug: str = "none",
            retries: int = 2) -> None:
    todo = select(family, include_scratch)
    if only:
        names = {r["name"] for r in SWEEP}
        unknown = set(only) - names
        if unknown:
            raise SystemExit(f"unknown arm(s): {sorted(unknown)}\n"
                             f"available: {sorted(names)}")
        # Preserve SWEEP order rather than the order given on the command line, so a
        # partial run and a full run execute the arms in the same sequence.
        todo = [r for r in todo if r["name"] in set(only)]
    log_dir = REPO / "results" / "sweep_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    print(f"sweep: {len(todo)} runs "
          f"({', '.join(r['name'] for r in todo)})\n")
    for i, run in enumerate(todo, 1):
        # The suffixed name, not run["name"], keys the status file, the log and the output
        # dir. Otherwise re-sweeping an arm under a different layout is reported "already
        # complete, skipped" against the ORIGINAL arm's status entry, and would overwrite
        # its results directory if it did run.
        name = run["name"] + suffix(attr_layout, rotate_aug)
        out = REPO / "results" / name
        if skip_existing and status.get(name, {}).get("returncode") == 0:
            print(f"[{i}/{len(todo)}] {name}: already complete, skipped")
            continue
        cmd = (["conda", "run", "-n", "new_brepmfr", "--no-capture-output"]
               + command(run, attr_layout, rotate_aug))
        log = log_dir / f"{name}.log"
        print(f"[{i}/{len(todo)}] {name} -> {out}\n    log {log}", flush=True)
        if dry:
            print("    " + " ".join(cmd))
            continue
        t0 = time.time()
        # A prior non-zero rc plus a surviving last.ckpt means this arm crashed partway;
        # continue it instead of re-paying for the epochs it already finished.
        crashed = status.get(name, {}).get("returncode") not in (None, 0)
        rc, attempts = launch(cmd, log, out, retries, resume_first=crashed)
        dt = time.time() - t0
        status[name] = {"returncode": rc, "minutes": round(dt / 60, 1),
                        "attempts": attempts,
                        "data": run["data"], "extra": run["extra"],
                        "family": run["family"], "log": str(log),
                        "attr_layout": attr_layout, "rotate_aug": rotate_aug,
                        "scratch": bool(run.get("scratch"))}
        status_path.write_text(json.dumps(status, indent=2))
        print(f"    {'OK' if rc == 0 else f'FAILED rc={rc}'} in {dt/60:.1f} min",
              flush=True)
        if rc != 0:
            print(f"    last lines of {log}:")
            tail = log.read_text().splitlines()[-15:]
            for ln in tail:
                print("      " + ln)
    print(f"\nstatus written to {status_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit", action="store_true", help="Print commands for SCRIPTS.md")
    ap.add_argument("--run", action="store_true", help="Execute the sweep sequentially")
    ap.add_argument("--family", type=int, choices=[5, 27], default=None,
                    help="Restrict to the 5-class or 27-class arms. The 27-class arms "
                         "train on ~23,500 models vs ~1,900, so run 5 first.")
    ap.add_argument("--only", nargs="+", metavar="ARM",
                    help="Run only these arms by name. Use to start the control arms "
                         "(which need no pcq data) before preprocessing has finished, "
                         "or to retry a single failure.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--rerun_completed", action="store_true",
                    help="Re-run arms already marked complete in status.json")
    ap.add_argument("--include_scratch", action="store_true",
                    help="Also run the arms with no --init_weights. OFF by default: all six "
                         "from-scratch family-5 arms collapsed to a constant face prediction "
                         "(macro F1 0.089-0.198 vs a 0.2 floor), and five of the seven "
                         "family-27 arms are from-scratch on ~12x the samples per epoch, so "
                         "a bare --family 27 would otherwise spend tens of GPU-hours "
                         "reproducing a known collapse. --emit still prints them all.")
    ap.add_argument("--attr_layout", default="categorical",
                    choices=["categorical", "onehot"],
                    help="Pinned into every emitted command rather than left to the model "
                         "default, which changed on 2026-09-04. 'categorical' reproduces the "
                         "recorded numbers; 'onehot' sweeps the FIXED model and writes to "
                         "'<arm>__onehot' so it cannot overwrite them.")
    ap.add_argument("--retries", type=int, default=2,
                    help="Relaunch a failed arm this many times, resuming from its "
                         "last.ckpt. Guards against the transient 'CUDA error: unknown "
                         "error' that killed t6a_init_t5_wf at epoch 11 and then left the "
                         "GPU idle for four days. 0 disables.")
    ap.add_argument("--rotate_aug", default="none",
                    choices=["none", "octahedral", "so3"],
                    help="Train-split rotation augmentation. Suffixes the arm name, as above.")
    args = ap.parse_args()
    if args.emit:
        # --emit prints every arm including the from-scratch ones unless explicitly
        # narrowed: SCRIPTS.md documents what the sweep IS, not just what is worth re-running.
        emit(args.family, True, args.attr_layout, args.rotate_aug)
    elif args.run:
        run_all(args.family, args.dry_run, not args.rerun_completed, args.only,
                args.include_scratch, args.attr_layout, args.rotate_aug, args.retries)
    else:
        ap.error("pass --emit or --run")


if __name__ == "__main__":
    main()
