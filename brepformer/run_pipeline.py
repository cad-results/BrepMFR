#!/usr/bin/env python3
"""End-to-end BrepFormer pipeline: preprocess -> train -> test -> analyze.

Runs all stages in sequence with sensible defaults. Use --skip_* flags to
skip individual stages, or --only to run a single stage.

Usage:
    # Full pipeline from raw data
    python brepformer/run_pipeline.py \
        --data_dir brepformer/data/mftrcad \
        --output_dir results/full_run

    # Full pipeline with face segmentation
    python brepformer/run_pipeline.py \
        --data_dir brepformer/data/mftrcad \
        --output_dir results/face_run \
        --face_segmentation \
        --face_seg_weight 2.0 --model_cls_weight 0.5

    # Skip preprocessing (already done)
    python brepformer/run_pipeline.py \
        --data_dir brepformer/data/mftrcad \
        --output_dir results/full_run \
        --skip_preprocess

    # Only test + analyze an existing checkpoint
    python brepformer/run_pipeline.py \
        --data_dir brepformer/data/mftrcad \
        --output_dir results/full_run \
        --only test \
        --checkpoint results/full_run/train/best-epoch=50-val/f1=0.8800.ckpt

    # Quick sanity check (fast_dev_run)
    python brepformer/run_pipeline.py \
        --data_dir brepformer/data/sample \
        --output_dir /tmp/sanity \
        --fast_dev_run --num_workers 0 --batch_size 4 \
        --hidden_dim 64 --ffn_dim 128 --num_heads 8 --num_kv_heads 4 --num_layers 2
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end BrepFormer pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Stage control ---
    parser.add_argument("--only", type=str, default=None,
                        choices=["preprocess", "train", "test", "analyze"],
                        help="Run only this stage (skips all others)")
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip preprocessing")
    parser.add_argument("--skip_train", action="store_true", help="Skip training")
    parser.add_argument("--skip_test", action="store_true", help="Skip testing")
    parser.add_argument("--skip_analyze", action="store_true", help="Skip analysis")

    # --- Paths ---
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Raw data directory with graphs/ and labels/ (or preprocessed dir if --skip_preprocess)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root output directory (preprocessed/, train/, test/, analysis/ created inside)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit checkpoint path (auto-detected from training if not set)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from this checkpoint")

    # --- Preprocessing ---
    parser.add_argument("--compute_descriptors", action="store_true", default=False,
                        help="Compute D2/angle multi-sample histograms (slower but improves quality)")
    parser.add_argument("--split_ratio", type=str, default="0.8,0.1,0.1")
    parser.add_argument("--seed", type=int, default=42)

    # --- Model architecture ---
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)

    # --- Face segmentation ---
    parser.add_argument("--face_segmentation", action="store_true", default=False,
                        help="Enable face-level segmentation head with automatic class weighting")
    parser.add_argument("--face_seg_weight", type=float, default=1.0,
                        help="Loss weight for face segmentation")
    parser.add_argument("--model_cls_weight", type=float, default=1.0,
                        help="Loss weight for model classification")
    parser.add_argument("--num_face_classes", type=int, default=27)
    parser.add_argument("--face_seg_hidden_dim", type=int, default=512)
    parser.add_argument("--face_seg_dropout", type=float, default=0.3)

    # --- Training ---
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--fast_dev_run", action="store_true",
                        help="Quick 1-batch sanity check (no checkpoints saved)")
    parser.add_argument("--limit_data", type=int, default=None,
                        help="Limit total dataset to N samples (proportionally across splits). "
                             "Manifest auto-propagated to test/analyze stages.")

    # --- Test/Analysis ---
    parser.add_argument("--real_classes", action="store_true", default=False,
                        help="Remap 27 classes to 8 real machining feature categories for test/analysis")
    parser.add_argument("--analyze_modes", type=str, default="all",
                        help="Comma-separated analysis modes or 'all'")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Samples for embedding analysis")

    return parser.parse_args()


def run(cmd, description):
    """Run a subprocess command, streaming output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  $ {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\nFAILED ({elapsed:.0f}s): {description}")
        sys.exit(result.returncode)
    print(f"\nDone ({elapsed:.0f}s): {description}")
    return result


def find_best_checkpoint(train_dir):
    """Find the best checkpoint in a training output directory."""
    train_dir = Path(train_dir)
    # Look for best-*.ckpt files
    ckpts = sorted(train_dir.glob("best-*.ckpt"))
    if ckpts:
        # Sort by F1 score in filename (higher is better)
        def extract_f1(p):
            name = p.stem
            for part in name.split("-"):
                if "f1=" in part:
                    try:
                        return float(part.split("=")[1])
                    except ValueError:
                        pass
            return 0.0
        ckpts.sort(key=extract_f1, reverse=True)
        return str(ckpts[0])
    # Fallback to last.ckpt
    last = train_dir / "last.ckpt"
    if last.exists():
        return str(last)
    return None


def main():
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    preprocessed_dir = root / "preprocessed"
    train_dir = root / "train"
    test_output = root / "test_results.json"
    train_output = root / "train_results.json"
    face_preds_output = root / "face_preds.json"
    train_face_preds_output = root / "train_face_preds.json"
    analysis_dir = root / "analysis"

    # Determine which stages to run
    if args.only:
        stages = {args.only}
    else:
        stages = {"preprocess", "train", "test", "analyze"}
        if args.skip_preprocess:
            stages.discard("preprocess")
        if args.skip_train:
            stages.discard("train")
        if args.skip_test:
            stages.discard("test")
        if args.skip_analyze:
            stages.discard("analyze")

    python = sys.executable

    # ---- PREPROCESS ----
    if "preprocess" in stages:
        cmd = [
            python, "brepformer/preprocess.py",
            "--data_dir", args.data_dir,
            "--output_dir", str(preprocessed_dir),
            "--split_ratio", args.split_ratio,
            "--seed", str(args.seed),
        ]
        if args.compute_descriptors:
            cmd.append("--compute_descriptors")
        run(cmd, "Stage 1/4: Preprocessing")
        data_dir_for_training = str(preprocessed_dir)
    else:
        # If skipped, use data_dir directly (assume it's already preprocessed)
        # or use the preprocessed subdir if it exists
        if preprocessed_dir.exists() and (
            (preprocessed_dir / "train").is_dir() or (preprocessed_dir / "train.pkl").exists()
        ):
            data_dir_for_training = str(preprocessed_dir)
        else:
            data_dir_for_training = args.data_dir

    # ---- TRAIN ----
    if "train" in stages:
        cmd = [
            python, "brepformer/train_preprocessed.py",
            "--data_dir", data_dir_for_training,
            "--output_dir", str(root),
            "--exp_name", "train",
            "--hidden_dim", str(args.hidden_dim),
            "--ffn_dim", str(args.ffn_dim),
            "--num_layers", str(args.num_layers),
            "--num_heads", str(args.num_heads),
            "--num_kv_heads", str(args.num_kv_heads),
            "--dropout", str(args.dropout),
            "--batch_size", str(args.batch_size),
            "--learning_rate", str(args.learning_rate),
            "--max_epochs", str(args.max_epochs),
            "--warmup_steps", str(args.warmup_steps),
            "--gradient_clip_val", str(args.gradient_clip_val),
            "--num_workers", str(args.num_workers),
            "--accumulate_grad_batches", str(args.accumulate_grad_batches),
            "--devices", str(args.devices),
            "--precision", str(args.precision),
            "--seed", str(args.seed),
        ]
        if args.resume_from:
            cmd += ["--resume_from", args.resume_from]
        if args.face_segmentation:
            cmd += [
                "--face_segmentation",
                "--face_seg_weight", str(args.face_seg_weight),
                "--model_cls_weight", str(args.model_cls_weight),
                "--num_face_classes", str(args.num_face_classes),
                "--face_seg_hidden_dim", str(args.face_seg_hidden_dim),
                "--face_seg_dropout", str(args.face_seg_dropout),
            ]
        if args.fast_dev_run:
            cmd.append("--fast_dev_run")
        if args.limit_data:
            cmd += ["--limit_data", str(args.limit_data)]
        run(cmd, "Stage 2/4: Training")

    # Resolve checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = find_best_checkpoint(train_dir)
    if checkpoint is None and ("test" in stages or "analyze" in stages):
        print("\nNo checkpoint found. Skipping test and analysis stages.")
        stages.discard("test")
        stages.discard("analyze")

    # Resolve manifest path for downstream stages
    manifest_path = train_dir / "limit_data_manifest.json"

    # ---- TEST ----
    if "test" in stages:
        cmd = [
            python, "brepformer/test_preprocessed.py",
            "--data_dir", data_dir_for_training,
            "--checkpoint", checkpoint,
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
            "--output_file", str(test_output),
        ]
        if args.face_segmentation:
            cmd += ["--output_face_preds", str(face_preds_output)]
        if args.real_classes:
            cmd.append("--real_classes")
        if manifest_path.exists():
            cmd += ["--limit_data_manifest", str(manifest_path)]
        run(cmd, "Stage 3/4: Testing")

    # ---- TRAIN EVAL ----
    if "test" in stages:
        cmd = [
            python, "brepformer/test_preprocessed.py",
            "--data_dir", data_dir_for_training,
            "--checkpoint", checkpoint,
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
            "--split", "train",
            "--output_file", str(train_output),
        ]
        if args.face_segmentation:
            cmd += ["--output_face_preds", str(train_face_preds_output)]
        if args.real_classes:
            cmd.append("--real_classes")
        if manifest_path.exists():
            cmd += ["--limit_data_manifest", str(manifest_path)]
        run(cmd, "Stage 3b/4: Train set evaluation")

    # ---- ANALYZE ----
    if "analyze" in stages:
        cmd = [
            python, "brepformer/analyze.py",
            "--checkpoint", checkpoint,
            "--data_dir", data_dir_for_training,
            "--mode", args.analyze_modes,
            "--output_dir", str(analysis_dir),
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
            "--num_samples", str(args.num_samples),
        ]
        if args.real_classes:
            cmd.append("--real_classes")
        if manifest_path.exists():
            cmd += ["--limit_data_manifest", str(manifest_path)]
        run(cmd, "Stage 4/4: Analysis")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"{'='*60}")
    print(f"  Output directory: {root}")
    if checkpoint:
        print(f"  Best checkpoint:  {checkpoint}")
    if test_output.exists():
        with open(test_output) as f:
            results = json.load(f)
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
    print()


if __name__ == "__main__":
    main()
