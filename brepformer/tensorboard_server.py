#!/usr/bin/env python3
"""TensorBoard server launcher for BrepFormer training visualization.

This script provides an easy way to launch TensorBoard for monitoring
BrepFormer training runs.

Usage:
    # Start TensorBoard on default port 6006
    python brepformer/tensorboard_server.py --log_dir results/brepformer

    # Start on custom port
    python brepformer/tensorboard_server.py --log_dir results --port 8080

    # Compare multiple experiments
    python brepformer/tensorboard_server.py --log_dir results --compare
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Launch TensorBoard for BrepFormer")

    parser.add_argument(
        "--log_dir",
        type=str,
        default="results",
        help="Directory containing TensorBoard logs",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="Port to run TensorBoard on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind TensorBoard to (use 0.0.0.0 for remote access)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Enable experiment comparison mode",
    )
    parser.add_argument(
        "--reload_interval",
        type=int,
        default=30,
        help="Reload interval in seconds",
    )
    parser.add_argument(
        "--samples_per_plugin",
        type=int,
        default=1000,
        help="Max samples to load per plugin",
    )

    return parser.parse_args()


def check_tensorboard_installed():
    """Check if TensorBoard is installed."""
    try:
        import tensorboard
        return True
    except ImportError:
        return False


def find_log_directories(base_dir: Path):
    """Find all directories containing TensorBoard event files."""
    log_dirs = []
    for path in base_dir.rglob("events.out.tfevents.*"):
        log_dir = path.parent
        if log_dir not in log_dirs:
            log_dirs.append(log_dir)
    return log_dirs


def main():
    """Main function to launch TensorBoard."""
    args = parse_args()

    # Check TensorBoard installation
    if not check_tensorboard_installed():
        print("Error: TensorBoard is not installed.")
        print("Install with: pip install tensorboard")
        sys.exit(1)

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory '{log_dir}' does not exist.")
        sys.exit(1)

    # Find log directories
    log_dirs = find_log_directories(log_dir)
    if not log_dirs:
        print(f"Warning: No TensorBoard event files found in '{log_dir}'")
        print("Make sure you have run training first.")

    print("=" * 60)
    print("BrepFormer TensorBoard Server")
    print("=" * 60)
    print(f"\nLog directory: {log_dir.absolute()}")
    print(f"Found {len(log_dirs)} experiment(s):")
    for d in log_dirs[:10]:  # Show first 10
        print(f"  - {d.relative_to(log_dir)}")
    if len(log_dirs) > 10:
        print(f"  ... and {len(log_dirs) - 10} more")

    print(f"\nStarting TensorBoard on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop\n")

    # Build TensorBoard command
    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", str(log_dir.absolute()),
        "--port", str(args.port),
        "--host", args.host,
        "--reload_interval", str(args.reload_interval),
        "--samples_per_plugin", f"scalars={args.samples_per_plugin}",
    ]

    if args.compare:
        # In compare mode, use each subdirectory as a separate run
        logdir_spec = ",".join(f"{d.name}:{d}" for d in log_dirs)
        cmd = [
            sys.executable, "-m", "tensorboard.main",
            "--logdir_spec", logdir_spec,
            "--port", str(args.port),
            "--host", args.host,
            "--reload_interval", str(args.reload_interval),
        ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nTensorBoard server stopped.")


if __name__ == "__main__":
    main()
