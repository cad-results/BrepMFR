#!/usr/bin/env python3
"""Export BrepFormer model for inference or deployment.

This script exports trained models to various formats:
- PyTorch state dict (for loading in Python)
- TorchScript (for C++ deployment)
- ONNX (for cross-platform deployment)

Usage:
    # Export to TorchScript
    python brepformer/export_model.py --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
        --format torchscript --output_path exported_model.pt

    # Export to ONNX
    python brepformer/export_model.py --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
        --format onnx --output_path exported_model.onnx

    # Export state dict only
    python brepformer/export_model.py --checkpoint results/trial2/best-epoch=111-val/f1=0.8558.ckpt \
        --format state_dict --output_path model_weights.pt
"""

import argparse
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

# Add safe globals for checkpoint loading
torch.serialization.add_safe_globals([pathlib.PosixPath])


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export BrepFormer model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["state_dict", "torchscript", "onnx"],
        default="state_dict",
        help="Export format",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for exported model",
    )
    parser.add_argument(
        "--max_faces",
        type=int,
        default=100,
        help="Maximum number of faces for tracing (TorchScript/ONNX)",
    )
    parser.add_argument(
        "--max_edges",
        type=int,
        default=200,
        help="Maximum number of edges for tracing (TorchScript/ONNX)",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str):
    """Load model from checkpoint."""
    from brepformer.configs.config import BrepClassifierConfig
    from brepformer.models.brep_classifier import BrepClassifier

    torch.serialization.add_safe_globals([BrepClassifierConfig])
    model = BrepClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def create_dummy_input(model, max_faces: int, max_edges: int, device: torch.device):
    """Create dummy input for tracing."""
    batch_size = 1
    config = model.config

    dummy_input = {
        "face_grid": torch.randn(batch_size, max_faces, 7, 10, 10, device=device),
        "face_attr": torch.randn(batch_size, max_faces, config.face_attr_dim, device=device),
        "edge_index": torch.randint(0, max_faces, (batch_size, 2, max_edges), device=device),
        "edge_attr": torch.randn(batch_size, max_edges, config.edge_attr_dim, device=device),
        "edge_grid": torch.randn(batch_size, max_edges, 12, 10, device=device),
        "spatial_pos": torch.randint(0, config.num_spatial, (batch_size, max_faces + 1, max_faces + 1), device=device),
        "in_degree": torch.randint(0, config.num_degree, (batch_size, max_faces), device=device),
        "attn_mask": torch.ones(batch_size, max_faces + 1, dtype=torch.bool, device=device),
    }

    return dummy_input


def export_state_dict(model, output_path: str):
    """Export model state dict."""
    print("Exporting state dict...")

    state_dict = model.state_dict()
    config_dict = {k: v for k, v in vars(model.config).items() if not k.startswith("_")}

    export_data = {
        "state_dict": state_dict,
        "config": config_dict,
    }

    torch.save(export_data, output_path)
    print(f"State dict saved to {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")


def export_torchscript(model, output_path: str, max_faces: int, max_edges: int):
    """Export model to TorchScript."""
    print("Exporting to TorchScript...")

    device = next(model.parameters()).device

    # Create wrapper for cleaner export
    class BrepClassifierWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, face_grid, face_attr, edge_index, edge_attr,
                    edge_grid, spatial_pos, in_degree, attn_mask):
            batch = {
                "face_grid": face_grid,
                "face_attr": face_attr,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "edge_grid": edge_grid,
                "spatial_pos": spatial_pos,
                "in_degree": in_degree,
                "attn_mask": attn_mask,
            }
            return self.model(batch)

    wrapper = BrepClassifierWrapper(model)
    wrapper.eval()

    dummy_input = create_dummy_input(model, max_faces, max_edges, device)

    try:
        # Try tracing
        traced = torch.jit.trace(
            wrapper,
            (
                dummy_input["face_grid"],
                dummy_input["face_attr"],
                dummy_input["edge_index"],
                dummy_input["edge_attr"],
                dummy_input["edge_grid"],
                dummy_input["spatial_pos"],
                dummy_input["in_degree"],
                dummy_input["attn_mask"],
            ),
        )
        traced.save(output_path)
        print(f"TorchScript model saved to {output_path}")
        print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"Error during TorchScript export: {e}")
        print("TorchScript export may not be fully supported due to dynamic graph operations.")


def export_onnx(model, output_path: str, max_faces: int, max_edges: int):
    """Export model to ONNX."""
    print("Exporting to ONNX...")

    try:
        import onnx
    except ImportError:
        print("Error: onnx not installed. Install with: pip install onnx")
        return

    device = next(model.parameters()).device

    # Create wrapper
    class BrepClassifierWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, face_grid, face_attr, edge_index, edge_attr,
                    edge_grid, spatial_pos, in_degree, attn_mask):
            batch = {
                "face_grid": face_grid,
                "face_attr": face_attr,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "edge_grid": edge_grid,
                "spatial_pos": spatial_pos,
                "in_degree": in_degree,
                "attn_mask": attn_mask,
            }
            return self.model(batch)

    wrapper = BrepClassifierWrapper(model)
    wrapper.eval()

    dummy_input = create_dummy_input(model, max_faces, max_edges, device)

    input_names = ["face_grid", "face_attr", "edge_index", "edge_attr",
                   "edge_grid", "spatial_pos", "in_degree", "attn_mask"]
    output_names = ["logits"]

    try:
        torch.onnx.export(
            wrapper,
            (
                dummy_input["face_grid"],
                dummy_input["face_attr"],
                dummy_input["edge_index"],
                dummy_input["edge_attr"],
                dummy_input["edge_grid"],
                dummy_input["spatial_pos"],
                dummy_input["in_degree"],
                dummy_input["attn_mask"],
            ),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "face_grid": {0: "batch", 1: "num_faces"},
                "face_attr": {0: "batch", 1: "num_faces"},
                "edge_index": {0: "batch", 2: "num_edges"},
                "edge_attr": {0: "batch", 1: "num_edges"},
                "edge_grid": {0: "batch", 1: "num_edges"},
                "spatial_pos": {0: "batch", 1: "num_nodes", 2: "num_nodes"},
                "in_degree": {0: "batch", 1: "num_faces"},
                "attn_mask": {0: "batch", 1: "num_nodes"},
                "logits": {0: "batch"},
            },
            opset_version=14,
        )
        print(f"ONNX model saved to {output_path}")
        print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

        # Verify
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully!")
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        print("ONNX export may not be fully supported due to dynamic graph operations.")


def main():
    """Main export function."""
    args = parse_args()

    print("=" * 60)
    print("BrepFormer Model Export")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Export
    if args.format == "state_dict":
        export_state_dict(model, args.output_path)
    elif args.format == "torchscript":
        export_torchscript(model, args.output_path, args.max_faces, args.max_edges)
    elif args.format == "onnx":
        export_onnx(model, args.output_path, args.max_faces, args.max_edges)

    print("\nExport complete!")


if __name__ == "__main__":
    main()
