#!/usr/bin/env python3
"""Export pipe fitting classifier for inference or deployment.

Supports state_dict, TorchScript, and ONNX export.

Usage:
    python brepclassifier/export_model.py \
        --checkpoint results/pipe_classifier/best.ckpt \
        --format state_dict \
        --output_path exported_pipe_classifier.pt
"""

import argparse
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

torch.serialization.add_safe_globals([pathlib.PosixPath])


def parse_args():
    parser = argparse.ArgumentParser(description="Export pipe fitting classifier")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--format", type=str,
                        choices=["state_dict", "torchscript", "onnx"],
                        default="state_dict")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_faces", type=int, default=100)
    parser.add_argument("--max_edges", type=int, default=200)
    return parser.parse_args()


def load_model(checkpoint_path: str):
    from brepformer.configs.config import BrepClassifierConfig
    from brepclassifier.configs.config import PipeFittingConfig
    from brepclassifier.models.pipe_classifier import PipeFittingClassifier

    torch.serialization.add_safe_globals([BrepClassifierConfig, PipeFittingConfig])
    model = PipeFittingClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def create_dummy_input(model, max_faces, max_edges, device):
    batch_size = 1
    config = model.config
    return {
        "face_grid": torch.randn(batch_size, max_faces, 7, 10, 10, device=device),
        "face_attr": torch.randn(batch_size, max_faces, config.face_attr_dim, device=device),
        "edge_index": torch.randint(0, max_faces, (batch_size, 2, max_edges), device=device),
        "edge_attr": torch.randn(batch_size, max_edges, config.edge_attr_dim, device=device),
        "edge_grid": torch.randn(batch_size, max_edges, 12, 10, device=device),
        "spatial_pos": torch.randint(0, config.num_spatial, (batch_size, max_faces + 1, max_faces + 1), device=device),
        "in_degree": torch.randint(0, config.num_degree, (batch_size, max_faces), device=device),
        "attn_mask": torch.ones(batch_size, max_faces + 1, dtype=torch.bool, device=device),
    }


def export_state_dict(model, output_path):
    print("Exporting state dict...")
    state_dict = model.state_dict()
    config_dict = {k: v for k, v in vars(model.config).items() if not k.startswith("_")}
    torch.save({"state_dict": state_dict, "config": config_dict}, output_path)
    print(f"State dict saved to {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")


def export_torchscript(model, output_path, max_faces, max_edges):
    print("Exporting to TorchScript...")
    device = next(model.parameters()).device

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, face_grid, face_attr, edge_index, edge_attr,
                    edge_grid, spatial_pos, in_degree, attn_mask):
            batch = {
                "face_grid": face_grid, "face_attr": face_attr,
                "edge_index": edge_index, "edge_attr": edge_attr,
                "edge_grid": edge_grid, "spatial_pos": spatial_pos,
                "in_degree": in_degree, "attn_mask": attn_mask,
            }
            return self.model(batch)

    wrapper = Wrapper(model)
    wrapper.eval()
    dummy = create_dummy_input(model, max_faces, max_edges, device)

    try:
        traced = torch.jit.trace(wrapper, tuple(dummy.values()))
        traced.save(output_path)
        print(f"TorchScript model saved to {output_path}")
    except Exception as e:
        print(f"Error: {e}")


def export_onnx(model, output_path, max_faces, max_edges):
    print("Exporting to ONNX...")
    try:
        import onnx
    except ImportError:
        print("Error: onnx not installed")
        return

    device = next(model.parameters()).device

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, face_grid, face_attr, edge_index, edge_attr,
                    edge_grid, spatial_pos, in_degree, attn_mask):
            batch = {
                "face_grid": face_grid, "face_attr": face_attr,
                "edge_index": edge_index, "edge_attr": edge_attr,
                "edge_grid": edge_grid, "spatial_pos": spatial_pos,
                "in_degree": in_degree, "attn_mask": attn_mask,
            }
            return self.model(batch)

    wrapper = Wrapper(model)
    wrapper.eval()
    dummy = create_dummy_input(model, max_faces, max_edges, device)

    try:
        torch.onnx.export(
            wrapper,
            tuple(dummy.values()),
            output_path,
            input_names=list(dummy.keys()),
            output_names=["logits"],
            dynamic_axes={
                "face_grid": {0: "batch", 1: "faces"},
                "face_attr": {0: "batch", 1: "faces"},
                "edge_index": {0: "batch", 2: "edges"},
                "edge_attr": {0: "batch", 1: "edges"},
                "edge_grid": {0: "batch", 1: "edges"},
                "spatial_pos": {0: "batch"},
                "in_degree": {0: "batch"},
                "attn_mask": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=14,
        )
        print(f"ONNX model saved to {output_path}")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified!")
    except Exception as e:
        print(f"Error: {e}")


def main():
    args = parse_args()
    print("=" * 60)
    print("Pipe Fitting Classifier Export")
    print("=" * 60)

    model = load_model(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if args.format == "state_dict":
        export_state_dict(model, args.output_path)
    elif args.format == "torchscript":
        export_torchscript(model, args.output_path, args.max_faces, args.max_edges)
    elif args.format == "onnx":
        export_onnx(model, args.output_path, args.max_faces, args.max_edges)

    print("\nExport complete!")


if __name__ == "__main__":
    main()
