"""Configuration dataclass for PipeFitting classifier."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PipeFittingConfig:
    """Configuration for pipe fitting classifier.

    Two-stage architecture: BrepFormer encoder + GAT classification head.
    Inherits all BrepFormer encoder params and adds GAT-specific params.
    """

    # ── BrepFormer encoder params (same defaults as brepformer/configs/config.py) ──
    hidden_dim: int = 256
    ffn_dim: int = 512
    num_heads: int = 32
    num_kv_heads: int = 8
    num_layers: int = 8

    # Graph structure parameters
    num_degree: int = 128
    num_spatial: int = 64
    num_edge_dis: int = 64
    multi_hop_max_dist: int = 16

    # B-rep type encodings
    num_face_types: int = 6
    num_edge_types: int = 6

    # Feature dimensions
    face_attr_dim: int = 14
    face_grid_channels: int = 7
    face_grid_size: int = 10
    edge_attr_dim: int = 15
    edge_grid_channels: int = 12
    edge_grid_size: int = 10

    # Encoder regularization
    dropout: float = 0.3
    attention_dropout: float = 0.3
    activation_dropout: float = 0.3

    # Encoder CNN parameters
    surface_encoder_channels: list = field(default_factory=lambda: [64, 128, 256])
    surface_encoder_output_dim: int = 128
    curve_encoder_channels: list = field(default_factory=lambda: [64, 128])
    d2_descriptor_dim: int = 64
    angle_descriptor_dim: int = 64
    num_convexity_types: int = 3
    max_seq_len: int = 512

    # ── Classification ──
    num_classes: int = 8
    multi_label: bool = False

    # ── GAT classification head ──
    gat_num_layers: int = 3
    gat_heads: int = 4
    gat_hidden_dim: int = 256
    gat_v2: bool = True
    gat_dropout: float = 0.3
    gat_pooling: str = "global_attention"

    # Dense head
    dense_dims: list = field(default_factory=lambda: [512, 256])
    dense_dropout: float = 0.3

    # ── Weight loading ──
    pretrained_encoder_ckpt: Optional[str] = None
    freeze_encoder: bool = False
    classifier_ckpt: Optional[str] = None

    # ── Training ──
    learning_rate: float = 1e-4
    encoder_lr_factor: float = 0.1  # encoder LR = base LR * this factor
    batch_size: int = 16
    warmup_steps: int = 500
    weight_decay: float = 0.01
    adam_beta1: float = 0.99
    adam_beta2: float = 0.999
    lr_factor: float = 0.5
    lr_patience: int = 5
    min_lr: float = 1e-6
    gradient_clip_val: float = 1.0
    max_epochs: int = 300

    # ── Class info ──
    class_names: list = field(default_factory=lambda: [
        "Elbow - Weld Fitting",
        "Elbow - Pipe End Fitting",
        "Elbow - Socket Fitting",
        "Tee - Weld Fitting",
        "Tee - Pipe End Fitting",
        "Tee - Socket Fitting",
        "Elbow - Miscellaneous",
        "Tee - Miscellaneous",
    ])
    class_weights: Optional[list] = None  # auto from metadata or manual

    # Data paths
    data_dir: Optional[str] = None
    label_file: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"

    def to_encoder_config(self):
        """Convert to BrepClassifierConfig for the encoder.

        Returns a BrepClassifierConfig with the encoder-relevant fields.
        """
        from brepformer.configs.config import BrepClassifierConfig
        return BrepClassifierConfig(
            hidden_dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            num_layers=self.num_layers,
            num_degree=self.num_degree,
            num_spatial=self.num_spatial,
            num_edge_dis=self.num_edge_dis,
            multi_hop_max_dist=self.multi_hop_max_dist,
            num_face_types=self.num_face_types,
            num_edge_types=self.num_edge_types,
            face_attr_dim=self.face_attr_dim,
            face_grid_channels=self.face_grid_channels,
            face_grid_size=self.face_grid_size,
            edge_attr_dim=self.edge_attr_dim,
            edge_grid_channels=self.edge_grid_channels,
            edge_grid_size=self.edge_grid_size,
            num_classes=self.num_classes,
            multi_label=self.multi_label,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            activation_dropout=self.activation_dropout,
            surface_encoder_channels=self.surface_encoder_channels,
            surface_encoder_output_dim=self.surface_encoder_output_dim,
            curve_encoder_channels=self.curve_encoder_channels,
            d2_descriptor_dim=self.d2_descriptor_dim,
            angle_descriptor_dim=self.angle_descriptor_dim,
            num_convexity_types=self.num_convexity_types,
            max_seq_len=self.max_seq_len,
        )
