"""Configuration dataclass for BrepFormer model."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BrepClassifierConfig:
    """Configuration for BrepFormer classifier.

    Hyperparameters follow the official BrepFormer paper and implementation.
    Reference: https://github.com/dyk-npu/BRepFormer
    """

    # Model architecture (exact values from official repo)
    hidden_dim: int = 256  # embedding_dim in encoder
    ffn_dim: int = 512  # ffn_embedding_dim
    num_heads: int = 32  # num_attention_heads
    num_kv_heads: int = 8  # for grouped query attention
    num_layers: int = 8  # num_encoder_layers

    # Graph structure parameters
    num_degree: int = 128  # in-degree embedding types
    num_spatial: int = 64  # spatial position types
    num_edge_dis: int = 64  # edge distance embedding
    multi_hop_max_dist: int = 16  # max hops for edge path

    # B-rep type encodings
    num_face_types: int = 6  # surface types (plane, cylinder, cone, sphere, torus, other)
    num_edge_types: int = 6  # curve types (line, circle, ellipse, bspline, other, seam)

    # Feature dimensions (from data format)
    face_attr_dim: int = 14  # face attributes dimension
    face_grid_channels: int = 7  # UV-grid channels (x, y, z, nx, ny, nz, mask)
    face_grid_size: int = 10  # UV-grid resolution
    edge_attr_dim: int = 15  # edge attributes dimension
    edge_grid_channels: int = 12  # curve grid channels
    edge_grid_size: int = 10  # curve grid resolution

    # Classification
    num_classes: int = 27  # number of machining feature classes (MFCAD/MFTRCAD)
    multi_label: bool = True  # multi-label classification by default

    # Regularization
    dropout: float = 0.3
    attention_dropout: float = 0.3
    activation_dropout: float = 0.3

    # Training (from official train.py)
    batch_size: int = 32  # official default (64 optional)
    learning_rate: float = 0.002
    warmup_steps: int = 5000  # linear warmup
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    lr_factor: float = 0.5  # ReduceLROnPlateau factor
    lr_patience: int = 5  # ReduceLROnPlateau patience
    min_lr: float = 1e-6
    gradient_clip_val: float = 1.0
    max_epochs: int = 200  # official default

    # Data paths (optional, can be set via CLI)
    data_dir: Optional[str] = None
    label_file: Optional[str] = None  # external whole-model labels file

    # Encoder parameters for SurfaceEncoder CNN
    surface_encoder_channels: list = field(default_factory=lambda: [64, 128, 256])
    surface_encoder_output_dim: int = 128

    # Encoder parameters for CurveEncoder CNN
    curve_encoder_channels: list = field(default_factory=lambda: [64, 128])

    # Descriptor dimensions
    d2_descriptor_dim: int = 64  # D2 shape descriptor histogram bins
    angle_descriptor_dim: int = 64  # angle descriptor histogram bins

    # Convexity types for edge convexity encoding
    num_convexity_types: int = 3  # convex, concave, smooth

    # Face segmentation
    face_segmentation: bool = False  # enables face segmentation head
    face_seg_weight: float = 1.0  # loss weight for face segmentation
    model_cls_weight: float = 1.0  # loss weight for model classification
    num_face_classes: int = 27  # number of face-level classes
    face_seg_hidden_dim: int = 512  # hidden dim for face seg MLP
    face_seg_dropout: float = 0.3  # dropout for face seg MLP
    face_class_weights: Optional[list] = None  # inverse-frequency class weights for face seg loss

    # Memory optimization
    gradient_checkpointing: bool = True  # trade compute for memory in encoder layers

    # Rotary Position Embeddings - disabled by default because B-rep faces
    # have no natural sequential ordering; spatial_pos already encodes topology
    use_rope: bool = False
    max_seq_len: int = 512  # only used when use_rope=True

    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
