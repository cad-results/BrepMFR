"""
BrepFormer: Transformer-Based B-rep Geometric Feature Recognition

Implementation based on the paper "BrepFormer: Transformer-Based B-rep Geometric Feature Recognition"
(arXiv:2504.07378) and official implementation (https://github.com/dyk-npu/BRepFormer)
"""

from brepformer.configs.config import BrepClassifierConfig
from brepformer.models.brep_classifier import BrepClassifier

__version__ = "0.1.0"
__all__ = ["BrepClassifierConfig", "BrepClassifier"]
