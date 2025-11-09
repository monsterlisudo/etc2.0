"""
Traffic Fingerprinting Network - 流量指纹识别项目

这个包提供了一个基于深度学习的网络流量分类和指纹识别系统。
使用多模态融合和强化学习机制来提高分类准确性和鲁棒性。
"""

__version__ = "1.0.0"
__author__ = "MiniMax Agent"
__email__ = "support@minimax.ai"

from .models import (
    MultiHeadSelfAttention,
    CrossModalAttention, 
    AdversarialGenerator,
    DynamicFeatureSelector,
    TrafficModel
)

from .data_preprocessing import TrafficPreprocessor
from .dataset import TrafficDataset

__all__ = [
    'MultiHeadSelfAttention',
    'CrossModalAttention', 
    'AdversarialGenerator',
    'DynamicFeatureSelector',
    'TrafficModel',
    'TrafficPreprocessor',
    'TrafficDataset'
]