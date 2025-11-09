"""
神经网络模型模块

包含多模态流量分类的所有神经网络组件：
- 多头自注意力机制
- 跨模态注意力机制  
- 对抗生成器
- 动态特征选择器
- 完整的多模态融合模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制
    
    用于处理时序特征，支持多种头数配置。
    
    Args:
        embed_dim (int): 嵌入维度
        num_heads (int): 注意力头数，默认为4
    """
    
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # 计算Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, embed_dim)
        K = self.key(x)  # (batch_size, seq_len, embed_dim)
        V = self.value(x)  # (batch_size, seq_len, embed_dim)

        # 分割多头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # 应用注意力权重
        attended = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)

        # 合并多头
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # 输出投影
        output = self.out(attended)

        return output


class CrossModalAttention(nn.Module):
    """跨模态注意力机制
    
    用于融合时序、拓扑和序列特征
    
    Args:
        time_dim (int): 时序特征维度
        topo_dim (int): 拓扑特征维度  
        seq_dim (int): 序列特征维度
    """

    def __init__(self, time_dim, topo_dim, seq_dim):
        super().__init__()
        self.time_dim = time_dim
        self.topo_dim = topo_dim
        self.seq_dim = seq_dim

        # 时间特征作为查询，其他特征作为键和值
        self.time_to_topo = nn.Linear(time_dim, topo_dim)
        self.time_to_seq = nn.Linear(time_dim, seq_dim)

        self.topo_proj = nn.Linear(topo_dim, 64)
        self.seq_proj = nn.Linear(seq_dim, 64)

        self.out_proj = nn.Linear(64, 64)

    def forward(self, time_feat, topo_feat, seq_feat):
        # 时间特征作为查询
        q_time = time_feat.unsqueeze(1)  # (batch_size, 1, time_dim)

        # 拓扑特征作为键和值
        k_topo = topo_feat.unsqueeze(1)  # (batch_size, 1, topo_dim)
        v_topo = topo_feat.unsqueeze(1)  # (batch_size, 1, topo_dim)

        # 序列特征作为键和值
        k_seq = seq_feat.unsqueeze(1)  # (batch_size, 1, seq_dim)
        v_seq = seq_feat.unsqueeze(1)  # (batch_size, 1, seq_dim)

        # 计算时间-拓扑注意力
        q_time_to_topo = self.time_to_topo(q_time)  # (batch_size, 1, topo_dim)
        topo_attn_scores = torch.matmul(q_time_to_topo, k_topo.transpose(-2, -1)) / (self.topo_dim ** 0.5)
        topo_attn_weights = F.softmax(topo_attn_scores, dim=-1)
        topo_attended = torch.matmul(topo_attn_weights, v_topo)  # (batch_size, 1, topo_dim)
        topo_attended = self.topo_proj(topo_attended.squeeze(1))  # (batch_size, 64)

        # 计算时间-序列注意力
        q_time_to_seq = self.time_to_seq(q_time)  # (batch_size, 1, seq_dim)
        seq_attn_scores = torch.matmul(q_time_to_seq, k_seq.transpose(-2, -1)) / (self.seq_dim ** 0.5)
        seq_attn_weights = F.softmax(seq_attn_scores, dim=-1)
        seq_attended = torch.matmul(seq_attn_weights, v_seq)  # (batch_size, 1, seq_dim)
        seq_attended = self.seq_proj(seq_attended.squeeze(1))  # (batch_size, 64)

        # 融合两种注意力结果
        cross_modal = topo_attended + seq_attended
        cross_modal = self.out_proj(cross_modal)

        return cross_modal


class TimeAdversary(nn.Module):
    """时间特征对抗生成器：扰动时间模式（轻量网络）"""
    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.LayerNorm(64),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)


class TopologyAdversary(nn.Module):
    """拓扑特征对抗生成器：保持网络结构，调整结构表示（轻量网络）"""
    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.LayerNorm(64),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)


class SequenceAdversary(nn.Module):
    """序列特征对抗生成器：扰动协议序列（深度网络捕捉时序相关性）"""
    def __init__(self, input_dim: int = 128, seq_len: int = 100):
        super().__init__()
        self.seq_len = seq_len
        
        self.generator = nn.Sequential(
            nn.Linear(input_dim * seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.LayerNorm(64),
            nn.Linear(64, input_dim * seq_len)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # [batch, input_dim * seq_len]
        perturbation_flat = self.generator(x_flat)  # [batch, input_dim * seq_len]
        perturbation = perturbation_flat.view(batch_size, self.seq_len, -1)  # [batch, seq_len, input_dim]
        return perturbation


class AdversarialGenerator(nn.Module):
    """轻量级对抗训练生成器（模态特定）"""
    def __init__(self, modality: str, feature_dim: int = 128, seq_len: int = 100):
        super().__init__()
        self.modality = modality
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        
        if modality == 'time':
            self.generator = TimeAdversary(feature_dim)
        elif modality == 'topo':
            self.generator = TopologyAdversary(feature_dim)
        elif modality == 'seq':
            self.generator = SequenceAdversary(feature_dim, seq_len)
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    def forward(self, x: torch.Tensor, epsilon: float = 0.05) -> torch.Tensor:
        """生成对抗样本
        
        Args:
            x: 输入特征
            epsilon: 扰动幅度控制因子
            
        Returns:
            adversarial_features: 对抗样本
        """
        # 生成扰动
        perturbation = self.generator(x)
        
        # Tanh归一化 + 自适应幅度控制
        perturbation = torch.tanh(perturbation) * epsilon
        
        # 应用扰动
        adversarial_features = x + perturbation
        
        return adversarial_features


class TimeFeatureEncoder(nn.Module):
    """时间特征编码器：带自注意力的双向LSTM"""
    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 双向LSTM
        self.bilstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True,
            dropout=0.1
        )
        
        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # 双向LSTM输出是2*hidden_dim
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.bilstm(x)  # [batch, seq_len, 2*hidden_dim]
        
        # 自注意力
        attn_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(lstm_out + attn_out)  # 残差连接
        
        # 全局池化到固定维度
        batch_size = attn_out.size(0)
        pooled = self.global_pool(attn_out.transpose(1, 2)).squeeze(-1)  # [batch, 2*hidden_dim]
        
        return pooled  # 128维输出


class TopologyFeatureEncoder(nn.Module):
    """拓扑特征编码器：多层MLP"""
    def __init__(self, input_dim: int = 10, hidden_dims: List[int] = [64, 128]):
        super().__init__()
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            in_dim = hidden_dim
            
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, input_dim]
        return self.mlp(x)  # [batch, 128]


class SequenceFeatureEncoder(nn.Module):
    """序列特征编码器：Transformer编码器"""
    def __init__(self, input_dim: int = 5, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(100, embed_dim))  # max_seq_len=100
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        batch_size, seq_len, _ = x.size()
        
        # 输入投影
        x = self.input_projection(x)  # [batch, seq_len, embed_dim]
        
        # 添加位置编码
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer编码
        x = self.transformer(x)  # [batch, seq_len, embed_dim]
        x = self.layer_norm(x)
        
        # 全局池化
        pooled = self.global_pool(x.transpose(1, 2)).squeeze(-1)  # [batch, embed_dim]
        
        return pooled  # 128维输出


class DQNFeatureSelector(nn.Module):
    """基于DQN的动态特征选择器"""
    def __init__(self, state_dim: int = 384, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # DQN网络：状态->动作值
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 权重生成网络
        self.weight_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 奖励历史（用于RL训练）
        self.reward_history = []
        
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        # state: [batch, 384] (拼接的3*128维特征)
        
        # 计算Q值（用于动作价值估计）
        q_values = self.q_network(state)
        
        # 计算特征权重（动作）
        raw_weights = self.weight_network(state)
        weights = F.softmax(raw_weights, dim=-1)  # [batch, 3]
        
        return {
            'weights': weights,
            'q_values': q_values,
            'raw_weights': raw_weights
        }
    
    def calculate_reward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                        weights: torch.Tensor, alpha: float = 1.0, beta: float = 0.1, 
                        gamma: float = 0.01, threshold: float = 0.5) -> torch.Tensor:
        """计算强化学习奖励函数"""
        batch_size = predictions.size(0)
        
        # 分类准确率奖励
        correct = (predictions.argmax(dim=1) == targets).float()
        accuracy_reward = torch.where(correct == 1, torch.tensor(1.0), torch.tensor(-1.0))
        
        # 特征多样性奖励（权重方差）
        mean_weight = weights.mean(dim=1, keepdim=True)  # [batch, 1]
        diversity_reward = torch.var(weights, dim=1)  # [batch]
        
        # 计算开销惩罚
        cost_penalty = (weights > threshold).sum(dim=1).float()  # [batch]
        
        # 总奖励
        total_reward = alpha * accuracy_reward + beta * diversity_reward - gamma * cost_penalty
        
        return total_reward  # [batch]


class DynamicFeatureSelector(nn.Module):
    """动态特征选择器（重写为基于强化学习的版本）"""
    def __init__(self, state_dim: int = 384, action_dim: int = 3):
        super().__init__()
        self.dqn_selector = DQNFeatureSelector(state_dim, action_dim)
        self.weight_history = []
        
    def forward(self, time_features: torch.Tensor, 
                topo_features: torch.Tensor, 
                seq_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 拼接所有特征作为状态
        batch_size = time_features.size(0)
        state = torch.cat([
            time_features.view(batch_size, -1),
            topo_features.view(batch_size, -1), 
            seq_features.view(batch_size, -1)
        ], dim=-1)  # [batch, 384]
        
        # DQN特征选择
        rl_output = self.dqn_selector(state)
        weights = rl_output['weights']
        
        # 保存权重历史
        if self.training:
            self.weight_history.append(weights.detach().cpu())
            if len(self.weight_history) > 1000:
                self.weight_history.pop(0)
        
        # 加权特征
        weighted_time = weights[:, 0:1] * time_features
        weighted_topo = weights[:, 1:2] * topo_features
        weighted_seq = weights[:, 2:3] * seq_features
        
        # 融合特征
        fused_output = weighted_time + weighted_topo + weighted_seq
        
        return {
            'fused_features': fused_output,
            'weights': weights,
            'state': state,
            'q_values': rl_output['q_values'],
            'weighted_features': {
                'time': weighted_time,
                'topo': weighted_topo,
                'seq': weighted_seq
            }
        }
        
    def get_weight_history(self):
        """获取权重历史并清空"""
        if not self.weight_history:
            return np.array([])
        history = torch.stack(self.weight_history).numpy()
        self.weight_history = []
        return history


class TrafficModel(nn.Module):
    """基于强化学习和轻量级对抗训练的加密流量分类模型
    
    集成了多模态编码器、强化学习特征选择和轻量级对抗训练
    
    Args:
        num_classes (int): 分类类别数
        time_dim (int): 时间特征维度（默认5维）
        topo_dim (int): 拓扑特征维度（默认10维） 
        seq_dim (int): 序列特征维度（默认5维）
        seq_len (int): 序列长度（默认100）
        use_adv (bool): 是否启用对抗训练
        use_rl (bool): 是否启用强化学习特征选择
    """

    def __init__(self, num_classes: int, time_dim: int = 5, topo_dim: int = 10, 
                 seq_dim: int = 5, seq_len: int = 100, use_adv: bool = True, 
                 use_rl: bool = True):
        super().__init__()
        self.use_adv = use_adv
        self.use_rl = use_rl
        self.seq_len = seq_len

        # 多模态特征编码器
        self.time_encoder = TimeFeatureEncoder(input_dim=time_dim, hidden_dim=64)  # 128维输出
        self.topo_encoder = TopologyFeatureEncoder(input_dim=topo_dim)  # 128维输出
        self.seq_encoder = SequenceFeatureEncoder(input_dim=seq_dim, embed_dim=64)  # 64维->128维

        # 轻量级对抗生成器（模态特定）
        if self.use_adv:
            self.time_adversary = AdversarialGenerator(modality='time', feature_dim=64, seq_len=seq_len)
            self.topo_adversary = AdversarialGenerator(modality='topo', feature_dim=64)
            self.seq_adversary = AdversarialGenerator(modality='seq', feature_dim=64, seq_len=seq_len)

        # 动态特征选择器（基于DQN的强化学习）
        if self.use_rl:
            self.feature_selector = DynamicFeatureSelector(state_dim=64*3, action_dim=3)  # 192维状态空间
        else:
            # 简单的固定权重融合
            self.fixed_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

        # 轻量级分类器（MLP + ReLU + LayerNorm + Softmax）
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),  # 输入融合特征
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, num_classes)
        )

    def forward(self, time_feat: torch.Tensor, topo_feat: torch.Tensor, 
                seq_feat: torch.Tensor, apply_perturb: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            time_feat: 时间特征 [batch, seq_len, 5]
            topo_feat: 拓扑特征 [batch, 10] 
            seq_feat: 序列特征 [batch, seq_len, 5]
            apply_perturb: 是否应用对抗扰动
            
        Returns:
            Dict包含分类结果、权重、特征等
        """
        batch_size = time_feat.size(0)
        
        # 轻量对抗训练扰动
        if self.use_adv and (apply_perturb or (self.training and torch.rand(1).item() < 0.3)):
            # 受控扰动机制：以概率p=0.3施加扰动
            epsilon = 0.05  # 初始扰动强度
            time_feat_adv = self.time_adversary(time_feat, epsilon)
            topo_feat_adv = self.topo_adversary(topo_feat, epsilon) 
            seq_feat_adv = self.seq_adversary(seq_feat, epsilon)
        else:
            time_feat_adv = time_feat
            topo_feat_adv = topo_feat
            seq_feat_adv = seq_feat

        # 多模态特征编码
        h_time = self.time_encoder(time_feat_adv)    # [batch, 128]
        h_topo = self.topo_encoder(topo_feat_adv)    # [batch, 128] 
        h_seq = self.seq_encoder(seq_feat_adv)       # [batch, 128]

        # 动态特征选择（基于强化学习）
        if self.use_rl:
            selector_output = self.feature_selector(h_time, h_topo, h_seq)
            weights = selector_output['weights']
            fused_features = selector_output['fused_features']
            weighted_features = selector_output['weighted_features']
        else:
            # 固定权重融合
            weights = F.softmax(self.fixed_weights, dim=0).unsqueeze(0).expand(batch_size, -1)
            weighted_time = weights[:, 0:1] * h_time
            weighted_topo = weights[:, 1:2] * h_topo
            weighted_seq = weights[:, 2:3] * h_seq
            fused_features = weighted_time + weighted_topo + weighted_seq
            weighted_features = {
                'time': weighted_time,
                'topo': weighted_topo,
                'seq': weighted_seq
            }

        # 分类
        logits = self.classifier(fused_features)  # [batch, num_classes]
        
        return {
            'logits': logits,
            'weights': weights,
            'fused_features': fused_features,
            'time_features': h_time,
            'topo_features': h_topo,
            'seq_features': h_seq,
            'weighted_features': weighted_features,
            'adversarial_applied': self.use_adv and (apply_perturb or (self.training and torch.rand(1).item() < 0.3))
        }

    def get_feature_weights_history(self):
        """获取特征选择器的权重历史"""
        if hasattr(self.feature_selector, 'get_weight_history'):
            return self.feature_selector.get_weight_history()
        else:
            return np.array([])
    
    def calculate_rl_reward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                          weights: torch.Tensor) -> torch.Tensor:
        """计算强化学习奖励函数"""
        if hasattr(self.feature_selector, 'dqn_selector'):
            return self.feature_selector.dqn_selector.calculate_reward(
                predictions, targets, weights)
        else:
            # 如果没有使用RL，返回0奖励
            return torch.zeros(predictions.size(0), device=predictions.device)
    
    def get_perturbation_probability(self) -> float:
        """获取当前扰动概率（用于分析）"""
        return 0.3 if self.use_adv else 0.0
