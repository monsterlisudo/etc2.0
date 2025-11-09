"""
模型测试

测试TrafficModel的各种功能
"""

import torch
import pytest
from src.traffic_classifier import TrafficModel


def test_model_initialization():
    """测试模型初始化"""
    model = TrafficModel(
        num_classes=5,
        time_dim=5,
        topo_dim=10,
        seq_dim=5,
        use_adv=True
    )
    
    assert model.num_classes == 5
    assert model.use_adv is True
    assert hasattr(model, 'time_encoder')
    assert hasattr(model, 'topo_encoder')
    assert hasattr(model, 'seq_encoder')
    assert hasattr(model, 'feature_selector')


def test_model_forward_pass():
    """测试前向传播"""
    model = TrafficModel(
        num_classes=3,
        time_dim=5,
        topo_dim=10,
        seq_dim=5,
        use_adv=False
    )
    
    # 创建示例输入
    batch_size = 2
    time_feat = torch.randn(batch_size, 10, 5)  # [batch, seq_len, time_dim]
    topo_feat = torch.randn(batch_size, 10)     # [batch, topo_dim]
    seq_feat = torch.randn(batch_size, 10, 5)   # [batch, seq_len, seq_dim]
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        # 训练模式
        logits, weights = model(time_feat, topo_feat, seq_feat, training=True)
        assert logits.shape == (batch_size, 3)  # num_classes = 3
        assert weights.shape == (batch_size, 3) # 3个特征类型的权重
        
        # 推理模式
        logits, weights = model(time_feat, topo_feat, seq_feat, training=False)
        assert logits.shape == (batch_size, 3)
        assert weights.shape == (batch_size, 3)


def test_feature_selector():
    """测试动态特征选择器"""
    from src.traffic_classifier.models import DynamicFeatureSelector
    
    selector = DynamicFeatureSelector(input_dim=128)
    
    # 创建示例状态
    batch_size = 4
    state = torch.randn(batch_size, 128)
    
    # 前向传播
    weights = selector(state)
    assert weights.shape == (batch_size, 3)  # 3个特征类型
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size))  # 权重和为1


def test_adversarial_generator():
    """测试对抗生成器"""
    from src.traffic_classifier.models import AdversarialGenerator
    
    # 测试序列特征
    gen_seq = AdversarialGenerator(feature_dim=5, is_sequential=True)
    seq_input = torch.randn(2, 10, 5)
    adversarial_seq = gen_seq(seq_input, epsilon=0.1)
    assert adversarial_seq.shape == seq_input.shape
    
    # 测试非序列特征
    gen_topo = AdversarialGenerator(feature_dim=10, is_sequential=False)
    topo_input = torch.randn(2, 10)
    adversarial_topo = gen_topo(topo_input, epsilon=0.1)
    assert adversarial_topo.shape == topo_input.shape


def test_multi_head_attention():
    """测试多头自注意力"""
    from src.traffic_classifier.models import MultiHeadSelfAttention
    
    attn = MultiHeadSelfAttention(embed_dim=8, num_heads=2)
    
    # 创建示例输入
    batch_size = 2
    seq_len = 5
    input_tensor = torch.randn(batch_size, seq_len, 8)
    
    # 前向传播
    output = attn(input_tensor)
    assert output.shape == input_tensor.shape


def test_cross_modal_attention():
    """测试跨模态注意力"""
    from src.traffic_classifier.models import CrossModalAttention
    
    cross_attn = CrossModalAttention(time_dim=5, topo_dim=10, seq_dim=8)
    
    # 创建示例输入
    time_feat = torch.randn(4, 5)
    topo_feat = torch.randn(4, 10)
    seq_feat = torch.randn(4, 8)
    
    # 前向传播
    output = cross_attn(time_feat, topo_feat, seq_feat)
    assert output.shape == (4, 64)  # 投影到64维


if __name__ == "__main__":
    print("运行模型测试...")
    
    test_model_initialization()
    print("✓ 模型初始化测试通过")
    
    test_model_forward_pass()
    print("✓ 前向传播测试通过")
    
    test_feature_selector()
    print("✓ 特征选择器测试通过")
    
    test_adversarial_generator()
    print("✓ 对抗生成器测试通过")
    
    test_multi_head_attention()
    print("✓ 多头自注意力测试通过")
    
    test_cross_modal_attention()
    print("✓ 跨模态注意力测试通过")
    
    print("\n所有测试通过! ✓")
