"""
工具函数模块

提供各种辅助功能：
- 可视化
- 评估指标
- 模型分析
- 数据处理工具
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple, Optional
import os
import json


def setup_matplotlib_for_plotting():
    """
    设置matplotlib和seaborn绘图配置
    在创建任何图表前调用此函数以确保正确渲染
    """
    import warnings
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 确保显示警告
    warnings.filterwarnings('default')

    # 配置matplotlib为非交互模式
    plt.switch_backend("Agg")

    # 设置图表样式
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # 配置跨平台兼容的字体
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_training_history(train_history: Dict, save_path: str = None):
    """绘制训练历史曲线"""
    setup_matplotlib_for_plotting()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 损失曲线
    axes[0].plot(train_history['train_loss'], label='训练损失', linewidth=2)
    axes[0].plot(train_history['val_loss'], label='验证损失', linewidth=2)
    axes[0].set_title('训练和验证损失变化', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失值')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(train_history['train_acc'], label='训练准确率', linewidth=2)
    axes[1].plot(train_history['val_acc'], label='验证准确率', linewidth=2)
    axes[1].set_title('训练和验证准确率变化', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存到: {save_path}")
    
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         save_path: str = None, title: str = "混淆矩阵"):
    """绘制混淆矩阵"""
    setup_matplotlib_for_plotting()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    
    return plt.gcf()


def analyze_feature_weights(weights_history: np.ndarray, 
                          feature_names: List[str] = None,
                          save_path: str = None):
    """分析特征权重变化"""
    setup_matplotlib_for_plotting()
    
    if feature_names is None:
        feature_names = ['时序特征', '拓扑特征', '序列特征']
    
    plt.figure(figsize=(12, 6))
    
    # 权重变化曲线
    for i, name in enumerate(feature_names):
        plt.plot(weights_history[:, i], label=name, linewidth=2, marker='o', markersize=3)
    
    plt.title('特征权重训练过程中变化趋势', fontsize=14, fontweight='bold')
    plt.xlabel('训练轮次 (Epoch)')
    plt.ylabel('权重值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加权重稳定性分析
    weight_stability = np.std(weights_history, axis=0)
    plt.text(0.02, 0.98, f'权重稳定性 (标准差):\n' + 
             '\n'.join([f'{name}: {stability:.4f}' for name, stability in zip(feature_names, weight_stability)]),
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征权重分析图已保存到: {save_path}")
    
    return plt.gcf()


def print_classification_metrics(test_results: Dict, class_names: List[str] = None):
    """打印分类指标"""
    report = test_results['classification_report']
    
    print("\n" + "="*60)
    print("分类性能报告")
    print("="*60)
    
    # 总体指标
    print(f"测试损失: {test_results['test_loss']:.4f}")
    print(f"测试准确率: {test_results['test_accuracy']:.4f}")
    print()
    
    # 各类别指标
    if class_names:
        print("各类别详细指标:")
        print("-" * 80)
        print(f"{'类别':<15} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'支持数':<10}")
        print("-" * 80)
        
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                print(f"{class_name:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                      f"{metrics['f1-score']:<10.3f} {int(metrics['support']):<10}")
    
    # 宏平均和加权平均
    print("-" * 80)
    print(f"{'宏平均':<15} {report['macro avg']['precision']:<10.3f} {report['macro avg']['recall']:<10.3f} "
          f"{report['macro avg']['f1-score']:<10.3f} {int(report['macro avg']['support']):<10}")
    print(f"{'加权平均':<15} {report['weighted avg']['precision']:<10.3f} {report['weighted avg']['recall']:<10.3f} "
          f"{report['weighted avg']['f1-score']:<10.3f} {int(report['weighted avg']['support']):<10}")


def calculate_model_complexity(model, input_shape: Tuple = None):
    """计算模型复杂度指标"""
    if input_shape is None:
        # 假设输入形状
        input_shape = (32, 100, 5)  # (batch, seq_len, features)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算模型大小 (MB)
    model_size = total_params * 4 / (1024 * 1024)  # 假设float32，4 bytes per param
    
    # 估算计算量 (FLOPs)
    dummy_input = torch.randn(input_shape)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    
    try:
        with torch.no_grad():
            # 使用thop库可以提供更精确的FLOPs计算
            # 这里提供一个简化的估算
            from thop import profile
            flops, params = profile(model, inputs=(dummy_input, dummy_input, dummy_input))
            flops = flops / 1e9  # 转换为GFLOPs
        thop_available = True
    except ImportError:
        flops = "N/A (需要安装thop库)"
        thop_available = False
    
    complexity_info = {
        '总参数数量': total_params,
        '可训练参数数量': trainable_params,
        '模型大小 (MB)': model_size,
        '计算量 (GFLOPs)': flops,
        'thop库可用': thop_available
    }
    
    print("\n" + "="*40)
    print("模型复杂度信息")
    print("="*40)
    for key, value in complexity_info.items():
        print(f"{key}: {value}")
    print("="*40)
    
    return complexity_info


def save_experiment_config(config: Dict, save_path: str):
    """保存实验配置"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"实验配置已保存到: {save_path}")


def load_experiment_config(config_path: str) -> Dict:
    """加载实验配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def validate_pcap_files(data_dir: str) -> Dict:
    """验证PCAP文件"""
    results = {
        'total_classes': 0,
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'class_stats': {},
        'errors': []
    }
    
    if not os.path.exists(data_dir):
        results['errors'].append(f"数据目录不存在: {data_dir}")
        return results
    
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    results['total_classes'] = len(classes)
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        pcap_files = [f for f in os.listdir(class_dir) if f.endswith('.pcap')]
        results['total_files'] += len(pcap_files)
        
        class_stats = {'total': len(pcap_files), 'valid': 0, 'invalid': 0}
        
        for pcap_file in pcap_files:
            pcap_path = os.path.join(class_dir, pcap_file)
            try:
                # 尝试读取文件
                from scapy.all import rdpcap
                packets = rdpcap(pcap_path)
                if len(packets) > 0:
                    class_stats['valid'] += 1
                else:
                    class_stats['invalid'] += 1
            except Exception as e:
                class_stats['invalid'] += 1
                results['errors'].append(f"{pcap_path}: {str(e)}")
        
        results['valid_files'] += class_stats['valid']
        results['invalid_files'] += class_stats['invalid']
        results['class_stats'][class_name] = class_stats
    
    # 打印验证结果
    print("\n" + "="*50)
    print("PCAP文件验证结果")
    print("="*50)
    print(f"类别总数: {results['total_classes']}")
    print(f"文件总数: {results['total_files']}")
    print(f"有效文件: {results['valid_files']}")
    print(f"无效文件: {results['invalid_files']}")
    
    if results['class_stats']:
        print("\n各类别文件统计:")
        print("-" * 30)
        for class_name, stats in results['class_stats'].items():
            print(f"{class_name}: {stats['valid']}/{stats['total']} 有效")
    
    if results['errors']:
        print(f"\n发现 {len(results['errors'])} 个错误:")
        for error in results['errors'][:10]:  # 只显示前10个错误
            print(f"  - {error}")
        if len(results['errors']) > 10:
            print(f"  ... 还有 {len(results['errors']) - 10} 个错误")
    
    print("="*50)
    
    return results


def set_random_seeds(seed: int = 42):
    """设置随机种子确保结果可复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"随机种子已设置为: {seed}")


def get_device_info():
    """获取设备信息"""
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else 'CPU',
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A'
    }
    
    print("\n" + "="*40)
    print("设备信息")
    print("="*40)
    for key, value in device_info.items():
        print(f"{key}: {value}")
    print("="*40)
    
    return device_info
