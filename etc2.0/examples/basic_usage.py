"""
基础使用示例

演示如何使用Traffic Fingerprinting项目进行：
1. 数据加载和预处理
2. 模型创建和训练
3. 模型测试和评估
4. 结果可视化
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入项目模块
from src.traffic_classifier import TrafficModel, TrafficDataset
from src.traffic_classifier.trainer import TrafficTrainer
from src.traffic_classifier.utils import (
    set_random_seeds, get_device_info, 
    plot_training_history, plot_confusion_matrix,
    print_classification_metrics
)


def example_basic_training():
    """基础训练示例"""
    print("=" * 60)
    print("基础训练示例")
    print("=" * 60)
    
    # 1. 设置环境
    set_random_seeds(42)
    device_info = get_device_info()
    
    # 2. 准备数据
    data_dir = "data/raw"  # 假设数据已经准备
    if not os.path.exists(data_dir):
        print("请先准备PCAP数据文件到 data/raw/ 目录下")
        print("数据结构应为:")
        print("data/raw/")
        print("├── class1/")
        print("│   ├── traffic1.pcap")
        print("│   └── traffic2.pcap")
        print("└── class2/")
        print("    └── traffic1.pcap")
        return
    
    # 3. 创建数据集
    print("\n创建数据集...")
    dataset = TrafficDataset(
        data_dir=data_dir,
        window_size=100
    )
    
    print(f"数据集信息:")
    print(f"  - 类别数: {len(dataset.classes)}")
    print(f"  - 类别列表: {dataset.classes}")
    print(f"  - 样本总数: {len(dataset)}")
    print(f"  - 类别分布: {dataset.class_counts}")
    
    # 4. 划分数据集
    from torch.utils.data import random_split
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, temp_dataset = random_split(dataset, [train_size, val_size + test_size])
    val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size])
    
    print(f"\n数据集划分:")
    print(f"  - 训练集: {len(train_dataset)} 样本")
    print(f"  - 验证集: {len(val_dataset)} 样本")
    print(f"  - 测试集: {len(test_dataset)} 样本")
    
    # 5. 创建模型
    print("\n创建模型...")
    model = TrafficModel(
        num_classes=len(dataset.classes),
        time_dim=5,
        topo_dim=10,
        seq_dim=5,
        use_adv=True
    )
    
    print(f"模型参数:")
    print(f"  - 总参数数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 6. 训练模型
    print("\n开始训练...")
    trainer = TrafficTrainer(model)
    
    # 使用较小的epoch数进行演示
    train_history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=16,  # 较小的batch size
        num_epochs=5,   # 较少的epoch数用于演示
        learning_rate=0.001,
        save_path="models/basic_model.pth"
    )
    
    # 7. 评估模型
    print("\n评估模型...")
    test_results = trainer.test(
        test_dataset,
        batch_size=16,
        model_path="models/basic_model.pth"
    )
    
    # 8. 打印结果
    print_classification_metrics(test_results, dataset.classes)
    
    # 9. 可视化结果
    print("\n生成可视化图表...")
    
    # 训练历史
    plot_training_history(
        train_history, 
        save_path="results/plots/training_history.png"
    )
    
    # 混淆矩阵
    plot_confusion_matrix(
        test_results['confusion_matrix'],
        dataset.classes,
        save_path="results/plots/confusion_matrix.png"
    )
    
    # 特征权重分析
    from src.traffic_classifier.utils import analyze_feature_weights
    if trainer.feature_weights_history:
        analyze_feature_weights(
            np.array(trainer.feature_weights_history),
            save_path="results/plots/feature_weights.png"
        )
    
    print("\n基础训练示例完成!")
    print("结果保存在:")
    print("  - 模型: models/basic_model.pth")
    print("  - 图表: results/plots/")


def example_custom_model():
    """自定义模型示例"""
    print("\n" + "=" * 60)
    print("自定义模型示例")
    print("=" * 60)
    
    # 创建自定义配置的模型
    model = TrafficModel(
        num_classes=5,          # 5个类别
        time_dim=10,            # 更多的时序特征
        topo_dim=15,            # 更多的拓扑特征
        seq_dim=8,              # 更多的序列特征
        use_adv=False           # 禁用对抗训练
    )
    
    # 创建训练器
    trainer = TrafficTrainer(model)
    
    print("自定义模型配置:")
    print(f"  - 类别数: {model.classifier[-1].out_features}")
    print(f"  - 对抗训练: {model.use_adv}")
    print(f"  - 总参数: {sum(p.numel() for p in model.parameters()):,}")


def example_data_analysis():
    """数据分析示例"""
    print("\n" + "=" * 60)
    print("数据分析示例")
    print("=" * 60)
    
    # 分析数据分布
    data_dir = "data/raw"
    if os.path.exists(data_dir):
        from src.traffic_classifier.utils import validate_pcap_files
        
        print("分析PCAP文件...")
        validation_results = validate_pcap_files(data_dir)
        
        # 统计每个类别的文件数量
        print("\n文件统计:")
        for class_name, stats in validation_results['class_stats'].items():
            print(f"  {class_name}: {stats['valid']}/{stats['total']} 有效文件")
        
        # 可视化数据分布
        if validation_results['class_stats']:
            import pandas as pd
            
            # 创建数据框
            data = []
            for class_name, stats in validation_results['class_stats'].items():
                data.append({
                    'Class': class_name,
                    'Valid_Files': stats['valid'],
                    'Invalid_Files': stats['invalid']
                })
            
            df = pd.DataFrame(data)
            
            # 绘制分布图
            plt.figure(figsize=(10, 6))
            plt.bar(df['Class'], df['Valid_Files'], alpha=0.7, label='有效文件')
            plt.bar(df['Class'], df['Invalid_Files'], bottom=df['Valid_Files'], alpha=0.7, label='无效文件')
            plt.title('数据文件分布')
            plt.xlabel('类别')
            plt.ylabel('文件数量')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('results/plots/data_distribution.png', dpi=300, bbox_inches='tight')
            print("数据分布图已保存到: results/plots/data_distribution.png")
    else:
        print("数据目录不存在，请先准备数据")


def example_inference():
    """推理示例"""
    print("\n" + "=" * 60)
    print("模型推理示例")
    print("=" * 60)
    
    # 加载已训练的模型
    model_path = "models/basic_model.pth"
    if not os.path.exists(model_path):
        print("请先训练模型或提供模型文件路径")
        return
    
    # 创建模型
    model = TrafficModel(num_classes=5)  # 需要知道类别数
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print("模型推理演示:")
    print("  - 模式: eval()")
    print("  - 设备: CPU")
    print("  - 输入: 时序、拓扑、序列特征")
    print("  - 输出: 分类logits + 特征权重")
    
    # 模拟推理（实际使用时需要真实数据）
    print("\n模拟推理过程:")
    with torch.no_grad():
        # 创建示例输入
        batch_size = 4
        time_feat = torch.randn(batch_size, 100, 5)    # [batch, seq_len, time_dim]
        topo_feat = torch.randn(batch_size, 10)        # [batch, topo_dim]
        seq_feat = torch.randn(batch_size, 100, 5)     # [batch, seq_len, seq_dim]
        
        # 前向传播
        logits, weights = model(time_feat, topo_feat, seq_feat, training=False)
        
        print(f"  - 输入形状: time={time_feat.shape}, topo={topo_feat.shape}, seq={seq_feat.shape}")
        print(f"  - 输出形状: logits={logits.shape}, weights={weights.shape}")
        print(f"  - 预测结果: {torch.softmax(logits, dim=-1)}")
        print(f"  - 特征权重: {weights}")


def main():
    """主函数"""
    print("Traffic Fingerprinting - 基础使用示例")
    print("=" * 60)
    
    # 创建必要的目录
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    
    # 运行示例
    try:
        # 基础训练示例
        example_basic_training()
        
        # 自定义模型示例
        example_custom_model()
        
        # 数据分析示例
        example_data_analysis()
        
        # 推理示例
        example_inference()
        
        print("\n" + "=" * 60)
        print("所有示例执行完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        print("请检查:")
        print("1. 是否已安装所有依赖包")
        print("2. 是否已准备PCAP数据文件")
        print("3. 数据目录结构是否正确")


if __name__ == "__main__":
    main()
