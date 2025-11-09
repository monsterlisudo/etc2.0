"""
主程序入口

提供命令行接口和Python API两种使用方式
支持正常训练、扰动实验等功能
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path

# 导入项目模块
from src.traffic_classifier import TrafficModel, TrafficDataset
from src.traffic_classifier.trainer import TrafficTrainer
from src.traffic_classifier.utils import (
    set_random_seeds, get_device_info, validate_pcap_files,
    plot_training_history, plot_confusion_matrix, print_classification_metrics
)


def setup_experiment_environment(args):
    """设置实验环境"""
    # 设置随机种子
    set_random_seeds(args.seed)
    
    # 获取设备信息
    device_info = get_device_info()
    
    # 创建结果目录
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n实验配置:")
    print(f"数据目录: {args.data_dir}")
    print(f"结果目录: {args.results_dir}")
    print(f"模型保存路径: {args.model_save_path}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"窗口大小: {args.window_size}")
    print(f"强化学习: {'禁用' if args.no_rl else '启用'}")
    print(f"对抗训练: {'禁用' if args.no_adv else '启用'}")
    
    return device_info


def validate_data(args):
    """验证数据"""
    print(f"\n验证数据...")
    validation_results = validate_pcap_files(args.data_dir)
    
    if validation_results['valid_files'] == 0:
        print("错误: 没有找到有效的PCAP文件!")
        return False
    
    if validation_results['errors']:
        print(f"警告: 发现 {len(validation_results['errors'])} 个错误")
        if len(validation_results['errors']) > 10:
            print("前10个错误示例:")
            for error in validation_results['errors'][:10]:
                print(f"  - {error}")
    
    return True


def create_model_and_trainer(args, dataset):
    """创建模型和训练器（支持新的强化学习架构）"""
    # 创建模型（使用论文中的参数设置）
    model = TrafficModel(
        num_classes=len(dataset.classes),
        time_dim=5,      # 时间特征维度
        topo_dim=10,     # 拓扑特征维度
        seq_dim=5,       # 序列特征维度
        seq_len=args.window_size,  # 序列长度
        use_adv=not args.no_adv,   # 启用对抗训练
        use_rl=not args.no_rl      # 启用强化学习
    )
    
    # 创建训练器（支持RL和对抗训练）
    trainer = TrafficTrainer(
        model, 
        use_rl=not args.no_rl,
        use_adv=not args.no_adv
    )
    
    return trainer


def run_normal_training(args):
    """正常运行模式"""
    print("\n" + "="*60)
    print("开始正常训练模式")
    print("="*60)
    
    # 设置环境
    device_info = setup_experiment_environment(args)
    
    # 验证数据
    if not validate_data(args):
        return
    
    # 创建数据集
    print(f"\n创建数据集...")
    dataset = TrafficDataset(args.data_dir, window_size=args.window_size)
    
    if len(dataset) == 0:
        print("错误: 数据集为空!")
        return
    
    # 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, temp_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size + test_size])
    val_dataset, test_dataset = torch.utils.data.random_split(
        temp_dataset, [val_size, test_size])
    
    print(f"数据集划分: 训练 {len(train_dataset)}, 验证 {len(val_dataset)}, 测试 {len(test_dataset)}")
    
    # 创建模型和训练器
    trainer = create_model_and_trainer(args, dataset)
    
    # 训练模型
    train_history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        save_path=args.model_save_path
    )
    
    # 绘制训练历史
    plot_training_history(train_history, save_path=os.path.join(args.results_dir, 'training_history.png'))
    
    # 测试模型
    print(f"\n开始测试...")
    test_results = trainer.test(test_dataset, batch_size=args.batch_size, model_path=args.model_save_path)
    
    # 打印测试结果
    print_classification_metrics(test_results, dataset.classes)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        test_results['confusion_matrix'], 
        dataset.classes,
        save_path=os.path.join(args.results_dir, 'confusion_matrix.png')
    )
    
    # 分析特征权重
    from src.traffic_classifier.utils import analyze_feature_weights
    analyze_feature_weights(
        np.array(trainer.feature_weights_history),
        save_path=os.path.join(args.results_dir, 'feature_weights.png')
    )
    
    print(f"\n实验完成! 结果保存在: {args.results_dir}")


def run_perturbation_experiment(args):
    """运行扰动实验"""
    print("\n" + "="*60)
    print("开始扰动实验模式")
    print("="*60)
    
    # 设置环境
    device_info = setup_experiment_environment(args)
    
    # 验证数据
    if not validate_data(args):
        return
    
    # 创建模型和训练器
    dataset = TrafficDataset(args.data_dir, window_size=args.window_size)
    trainer = create_model_and_trainer(args, dataset)
    
    # 运行扰动实验
    print(f"\n运行扰动实验...")
    results = trainer.run_perturbation_experiment(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    print(f"\n扰动实验完成! 结果保存在: {args.results_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='流量指纹识别模型训练和评估',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 正常训练
  python main.py --data_dir /path/to/data --mode train
  
  # 扰动实验
  python main.py --data_dir /path/to/data --mode perturb
  
  # 自定义参数
  python main.py --data_dir /path/to/data --epochs 100 --batch_size 64 --lr 0.0001
        """
    )
    
    # 基础参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='包含数据集的目录')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'perturb'],
                       help='运行模式: train(正常训练) 或 perturb(扰动实验)')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='结果保存目录')
    parser.add_argument('--model_save_path', type=str, default='best_model.pth',
                       help='模型保存路径')
    
    # 训练参数（使用论文中的设置）
    parser.add_argument('--batch_size', type=int, default=128,
                       help='训练批次大小（论文默认: 128）')
    parser.add_argument('--epochs', type=int, default=25,
                       help='训练轮数（论文默认: 25）')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='学习率（论文默认: 1e-3）')
    parser.add_argument('--window_size', type=int, default=100,
                       help='时间窗口大小（论文默认: 100）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    # 模型参数
    parser.add_argument('--no_adv', action='store_true',
                       help='禁用对抗训练')
    parser.add_argument('--no_rl', action='store_true', 
                       help='禁用强化学习特征选择')
    
    args = parser.parse_args()
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 显示设备信息
    get_device_info()
    
    # 根据模式运行
    if args.mode == 'train':
        run_normal_training(args)
    elif args.mode == 'perturb':
        run_perturbation_experiment(args)
    else:
        print(f"错误: 未知的运行模式 {args.mode}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
