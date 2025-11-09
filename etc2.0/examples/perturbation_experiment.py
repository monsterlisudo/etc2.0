"""
扰动实验示例

演示如何使用扰动实验验证强化学习机制的有效性
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入项目模块
from src.traffic_classifier import TrafficModel, TrafficDataset
from src.traffic_classifier.trainer import TrafficTrainer
from src.traffic_classifier.utils import set_random_seeds, get_device_info


def run_perturbation_experiment():
    """运行完整的扰动实验"""
    print("=" * 70)
    print("扰动实验 - 验证强化学习机制有效性")
    print("=" * 70)
    
    # 1. 设置环境
    set_random_seeds(42)
    device_info = get_device_info()
    
    # 2. 检查数据
    data_dir = "data/raw"
    if not os.path.exists(data_dir):
        print("请先准备PCAP数据文件到 data/raw/ 目录下")
        return
    
    # 3. 创建模型
    print("\n创建模型...")
    model = TrafficModel(
        num_classes=10,  # 假设10个类别
        time_dim=5,
        topo_dim=10,
        seq_dim=5,
        use_adv=True
    )
    
    # 4. 创建训练器
    trainer = TrafficTrainer(model)
    
    # 5. 运行扰动实验
    print("\n开始扰动实验...")
    results = trainer.run_perturbation_experiment(
        data_dir=data_dir,
        batch_size=16,
        num_epochs=5,  # 较少的epoch用于演示
        learning_rate=0.001
    )
    
    # 6. 分析结果
    print("\n实验结果分析:")
    analyze_rl_effectiveness(results)
    
    print("\n扰动实验完成!")
    print("结果图表保存在: results/plots/perturbation_experiment_results.png")
    print("详细数据保存在: perturbation_experiment_details.csv")


def analyze_rl_effectiveness(results):
    """分析强化学习机制的有效性"""
    print("\n" + "=" * 50)
    print("强化学习机制有效性分析")
    print("=" * 50)
    
    # 获取基线权重（none扰动）
    if 'none' not in results:
        print("错误: 缺少基线实验结果")
        return
    
    baseline = results['none']['final_weights']
    print(f"基线权重分布: 时序={baseline[0]:.4f}, 拓扑={baseline[1]:.4f}, 序列={baseline[2]:.4f}")
    
    # 分析每种扰动的影响
    perturbation_names = {'time': '时序特征', 'topo': '拓扑特征', 'seq': '序列特征'}
    
    effectiveness_scores = {}
    
    for perturb_type, perturb_name in perturbation_names.items():
        if perturb_type in results:
            perturbed = results[perturb_type]['final_weights']
            
            # 计算被扰动特征的权重变化
            perturb_idx = {'time': 0, 'topo': 1, 'seq': 2}[perturb_type]
            weight_reduction = (baseline[perturb_idx] - perturbed[perturb_idx]) / baseline[perturb_idx] * 100
            
            # 计算其他特征权重的补偿变化
            compensation_changes = []
            for i, name in enumerate(perturbation_names.values()):
                if i != perturb_idx:
                    change = (perturbed[i] - baseline[i]) / baseline[i] * 100
                    compensation_changes.append(f"{name}: {change:+.2f}%")
            
            # 评估有效性
            is_effective = weight_reduction > 10  # 权重降低超过10%认为有效
            
            effectiveness_scores[perturb_type] = {
                'weight_reduction': weight_reduction,
                'is_effective': is_effective,
                'score': min(weight_reduction, 100) / 100  # 归一化到0-1
            }
            
            print(f"\n{perturb_name}扰动:")
            print(f"  权重降低: {weight_reduction:.2f}%")
            print(f"  其他特征补偿: {', '.join(compensation_changes)}")
            print(f"  机制有效性: {'✓ 有效' if is_effective else '✗ 无效'}")
    
    # 计算整体有效性分数
    if effectiveness_scores:
        avg_score = np.mean([score['score'] for score in effectiveness_scores.values()])
        print(f"\n整体有效性分数: {avg_score:.3f} / 1.000")
        
        if avg_score >= 0.5:
            print("✓ 强化学习机制整体有效!")
        else:
            print("✗ 强化学习机制效果有限")
    
    # 生成详细报告
    generate_experiment_report(results, effectiveness_scores)


def generate_experiment_report(results, effectiveness_scores):
    """生成实验报告"""
    report_path = "results/reports/perturbation_experiment_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 扰动实验报告\n\n")
        f.write("## 实验目的\n")
        f.write("验证动态特征选择器中的强化学习机制是否能够根据特征扰动自动调整权重分配。\n\n")
        
        f.write("## 实验设计\n")
        f.write("- 基线组：无扰动\n")
        f.write("- 实验组：分别对时序、拓扑、序列特征添加扰动\n")
        f.write("- 期望：被扰动特征的权重应降低，其他特征权重应补偿性增加\n\n")
        
        f.write("## 实验结果\n\n")
        
        # 结果表格
        f.write("### 权重变化分析\n\n")
        f.write("| 扰动类型 | 权重降低 (%) | 有效性 | 评价 |\n")
        f.write("|---------|-------------|--------|------|\n")
        
        for perturb_type, score in effectiveness_scores.items():
            perturb_name = {'time': '时序特征', 'topo': '拓扑特征', 'seq': '序列特征'}[perturb_type]
            f.write(f"| {perturb_name} | {score['weight_reduction']:.2f}% | ")
            f.write(f"{'✓' if score['is_effective'] else '✗'} | ")
            f.write(f"{'有效' if score['is_effective'] else '无效'} |\n")
        
        f.write("\n### 结论\n\n")
        avg_score = np.mean([score['score'] for score in effectiveness_scores.values()])
        if avg_score >= 0.5:
            f.write("强化学习机制整体有效，能够根据特征扰动自动调整权重分配。\n")
        else:
            f.write("强化学习机制效果有限，可能需要调整超参数或网络结构。\n")
        
        f.write("\n### 建议\n\n")
        if avg_score < 0.5:
            f.write("- 考虑增加训练轮数\n")
            f.write("- 调整学习率\n")
            f.write("- 改进特征选择器架构\n")
            f.write("- 增加数据量或数据质量\n")
        else:
            f.write("- 机制工作良好，可应用于实际场景\n")
            f.write("- 可进一步优化以提高鲁棒性\n")
    
    print(f"\n详细实验报告已保存到: {report_path}")


def visualize_perturbation_results(results):
    """可视化扰动实验结果"""
    setup_matplotlib_for_plotting()
    
    # 收集数据
    perturbation_types = ['none', 'time', 'topo', 'seq']
    feature_names = ['时序特征', '拓扑特征', '序列特征']
    
    data = []
    for perturb_type in perturbation_types:
        if perturb_type in results:
            weights = results[perturb_type]['final_weights']
            for i, feature_name in enumerate(feature_names):
                data.append({
                    '扰动类型': perturb_type.upper(),
                    '特征类型': feature_name,
                    '权重': weights[i]
                })
    
    if not data:
        print("没有可视化数据")
        return
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # 子图1: 权重对比
    plt.subplot(2, 2, 1)
    import pandas as pd
    df = pd.DataFrame(data)
    pivot_df = df.pivot(index='扰动类型', columns='特征类型', values='权重')
    pivot_df.plot(kind='bar', ax=plt.gca())
    plt.title('不同扰动下的特征权重对比')
    plt.ylabel('平均权重')
    plt.xticks(rotation=45)
    plt.legend(title='特征类型')
    
    # 子图2: 热力图
    plt.subplot(2, 2, 2)
    import seaborn as sns
    heatmap_data = pivot_df.values
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=feature_names, yticklabels=pivot_df.index)
    plt.title('特征权重热力图')
    
    # 子图3: 权重变化趋势
    plt.subplot(2, 2, 3)
    for perturb_type in perturbation_types:
        if perturb_type in results and results[perturb_type]['weight_history']:
            history = np.array(results[perturb_type]['weight_history'])
            plt.plot(history[:, 0], label=f'{perturb_type}-时序', linestyle='--')
            plt.plot(history[:, 1], label=f'{perturb_type}-拓扑', linestyle='-.')
            plt.plot(history[:, 2], label=f'{perturb_type}-序列', linestyle=':')
    
    plt.title('训练过程中特征权重变化')
    plt.xlabel('Epoch')
    plt.ylabel('权重')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 子图4: 有效性分析
    plt.subplot(2, 2, 4)
    if 'none' in results:
        baseline = results['none']['final_weights']
        impact_analysis = []
        perturbation_names = {'time': '时序特征', 'topo': '拓扑特征', 'seq': '序列特征'}
        
        for perturb_type in ['time', 'topo', 'seq']:
            if perturb_type in results:
                perturbed = results[perturb_type]['final_weights']
                perturb_idx = {'time': 0, 'topo': 1, 'seq': 2}[perturb_type]
                reduction = (baseline[perturb_idx] - perturbed[perturb_idx]) / baseline[perturb_idx] * 100
                impact_analysis.append({
                    '扰动类型': perturbation_names[perturb_type],
                    '权重降低百分比': reduction
                })
        
        if impact_analysis:
            impact_df = pd.DataFrame(impact_analysis)
            plt.bar(impact_df['扰动类型'], impact_df['权重降低百分比'])
            plt.title('扰动对特征权重的影响')
            plt.ylabel('权重降低百分比 (%)')
            plt.xticks(rotation=45)
            plt.axhline(y=10, color='r', linestyle='--', label='有效性阈值 (10%)')
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/plots/detailed_perturbation_analysis.png', dpi=300, bbox_inches='tight')
    print("详细分析图已保存到: results/plots/detailed_perturbation_analysis.png")


def setup_matplotlib_for_plotting():
    """设置matplotlib配置"""
    import warnings
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    warnings.filterwarnings('default')
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False


def main():
    """主函数"""
    print("Traffic Fingerprinting - 扰动实验示例")
    
    # 创建必要目录
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/reports", exist_ok=True)
    
    # 运行扰动实验
    run_perturbation_experiment()
    
    print("\n扰动实验示例完成!")


if __name__ == "__main__":
    main()
