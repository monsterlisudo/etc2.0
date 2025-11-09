"""
训练器模块

负责模型训练、验证和测试，包括：
- 正常训练流程
- 扰动实验
- 早停机制
- 特征权重分析
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
import os
from typing import Dict, List, Tuple, Optional

from .dataset import TrafficDataset


class TrafficTrainer:
    """流量分类模型训练器"""
    
    def __init__(self, model, device='auto', use_rl=True, use_adv=True):
        self.model = model
        self.device = device if device != 'auto' else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 强化学习和对抗训练配置
        self.use_rl = use_rl
        self.use_adv = use_adv
        self.exploration_rate = 0.1  # 探索率
        self.epsilon = 0.05  # 对抗扰动幅度
        self.epsilon_decay = 0.995  # 探索率衰减
        self.min_epsilon = 0.01  # 最小探索率
        
        # 奖励函数参数（论文中的设置）
        self.alpha = 1.0  # 准确率奖励权重
        self.beta = 0.1   # 多样性奖励权重
        self.gamma = 0.01 # 计算开销惩罚权重
        self.threshold = 0.5  # 权重阈值
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'rl_rewards': [],  # 强化学习奖励
            'adversarial_acc': [],  # 对抗样本准确率
            'robustness_scores': [],  # 鲁棒性分数
            'exploration_rates': []  # 探索率变化
        }
        
        # 特征权重历史
        self.feature_weights_history = []
    
    def train_epoch(self, train_loader, criterion, optimizer, epoch: int, num_epochs: int) -> Tuple[float, float, float]:
        """训练一个epoch（支持强化学习和对抗训练）"""
        self.model.train()
        total_loss = 0.0
        total_rl_reward = 0.0
        correct = 0
        total = 0
        epoch_weights = []
        
        # 更新探索率（论文中的指数衰减）
        if self.use_rl and self.use_adv:
            self.exploration_rate = max(self.min_epsilon, 
                                      self.exploration_rate * (self.epsilon_decay ** epoch))
        
        for batch_idx, batch in enumerate(train_loader):
            time_feat = batch['time_feat'].to(self.device)
            topo_feat = batch['topo_feat'].to(self.device)
            seq_feat = batch['seq_feat'].to(self.device)
            labels = batch['label'].to(self.device)

            optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(time_feat, topo_feat, seq_feat, apply_perturb=False)
            logits = outputs['logits']
            weights = outputs['weights']
            
            # 计算分类损失
            classification_loss = criterion(logits, labels)
            
            # 计算强化学习奖励
            rl_reward = 0.0
            if self.use_rl:
                rl_reward = self.model.calculate_rl_reward(logits, labels, weights)
                total_rl_reward += rl_reward.mean().item()
            
            # 总损失（分类损失 + 对抗损失）
            total_loss_batch = classification_loss
            
            # 强化学习损失（如果启用）
            if self.use_rl and self.use_adv:
                # 探索阶段：在训练过程中以一定概率使用对抗样本
                if torch.rand(1).item() < 0.3:  # 论文中的p=0.3
                    adversarial_outputs = self.model(time_feat, topo_feat, seq_feat, apply_perturb=True)
                    adv_logits = adversarial_outputs['logits']
                    adv_loss = criterion(adv_logits, labels)
                    total_loss_batch += 0.1 * adv_loss  # 论文中的lambda权重
            
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += total_loss_batch.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 记录特征权重
            epoch_weights.append(weights.detach().cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        avg_rl_reward = total_rl_reward / len(train_loader) if self.use_rl else 0.0
        
        # 记录平均特征权重
        if epoch_weights:
            avg_weights = np.mean(np.vstack(epoch_weights), axis=0)
            self.feature_weights_history.append(avg_weights)
        
        return avg_loss, accuracy, avg_rl_reward
    
    def validate_epoch(self, val_loader, criterion) -> Tuple[float, float, Dict]:
        """验证一个epoch（支持对抗样本评估）"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 对抗样本测试
        adversarial_correct = 0
        all_predictions = []
        all_labels = []
        all_weights = []
        
        with torch.no_grad():
            for batch in val_loader:
                time_feat = batch['time_feat'].to(self.device)
                topo_feat = batch['topo_feat'].to(self.device)
                seq_feat = batch['seq_feat'].to(self.device)
                labels = batch['label'].to(self.device)

                # 标准测试
                outputs = self.model(time_feat, topo_feat, seq_feat, apply_perturb=False)
                logits = outputs['logits']
                loss = criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_weights.extend(outputs['weights'].cpu().numpy())
                
                # 对抗样本测试
                if self.use_adv:
                    adv_outputs = self.model(time_feat, topo_feat, seq_feat, apply_perturb=True)
                    adv_logits = adv_outputs['logits']
                    _, adv_predicted = torch.max(adv_logits.data, 1)
                    adversarial_correct += (adv_predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        adversarial_accuracy = adversarial_correct / total if self.use_adv else 0.0
        
        # 计算鲁棒性分数（论文定义）
        robustness_score = (adversarial_accuracy / accuracy) * 100 if accuracy > 0 and self.use_adv else 100.0
        
        validation_info = {
            'accuracy': accuracy,
            'adversarial_accuracy': adversarial_accuracy,
            'robustness_score': robustness_score,
            'feature_weights': np.mean(np.vstack(all_weights), axis=0),
            'predictions': all_predictions,
            'labels': all_labels
        }
        
        return avg_loss, accuracy, validation_info
    
    def train(self, 
              train_dataset: TrafficDataset, 
              val_dataset: TrafficDataset,
              batch_size: int = 128,  # 论文中的设置
              num_epochs: int = 25,   # 论文中的设置
              learning_rate: float = 1e-3,  # 论文中的设置
              patience: int = 10,
              save_path: str = 'best_model.pth') -> Dict:
        """完整训练流程（支持强化学习和对抗训练）"""
        
        # 数据加载器
        train_sampler = train_dataset.get_weighted_sampler()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 优化器和损失函数
        criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weights.to(self.device))
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # 训练配置
        best_val_loss = float('inf')
        patience_counter = 0
        best_robustness = 0.0
        
        print(f"开始训练，设备: {self.device}")
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        print(f"类别数: {len(train_dataset.classes)}")
        print(f"强化学习: {'启用' if self.use_rl else '禁用'}")
        print(f"对抗训练: {'启用' if self.use_adv else '禁用'}")
        print(f"探索率: {self.exploration_rate:.3f}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # 训练阶段（支持RL和对抗训练）
            train_loss, train_acc, train_rl_reward = self.train_epoch(
                train_loader, criterion, optimizer, epoch, num_epochs)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            if self.use_rl:
                self.train_history['rl_rewards'].append(train_rl_reward)
                self.train_history['exploration_rates'].append(self.exploration_rate)
            
            # 验证阶段（包含对抗样本评估）
            val_loss, val_acc, val_info = self.validate_epoch(val_loader, criterion)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            
            if self.use_adv:
                self.train_history['adversarial_acc'].append(val_info['adversarial_accuracy'])
                self.train_history['robustness_scores'].append(val_info['robustness_score'])
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 打印训练信息
            elapsed = time.time() - start_time\n            log_info = f'Epoch {epoch+1:3d}/{num_epochs:3d} | ' \\
                      f'Time: {elapsed:6.1f}s | ' \\
                      f'Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} | ' \\
                      f'Val: Loss={val_loss:.4f}, Acc={val_acc:.4f} | ' \\
                      f'LR={optimizer.param_groups[0]["lr"]:.2e}'
            
            if self.use_rl:
                log_info += f' | RL_Reward={train_rl_reward:.4f} | Exp={self.exploration_rate:.3f}'
            
            if self.use_adv:
                log_info += f' | Adv_Acc={val_info["adversarial_accuracy"]:.4f} | Robust={val_info["robustness_score"]:.1f}%'
            
            print(log_info)
            
            # 特征权重分析
            current_weights = val_info['feature_weights']
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                print(f'  特征权重: 时间={current_weights[0]:.3f}, 拓扑={current_weights[1]:.3f}, 序列={current_weights[2]:.3f}')
            
            # 早停和模型保存（基于综合指标）
            current_score = val_acc
            if self.use_adv:
                current_score = 0.7 * val_acc + 0.3 * (val_info['adversarial_accuracy'] / 100)
            
            if current_score > best_robustness:
                best_robustness = current_score
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f'  >>> New best model saved (score: {current_score:.4f})')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(save_path))
        
        print(f"训练完成！最佳验证损失: {best_val_loss:.4f}")
        if self.use_adv:
            final_robustness = self.train_history['robustness_scores'][-1] if self.train_history['robustness_scores'] else 0
            print(f"最终鲁棒性分数: {final_robustness:.1f}%")
        
        return self.train_history
    
    def test(self, test_dataset: TrafficDataset, batch_size: int = 128, model_path: str = 'best_model.pth') -> Dict:
        """测试模型（支持对抗样本评估和强化学习）"""
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        adversarial_correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        all_weights = []
        adversarial_predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                time_feat = batch['time_feat'].to(self.device)
                topo_feat = batch['topo_feat'].to(self.device)
                seq_feat = batch['seq_feat'].to(self.device)
                labels = batch['label'].to(self.device)

                # 标准测试
                outputs = self.model(time_feat, topo_feat, seq_feat, apply_perturb=False)
                logits = outputs['logits']
                loss = criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_weights.extend(outputs['weights'].cpu().numpy())
                
                # 对抗样本测试
                if self.use_adv:
                    adv_outputs = self.model(time_feat, topo_feat, seq_feat, apply_perturb=True)
                    adv_logits = adv_outputs['logits']
                    _, adv_predicted = torch.max(adv_logits.data, 1)
                    adversarial_correct += (adv_predicted == labels).sum().item()
                    adversarial_predictions.extend(adv_predicted.cpu().numpy())
        
        test_loss = total_loss / len(test_loader)
        test_acc = correct / total
        avg_weights = np.mean(np.vstack(all_weights), axis=0)
        
        # 对抗样本结果
        adversarial_acc = adversarial_correct / total if self.use_adv and total > 0 else 0.0
        robustness_score = (adversarial_acc / test_acc) * 100 if test_acc > 0 and self.use_adv else 100.0
        
        # 计算指标
        cm = confusion_matrix(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, 
                                     target_names=test_dataset.classes, 
                                     output_dict=True)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'adversarial_accuracy': adversarial_acc,
            'robustness_score': robustness_score,
            'confusion_matrix': cm,
            'classification_report': report,
            'feature_weights': avg_weights,
            'all_labels': all_labels,
            'all_predictions': all_predictions,
            'all_weights': all_weights,
            'adversarial_predictions': adversarial_predictions if self.use_adv else None
        }
        
        # 打印测试结果
        print(f"\\n测试结果:")
        print(f"标准测试准确率: {test_acc:.4f}")
        if self.use_adv:
            print(f"对抗样本准确率: {adversarial_acc:.4f}")
            print(f"鲁棒性分数: {robustness_score:.1f}%")
        print(f"特征权重: 时间={avg_weights[0]:.3f}, 拓扑={avg_weights[1]:.3f}, 序列={avg_weights[2]:.3f}")
        
        return results
    
    def run_perturbation_experiment(self, data_dir: str, batch_size: int = 32, 
                                  num_epochs: int = 10, learning_rate: float = 1e-3) -> Dict:
        """运行扰动实验"""
        
        perturbation_types = ['time', 'topo', 'seq', 'none']
        results = {}
        
        for perturb_type in perturbation_types:
            print(f"\n{'=' * 60}")
            print(f"开始扰动实验: {perturb_type.upper()} 特征扰动")
            print(f"{'=' * 60}")
            
            # 根据扰动类型创建数据集
            add_perturbation = (perturb_type != 'none')
            dataset = TrafficDataset(
                data_dir,
                add_perturbation=add_perturbation,
                perturbed_feature_type=perturb_type if add_perturbation else None
            )
            
            if len(dataset) == 0:
                print(f"错误: 没有加载到数据。")
                continue
            
            # 划分数据集
            train_size = int(0.7 * len(dataset))
            val_size = int(0.15 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            
            train_dataset, temp_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size + test_size])
            val_dataset, test_dataset = torch.utils.data.random_split(
                temp_dataset, [val_size, test_size])
            
            # 训练模型
            criterion = nn.CrossEntropyLoss(weight=dataset.class_weights.to(self.device))
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
            
            # 简化的训练循环（用于演示）
            feature_weights_history = []
            self.model.train()
            
            for epoch in range(min(10, num_epochs)):
                self.train_epoch(None, criterion, optimizer)  # 简化版本
                
                # 记录权重
                if self.feature_weights_history:
                    avg_weights = self.feature_weights_history[-1]
                    feature_weights_history.append(avg_weights)
                    print(f"Epoch {epoch + 1}: 特征权重 - 时序: {avg_weights[0]:.4f}, "
                          f"拓扑: {avg_weights[1]:.4f}, 序列: {avg_weights[2]:.4f}")
            
            # 测试
            test_results = self.test(test_dataset, batch_size)
            
            # 存储结果
            results[perturb_type] = {
                'final_weights': test_results['feature_weights'],
                'weight_history': feature_weights_history,
                'test_accuracy': test_results['test_accuracy']
            }
            
            print(f"{perturb_type.upper()} 扰动实验完成")
            print(f"最终特征权重 - 时序: {test_results['feature_weights'][0]:.4f}, "
                  f"拓扑: {test_results['feature_weights'][1]:.4f}, "
                  f"序列: {test_results['feature_weights'][2]:.4f}")
        
        # 分析结果
        self.analyze_perturbation_results(results, perturbation_types)
        
        return results
    
    def analyze_perturbation_results(self, results: Dict, perturbation_types: List[str]):
        """分析扰动实验结果"""
        print(f"\n{'=' * 60}")
        print("扰动实验结果分析")
        print(f"{'=' * 60}")
        
        # 创建结果数据框
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
        
        df = pd.DataFrame(data)
        
        # 绘制分析图
        self.plot_perturbation_analysis(df, results, perturbation_types)
        
        # 详细分析
        print("\n详细分析结果:")
        print(df)
        
        # 验证强化学习机制
        print("\n强化学习机制有效性验证:")
        if 'none' in results:
            baseline = results['none']['final_weights']
            for perturb_type in perturbation_types:
                if perturb_type != 'none' and perturb_type in results:
                    perturbed = results[perturb_type]['final_weights']
                    
                    # 找出被扰动的特征索引
                    perturb_idx = {'time': 0, 'topo': 1, 'seq': 2}.get(perturb_type, -1)
                    if perturb_idx >= 0:
                        weight_reduction = (baseline[perturb_idx] - perturbed[perturb_idx]) / baseline[perturb_idx] * 100
                        print(f"{perturb_type.upper()}特征扰动: 权重降低 {weight_reduction:.2f}%")
                        
                        if weight_reduction > 10:
                            print(f"  ✓ 强化学习机制有效降低了对{perturb_type}特征的依赖")
                        else:
                            print(f"  ✗ 强化学习机制对{perturb_type}特征的权重调节不明显")
        
        # 保存结果
        df.to_csv('perturbation_experiment_details.csv', index=False)
        print("\n详细结果已保存到 'perturbation_experiment_details.csv'")
    
    def plot_perturbation_analysis(self, df: pd.DataFrame, results: Dict, perturbation_types: List[str]):
        """绘制扰动分析图"""
        plt.figure(figsize=(15, 10))
        
        # 子图1: 不同扰动下的特征权重对比
        plt.subplot(2, 2, 1)
        pivot_df = df.pivot(index='扰动类型', columns='特征类型', values='权重')
        pivot_df.plot(kind='bar', ax=plt.gca())
        plt.title('不同扰动类型下的特征权重对比')
        plt.ylabel('平均权重')
        plt.xticks(rotation=45)
        plt.legend(title='特征类型')
        
        # 子图2: 权重变化热力图
        plt.subplot(2, 2, 2)
        heatmap_data = pivot_df.values
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=['时序特征', '拓扑特征', '序列特征'], 
                    yticklabels=pivot_df.index)
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
        
        # 子图4: 扰动影响分析
        plt.subplot(2, 2, 4)
        if 'none' in results:
            baseline = results['none']['final_weights']
            impact_analysis = []
            
            for perturb_type in perturbation_types:
                if perturb_type != 'none' and perturb_type in results:
                    perturbed = results[perturb_type]['final_weights']
                    feature_names = ['时序特征', '拓扑特征', '序列特征']
                    for i, feature_name in enumerate(feature_names):
                        reduction = (baseline[i] - perturbed[i]) / baseline[i] * 100
                        impact_analysis.append({
                            '扰动类型': perturb_type.upper(),
                            '特征类型': feature_name,
                            '权重降低百分比': reduction
                        })
            
            if impact_analysis:
                impact_df = pd.DataFrame(impact_analysis)
                pivot_impact = impact_df.pivot(index='扰动类型', columns='特征类型', values='权重降低百分比')
                pivot_impact.plot(kind='bar', ax=plt.gca())
                plt.title('扰动对特征权重的影响（降低百分比）')
                plt.ylabel('权重降低百分比 (%)')
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('perturbation_experiment_results.png', dpi=300, bbox_inches='tight')
        print("扰动实验结果图已保存为 'perturbation_experiment_results.png'")
