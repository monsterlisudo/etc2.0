"""
数据集模块

自定义PyTorch数据集类，支持：
- 多模态特征加载
- 类别权重计算
- 数据标准化
- 特征扰动支持
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
from .data_preprocessing import TrafficPreprocessor


class TrafficDataset(Dataset):
    """流量分类数据集
    
    继承自PyTorch Dataset，处理网络流量数据
    支持时序、拓扑、序列三种特征模态
    
    Args:
        data_dir (str): 数据目录路径
        window_size (int): 窗口大小
        add_perturbation (bool): 是否添加扰动
        perturbed_feature_type (str): 扰动的特征类型
    """
    
    def __init__(self, data_dir, window_size=100, add_perturbation=False, perturbed_feature_type=None):
        self.preprocessor = TrafficPreprocessor(window_size, add_perturbation, perturbed_feature_type)
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.data = []
        self.class_counts = {c: 0 for c in self.classes}
        self.add_perturbation = add_perturbation
        self.perturbed_feature_type = perturbed_feature_type

        # 加载数据
        print(f"Loading data from {data_dir}...")
        if self.add_perturbation:
            print(f"Adding perturbation to {perturbed_feature_type} features")

        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            print(f"Processing class: {class_name}")
            class_files = [f for f in os.listdir(class_dir) if f.endswith('.pcap')]
            processed = 0

            for pcap_file in class_files:
                pcap_path = os.path.join(class_dir, pcap_file)
                time_feat, topo_feat, seq_feat = self.preprocessor.process_pcap(pcap_path)
                if time_feat is not None and topo_feat is not None and seq_feat is not None:
                    self.data.append((time_feat, topo_feat, seq_feat, self.class_to_idx[class_name]))
                    self.class_counts[class_name] += 1
                    processed += 1
                else:
                    print(f"Warning: Skipping invalid file {pcap_path}")

            print(f"  - Processed {processed}/{len(class_files)} files for class {class_name}")

        print(f"Total samples: {len(self.data)}")
        print(f"Class distribution: {self.class_counts}")

        # 标准化参数计算（增强稳定性）
        if len(self.data) > 0:
            time_features = np.vstack([d[0] for d in self.data])
            topo_features = np.vstack([d[1] for d in self.data])
            seq_features = np.vstack([d[2] for d in self.data])

            # 计算均值和标准差，处理零标准差的情况
            self.time_mean = torch.tensor(time_features.mean(axis=0), dtype=torch.float32)
            time_std = time_features.std(axis=0)
            time_std[time_std == 0] = 1.0  # 避免除零
            self.time_std = torch.tensor(time_std, dtype=torch.float32)
            
            self.topo_mean = torch.tensor(topo_features.mean(axis=0), dtype=torch.float32)
            topo_std = topo_features.std(axis=0)
            topo_std[topo_std == 0] = 1.0  # 避免除零
            self.topo_std = torch.tensor(topo_std, dtype=torch.float32)

            self.seq_mean = torch.tensor(seq_features.mean(axis=0), dtype=torch.float32)
            seq_std = seq_features.std(axis=0)
            seq_std[seq_std == 0] = 1.0  # 避免除零
            self.seq_std = torch.tensor(seq_std, dtype=torch.float32)
        else:
            print("Error: No valid data found.")
            raise ValueError("No valid data found in the dataset.")

        # 计算类别权重（优化不平衡问题）
        total_samples = sum(self.class_counts.values())
        self.class_weights = torch.tensor(
            [total_samples / (len(self.classes) * max(1, self.class_counts[c])) for c in self.classes],
            dtype=torch.float32
        )

    def __getitem__(self, idx):
        time_feat, topo_feat, seq_feat, label = self.data[idx]

        # 标准化
        time_feat = (torch.FloatTensor(time_feat) - self.time_mean) / self.time_std
        topo_feat = (torch.FloatTensor(topo_feat) - self.topo_mean) / self.topo_std
        seq_feat = (torch.FloatTensor(seq_feat) - self.seq_mean) / self.seq_std

        return {
            'time_feat': time_feat,
            'topo_feat': topo_feat,
            'seq_feat': seq_feat,
            'label': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)
    
    def get_weighted_sampler(self):
        """获取加权随机采样器，用于处理类别不平衡"""
        train_weights = []
        for _, _, _, label in self.data:
            class_name = self.classes[label]
            weight = 1.0 / np.sqrt(self.class_counts[class_name])
            train_weights.append(weight)
        
        train_weights = torch.tensor(train_weights, dtype=torch.float)
        train_weights = train_weights / train_weights.sum() * len(train_weights)
        return WeightedRandomSampler(weights=train_weights, num_samples=len(self.data), replacement=True)
