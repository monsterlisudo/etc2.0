# Traffic Fingerprinting Network - 流量指纹识别项目

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目简介

Traffic Fingerprinting Network 是一个基于深度学习的网络流量分类和指纹识别系统。该项目采用多模态融合技术，结合强化学习机制，能够有效识别不同类型网络流量并进行准确分类。

### 核心特性

- **多模态特征提取**: 支持时序、拓扑、序列三种特征模态
- **强化学习机制**: 动态特征选择器实现智能权重调节
- **对抗训练**: 增强模型鲁棒性和泛化能力
- **扰动实验**: 验证强化学习机制的有效性
- **可视化分析**: 提供丰富的训练和结果分析工具

## 项目结构

```
traffic-fingerprinting/
├── README.md                 # 项目说明文档
├── requirements.txt          # 依赖包列表
├── setup.py                 # 安装配置
├── .gitignore              # Git忽略文件
├── LICENSE                 # 开源协议
├── 
├── src/                    # 源代码目录
│   ├── traffic_classifier/ # 核心包
│   │   ├── __init__.py    # 包初始化
│   │   ├── models.py      # 神经网络模型
│   │   ├── data_preprocessing.py # 数据预处理
│   │   ├── dataset.py     # 数据集类
│   │   ├── trainer.py     # 训练器
│   │   └── utils.py       # 工具函数
│   └── main.py            # 主程序入口
├── 
├── config/                 # 配置文件目录
│   ├── model_config.yaml  # 模型配置
│   ├── training_config.yaml # 训练配置
│   └── experiment_config.yaml # 实验配置
├── 
├── examples/              # 示例代码
│   ├── basic_usage.py     # 基础使用示例
│   ├── perturbation_experiment.py # 扰动实验示例
│   └── visualization.py   # 可视化示例
├── 
├── tests/                # 测试代码
│   ├── test_models.py    # 模型测试
│   ├── test_data.py      # 数据测试
│   └── test_trainer.py   # 训练器测试
├── 
├── docs/                 # 文档目录
│   ├── api/              # API文档
│   ├── tutorial/         # 教程文档
│   └── research/         # 研究文档
├── 
├── data/                 # 数据目录
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后数据
├── 
├── models/               # 模型保存目录
│   └── checkpoints/      # 模型检查点
├── 
└── results/              # 结果目录
    ├── logs/             # 训练日志
    ├── plots/            # 图表文件
    └── reports/          # 实验报告
```

## 安装说明

### 环境要求

- Python 3.7+
- PyTorch 1.9+
- CUDA 11.0+ (推荐，用于GPU加速)

### 快速安装

1. **克隆项目**
```bash
git clone https://github.com/your-username/traffic-fingerprinting.git
cd traffic-fingerprinting
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **安装项目**
```bash
pip install -e .
```

### 详细安装

如果需要安装额外的功能依赖：
```bash
# 安装可视化工具
pip install plotly>=5.0.0

# 安装Jupyter支持
pip install jupyter>=1.0.0

# 安装性能分析工具
pip install thop>=0.1.0
```

## 使用指南

### 1. 基础使用

#### 准备数据
项目需要按照以下结构组织PCAP文件：
```
data/
├── website_1/
│   ├── traffic1.pcap
│   ├── traffic2.pcap
│   └── ...
├── website_2/
│   ├── traffic1.pcap
│   └── ...
└── ...
```

#### 训练模型
```bash
# 基础训练
python src/main.py --data_dir /path/to/data --mode train

# 自定义参数训练
python src/main.py \
    --data_dir /path/to/data \
    --mode train \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0001 \
    --results_dir ./my_results
```

#### 运行扰动实验
```bash
python src/main.py --data_dir /path/to/data --mode perturb
```

### 2. Python API使用

```python
from src.traffic_classifier import TrafficModel, TrafficDataset
from src.traffic_classifier.trainer import TrafficTrainer

# 创建数据集
dataset = TrafficDataset('data/', window_size=100)

# 创建模型
model = TrafficModel(
    num_classes=len(dataset.classes),
    time_dim=5,
    topo_dim=10,
    seq_dim=5
)

# 创建训练器
trainer = TrafficTrainer(model)

# 训练模型
trainer.train(train_dataset, val_dataset, epochs=50)

# 测试模型
results = trainer.test(test_dataset)
print(f"Test Accuracy: {results['test_accuracy']:.4f}")
```

### 3. 示例代码

参考 `examples/` 目录下的示例代码：

- `basic_usage.py` - 基础使用教程
- `perturbation_experiment.py` - 扰动实验示例  
- `visualization.py` - 可视化工具使用

## 核心算法

### 1. 多模态特征融合

- **时序特征**: LSTM编码器 + 多头自注意力
- **拓扑特征**: 全连接网络 + 层归一化
- **序列特征**: Transformer编码器

### 2. 强化学习机制

动态特征选择器通过学习到的权重实现智能特征融合：
- 输入：融合后的多模态特征
- 输出：三类特征的重要性权重 [time_weight, topo_weight, seq_weight]
- 训练：通过端到端反向传播学习最优权重分配

### 3. 对抗训练

使用AdversarialGenerator生成对抗样本：
- 增强模型对噪声的鲁棒性
- 防止过拟合，提高泛化能力

## 性能指标

### 1. 分类性能
- 准确率 (Accuracy)
- 精确率 (Precision)  
- 召回率 (Recall)
- F1分数 (F1-Score)

### 2. 模型效率
- 推理时间
- 内存占用
- 计算复杂度

### 3. 强化学习验证
- 特征权重变化趋势
- 扰动实验结果分析
- 权重稳定性指标

## 研究论文

本项目基于最新的深度学习和强化学习技术，主要创新点：

1. **多模态融合**: 首次将时序、拓扑、序列特征统一建模
2. **强化学习机制**: 动态特征选择器实现智能权重调节
3. **扰动实验**: 创新的验证方法证明强化学习机制有效性

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 开发规范

- 遵循 PEP 8 代码风格
- 添加必要的单元测试
- 更新相关文档
- 确保所有测试通过

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

- 作者:Li Gao
- 邮箱: xdgaoli@qq.com
- 项目链接: https://github.com/monsterlisudo/etc2.0

## 致谢



## 更新日志

### v1.0.0 (2025-11-09)
- 初始版本发布
- 多模态流量分类模型
- 强化学习动态特征选择
- 对抗训练机制
- 扰动实验验证
- 完整训练和测试流程
