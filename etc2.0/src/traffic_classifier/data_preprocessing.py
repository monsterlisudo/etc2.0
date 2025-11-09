"""
数据预处理模块

负责处理网络流量数据（PCAP文件）并提取：
- 时序特征：包大小、时间间隔等
- 拓扑特征：IP、端口、连接信息等  
- 序列特征：协议、端口号、TCP标志等
"""

import os
import numpy as np
from scapy.all import rdpcap


class TrafficPreprocessor:
    """流量数据预处理器
    
    从PCAP文件中提取多模态特征，支持特征扰动功能
    
    Args:
        window_size (int): 窗口大小，默认为100个包
        add_perturbation (bool): 是否添加特征扰动
        perturbed_feature_type (str): 扰动的特征类型 ('time', 'topo', 'seq')
    """
    
    def __init__(self, window_size=100, add_perturbation=False, perturbed_feature_type=None):
        self.window_size = window_size
        self.add_perturbation = add_perturbation
        self.perturbed_feature_type = perturbed_feature_type  # 'time', 'topo', 'seq'

    def add_feature_perturbation(self, time_feat, topo_feat, seq_feat):
        """为指定类型的特征添加扰动"""
        if self.perturbed_feature_type == 'time':
            # 为时序特征添加高斯噪声
            noise = np.random.normal(0, 0.5, time_feat.shape)
            time_feat = time_feat + noise
        elif self.perturbed_feature_type == 'topo':
            # 为拓扑特征添加随机扰动
            noise = np.random.normal(0, 0.3, topo_feat.shape)
            topo_feat = topo_feat + noise
        elif self.perturbed_feature_type == 'seq':
            # 为序列特征添加随机扰动
            noise = np.random.normal(0, 0.4, seq_feat.shape)
            seq_feat = seq_feat + noise

        return time_feat, topo_feat, seq_feat

    def process_pcap(self, pcap_path):
        """处理单个pcap文件，提取时序、拓扑和序列特征"""
        try:
            packets = rdpcap(pcap_path)
        except Exception as e:
            print(f"Error reading {pcap_path}: {str(e)}")
            return None, None, None

        if len(packets) == 0:
            return None, None, None

        time_features = []
        sequence_features = []
        flow_stats = {
            'pkt_sizes': [],
            'inter_arrival_times': [],
            'protocols': set(),
            'tcp_flags': [],
            'udp_lens': []
        }

        # 初始化
        prev_time = packets[0].time

        # 提取基本特征
        for i, pkt in enumerate(packets[:self.window_size]):
            # 计算时间差
            time_diff = float(pkt.time - prev_time) if i > 0 else 0.0
            pkt_time = float(pkt.time % 1e6) if hasattr(pkt, 'time') else 0.0
            pkt_len = float(len(pkt))

            # 基本时序特征
            time_features.append([
                pkt_len,  # 包长度
                time_diff,  # 包间隔
                1.0 if pkt.haslayer('IP') else 0.0,  # 是否为IP包
                pkt_time,  # 包时间戳
                float(i) / self.window_size  # 相对位置特征
            ])

            # 协议特征提取（增强）
            protocol = 0.0
            src_port = 0.0
            dst_port = 0.0
            tcp_flag = 0.0
            udp_len = 0.0

            if pkt.haslayer('IP'):
                protocol = float(pkt['IP'].proto)
                flow_stats['protocols'].add(protocol)

                # TCP特征
                if pkt.haslayer('TCP'):
                    src_port = float(pkt['TCP'].sport)
                    dst_port = float(pkt['TCP'].dport)
                    # 提取TCP标志位作为特征
                    tcp_flag = float(int(pkt['TCP'].flags))
                    flow_stats['tcp_flags'].append(tcp_flag)

                # UDP特征
                elif pkt.haslayer('UDP'):
                    src_port = float(pkt['UDP'].sport)
                    dst_port = float(pkt['UDP'].dport)
                    udp_len = float(pkt['UDP'].len)
                    flow_stats['udp_lens'].append(udp_len)

            sequence_features.append([
                protocol,
                src_port / 65535.0,  # 归一化端口号
                dst_port / 65535.0,  # 归一化端口号
                tcp_flag / 63.0,  # 归一化TCP标志 (6位标志，最大值为63)
                udp_len / 1500.0  # 归一化UDP长度
            ])

            # 更新统计信息
            flow_stats['pkt_sizes'].append(pkt_len)
            flow_stats['inter_arrival_times'].append(time_diff)

            # 更新前一个包时间
            prev_time = pkt.time

        # 填充处理
        if len(time_features) < self.window_size:
            pad_len = self.window_size - len(time_features)
            time_features += [[0.0] * 5 for _ in range(pad_len)]
            sequence_features += [[0.0] * 5 for _ in range(pad_len)]

        # 提取增强的拓扑特征
        src_ips = set()
        dst_ips = set()
        ports = set()
        connections = set()  # 源IP-目的IP对

        for pkt in packets[:self.window_size]:
            if pkt.haslayer('IP'):
                src_ip = pkt['IP'].src
                dst_ip = pkt['IP'].dst
                src_ips.add(src_ip)
                dst_ips.add(dst_ip)
                connections.add((src_ip, dst_ip))

                # 端口信息
                if pkt.haslayer('TCP') or pkt.haslayer('UDP'):
                    proto_layer = 'TCP' if pkt.haslayer('TCP') else 'UDP'
                    ports.add(pkt[proto_layer].sport)
                    ports.add(pkt[proto_layer].dport)

        # 统计特征
        pkt_sizes = flow_stats['pkt_sizes']
        iat = flow_stats['inter_arrival_times']

        topology_features = [
            float(len(src_ips)),  # 源IP数量
            float(len(dst_ips)),  # 目的IP数量
            float(len(src_ips & dst_ips)),  # 源目IP重叠
            float(len(connections)),  # 连接数量
            float(len(ports)),  # 端口数量
            float(len(flow_stats['protocols'])),  # 协议种类数
            np.mean(pkt_sizes) if pkt_sizes else 0.0,  # 平均包大小
            np.std(pkt_sizes) if len(pkt_sizes) > 1 else 0.0,  # 包大小标准差
            np.mean(iat) if iat else 0.0,  # 平均包间隔
            np.std(iat) if len(iat) > 1 else 0.0,  # 包间隔标准差
        ]

        time_feat = np.array(time_features, dtype=np.float32)
        topo_feat = np.array(topology_features, dtype=np.float32)
        seq_feat = np.array(sequence_features, dtype=np.float32)

        # 添加特征扰动（如果启用）
        if self.add_perturbation:
            time_feat, topo_feat, seq_feat = self.add_feature_perturbation(time_feat, topo_feat, seq_feat)

        return time_feat, topo_feat, seq_feat
