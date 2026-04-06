"""
聚类稳定性分析模块
用于收集和可视化聚类在不同时间步之间的稳定性和漂移
"""
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


class ClusterStabilityTracker:
    """跟踪聚类在不同时间步之间的稳定性"""

    def __init__(self, output_dir: str = "stability_analysis"):
        """
        Args:
            output_dir: 保存可视化图片的目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 存储每个时间步的数据: {timestep: {layer_idx: {q_centroids, k_centroids, qlabels, klabels}}}
        self.timestep_data: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = defaultdict(
            lambda: defaultdict(dict)
        )

    def record_timestep(
        self,
        timestep: int,
        layer_idx: int,
        q_centroids: torch.Tensor,
        k_centroids: torch.Tensor,
        qlabels: torch.Tensor,
        klabels: torch.Tensor,
    ):
        """
        记录某个时间步和层的聚类数据

        Args:
            timestep: 时间步 (通常从 50 到 0)
            layer_idx: 层索引
            q_centroids: Query 质心 [B*H, num_q_centroids, D]
            k_centroids: Key 质心 [B*H, num_k_centroids, D]
            qlabels: Query 标签 [B*H, seq_len]
            klabels: Key 标签 [B*H, seq_len]
        """
        # 转换为 CPU 并 detach，避免占用 GPU 内存
        self.timestep_data[timestep][layer_idx] = {
            "q_centroids": q_centroids.detach().cpu() if isinstance(q_centroids, torch.Tensor) else q_centroids,
            "k_centroids": k_centroids.detach().cpu() if isinstance(k_centroids, torch.Tensor) else k_centroids,
            "qlabels": qlabels.detach().cpu() if isinstance(qlabels, torch.Tensor) else qlabels,
            "klabels": klabels.detach().cpu() if isinstance(klabels, torch.Tensor) else klabels,
        }

    def compute_cluster_iou(
        self, labels_t: torch.Tensor, labels_t1: torch.Tensor, num_clusters: int
    ) -> float:
        """
        计算两个时间步之间聚类的 IoU (交并比)

        Args:
            labels_t: 时间步 t 的标签 [B*H, seq_len]
            labels_t1: 时间步 t+1 的标签 [B*H, seq_len]
            num_clusters: 聚类数量

        Returns:
            Average IoU across all clusters
        """
        # 将标签展平
        labels_t_flat = labels_t.flatten().long()
        labels_t1_flat = labels_t1.flatten().long()

        # 计算每个聚类的 IoU
        ious = []
        for k in range(num_clusters):
            mask_t = labels_t_flat == k
            mask_t1 = labels_t1_flat == k

            intersection = (mask_t & mask_t1).sum().item()
            union = (mask_t | mask_t1).sum().item()

            if union > 0:
                iou = intersection / union
                ious.append(iou)

        return np.mean(ious) if ious else 0.0

    def compute_centroid_drift(
        self, centroids_t: torch.Tensor, centroids_t1: torch.Tensor
    ) -> float:
        """
        计算质心漂移距离

        Args:
            centroids_t: 时间步 t 的质心 [B*H, num_clusters, D]
            centroids_t1: 时间步 t+1 的质心 [B*H, num_clusters, D]

        Returns:
            Total drift distance (sum of squared L2 distances)
        """
        # 计算每个质心的 L2 距离
        diff = centroids_t - centroids_t1
        squared_distances = (diff ** 2).sum(dim=-1)  # [B*H, num_clusters]
        total_drift = squared_distances.sum().item()

        return total_drift

    def compute_stability_metrics(self, layer_idx: int) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
        """
        计算某个层的稳定性指标

        Args:
            layer_idx: 层索引

        Returns:
            timesteps: 时间步列表（从大到小，如 50, 49, ..., 0）
            q_ious: Query 聚类的 IoU 列表
            k_ious: Key 聚类的 IoU 列表
            q_drifts: Query 质心漂移列表
            k_drifts: Key 质心漂移列表
        """
        # 获取该层所有时间步的数据
        layer_data = {}
        for timestep, layers in self.timestep_data.items():
            if layer_idx in layers:
                layer_data[timestep] = layers[layer_idx]

        if len(layer_data) < 2:
            return [], [], [], [], []

        # 按时间步排序（从大到小）
        timesteps = sorted(layer_data.keys(), reverse=True)

        q_ious = []
        k_ious = []
        q_drifts = []
        k_drifts = []

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t1 = timesteps[i + 1]

            data_t = layer_data[t]
            data_t1 = layer_data[t1]

            # 计算 IoU
            q_centroids_t = data_t["q_centroids"]
            q_centroids_t1 = data_t1["q_centroids"]
            qlabels_t = data_t["qlabels"]
            qlabels_t1 = data_t1["qlabels"]

            k_centroids_t = data_t["k_centroids"]
            k_centroids_t1 = data_t1["k_centroids"]
            klabels_t = data_t["klabels"]
            klabels_t1 = data_t1["klabels"]

            # Query IoU
            num_q_clusters = q_centroids_t.shape[1]
            q_iou = self.compute_cluster_iou(qlabels_t, qlabels_t1, num_q_clusters)
            q_ious.append(q_iou)

            # Key IoU
            num_k_clusters = k_centroids_t.shape[1]
            k_iou = self.compute_cluster_iou(klabels_t, klabels_t1, num_k_clusters)
            k_ious.append(k_iou)

            # Query Drift
            q_drift = self.compute_centroid_drift(q_centroids_t, q_centroids_t1)
            q_drifts.append(q_drift)

            # Key Drift
            k_drift = self.compute_centroid_drift(k_centroids_t, k_centroids_t1)
            k_drifts.append(k_drift)

        # timesteps 需要去掉最后一个（因为没有 t+1）
        return timesteps[:-1], q_ious, k_ious, q_drifts, k_drifts

    def visualize_step_comparison(
        self, timestep: int, layer_idx: int
    ) -> Optional[str]:
        """
        可视化当前 step 与上一个 step 的对比（如果存在上一个 step）

        Args:
            timestep: 当前时间步
            layer_idx: 层索引

        Returns:
            保存路径，如果没有上一个 step 则返回 None
        """
        # 获取该层所有时间步的数据
        layer_data = {}
        for ts, layers in self.timestep_data.items():
            if layer_idx in layers:
                layer_data[ts] = layers[layer_idx]

        if len(layer_data) < 2:
            return None  # 需要至少两个 step 才能对比

        # 按时间步排序（从大到小，因为 diffusion 是从高 timestep 到低 timestep）
        timesteps = sorted(layer_data.keys(), reverse=True)

        # 找到当前 timestep 和上一个 timestep
        # 注意：在 diffusion 中，timestep 从大到小（1000→999→998...）
        # 所以"上一个 step"是指更早处理的 step（更大的 timestep）
        if timestep not in timesteps:
            return None

        current_idx = timesteps.index(timestep)
        if current_idx == 0:
            return None  # 这是第一个 step（最大的 timestep），没有上一个

        prev_timestep = timesteps[current_idx - 1]  # 上一个 step 是更大的 timestep
        data_t = layer_data[timestep]
        data_t1 = layer_data[prev_timestep]

        # 计算 IoU
        qlabels_t = data_t["qlabels"]
        qlabels_t1 = data_t1["qlabels"]
        klabels_t = data_t["klabels"]
        klabels_t1 = data_t1["klabels"]

        num_q_clusters = data_t["q_centroids"].shape[1]
        num_k_clusters = data_t["k_centroids"].shape[1]

        q_iou = self.compute_cluster_iou(qlabels_t, qlabels_t1, num_q_clusters)
        k_iou = self.compute_cluster_iou(klabels_t, klabels_t1, num_k_clusters)

        # 计算 Drift
        q_centroids_t = data_t["q_centroids"]
        q_centroids_t1 = data_t1["q_centroids"]
        k_centroids_t = data_t["k_centroids"]
        k_centroids_t1 = data_t1["k_centroids"]

        q_drift = self.compute_centroid_drift(q_centroids_t, q_centroids_t1)
        k_drift = self.compute_centroid_drift(k_centroids_t, k_centroids_t1)

        # 创建对比图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 左图: IoU 对比（柱状图）
        categories = ["Q Cluster", "K Cluster"]
        ious = [q_iou, k_iou]
        colors = ["#3498db", "#e74c3c"]
        bars = ax1.bar(categories, ious, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
        ax1.set_ylabel("IoU", fontsize=12)
        ax1.set_title(f"Cluster IoU: Step {prev_timestep} → {timestep}\n(Layer {layer_idx})", fontsize=13, fontweight="bold")
        ax1.set_ylim([0, 1.1])
        ax1.grid(True, alpha=0.3, axis="y")
        # 在柱状图上添加数值标签
        for bar, iou in zip(bars, ious):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{iou:.3f}', ha='center', va='bottom', fontsize=11, fontweight="bold")

        # 右图: Drift 对比（柱状图）
        drifts = [q_drift, k_drift]
        bars2 = ax2.bar(categories, drifts, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
        ax2.set_ylabel("Drift Distance (L2²)", fontsize=12)
        ax2.set_title(f"Centroid Drift: Step {prev_timestep} → {timestep}\n(Layer {layer_idx})", fontsize=13, fontweight="bold")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3, axis="y")
        # 在柱状图上添加数值标签
        for bar, drift in zip(bars2, drifts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{drift:.2e}', ha='center', va='bottom', fontsize=10, fontweight="bold")

        plt.tight_layout()

        # 保存图片
        save_path = os.path.join(self.output_dir, f"step_comparison_layer_{layer_idx}_step_{timestep}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    def visualize_stability(
        self, layer_idx: int, save_path: Optional[str] = None
    ):
        """
        可视化某个层的稳定性分析结果（所有时间步的完整图表）

        Args:
            layer_idx: 层索引
            save_path: 保存路径，如果为 None 则使用默认路径
        """
        timesteps, q_ious, k_ious, q_drifts, k_drifts = self.compute_stability_metrics(
            layer_idx
        )

        if len(timesteps) == 0:
            print(f"No data for layer {layer_idx}")
            return

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 图1: Cluster IoU over Time
        ax1.plot(timesteps, q_ious, label="Q Cluster IoU", marker="o", linewidth=2)
        ax1.plot(timesteps, k_ious, label="K Cluster IoU", marker="s", linewidth=2)
        ax1.set_xlabel("Timestep", fontsize=12)
        ax1.set_ylabel("Average IoU", fontsize=12)
        ax1.set_title(f"Cluster IoU over Time (Layer {layer_idx})", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.1])

        # 图2: Centroid Drift Distance
        ax2.plot(timesteps, q_drifts, label="Q Centroid Drift", marker="o", linewidth=2)
        ax2.plot(timesteps, k_drifts, label="K Centroid Drift", marker="s", linewidth=2)
        ax2.set_xlabel("Timestep", fontsize=12)
        ax2.set_ylabel("Drift Distance (L2²)", fontsize=12)
        ax2.set_title(f"Centroid Drift Distance (Layer {layer_idx})", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")  # 使用对数刻度，因为漂移可能变化很大

        plt.tight_layout()

        # 保存图片
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"stability_layer_{layer_idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Stability analysis saved to: {save_path}")
        plt.close()

    def visualize_all_layers(self):
        """可视化所有层"""
        # 获取所有层索引
        all_layers = set()
        for timestep_data in self.timestep_data.values():
            all_layers.update(timestep_data.keys())

        for layer_idx in sorted(all_layers):
            self.visualize_stability(layer_idx)
