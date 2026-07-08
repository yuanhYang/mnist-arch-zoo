"""数据增强工具 — 适用于 MNIST 的 numpy/Tensor 级变换。

本项目使用 fetch_openml 加载数据（numpy 数组），然后用 TensorDataset。
提供三种增强方式：
1. CPU 逐样本增强：AugmentedDataset 包装 TensorDataset + torchvision transforms
2. GPU 批量增强：GPUBatchAugment 在 GPU 上对整个 batch 做变换（推荐，GPU 利用率高）
3. numpy 级增强：在转 Tensor 之前应用（使用 scipy.ndimage）

MNIST 增强注意事项：
- 可以用 RandomRotation（小角度）、RandomAffine（平移/缩放）、RandomPerspective
- **不能用 RandomHorizontalFlip**：数字翻转后语义改变（6→9）
- **不能用 RandomVerticalFlip**：同样改变语义
- RandomErasing/Cutout 对 CNN/ResNet 有效，对 MLP 效果有限
- GPU 批量增强将旋转+平移+缩放合并为一次 grid_sample，大幅减少重采样开销
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class AugmentedDataset(Dataset):
    """为 TensorDataset 添加数据增强的包装器。

    Args:
        data:   图像 Tensor，形状 (N, C, H, W) 或 (N, H, W)
        targets: 标签 Tensor，形状 (N,)
        transform: torchvision.transforms 组合（作用于单个样本）
        flatten:  是否在 transform 后展平（MLP 需要）
    """

    def __init__(self, data, targets, transform=None, flatten=False):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.flatten = flatten

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]

        if self.transform is not None:
            x = self.transform(x)

        if self.flatten:
            x = x.view(-1)

        return x, y


def get_mnist_transforms(augment=True):
    """获取 MNIST 标准变换 pipeline。

    适用于 (1, 28, 28) 图像 Tensor（值域 [0, 1]）。

    Args:
        augment: True 返回训练增强，False 返回仅 Normalize

    Returns:
        torchvision.transforms.Compose
    """
    import torchvision.transforms as T

    if augment:
        # 优化：旋转 + 平移 + 缩放合并为一次 RandomAffine，减少一次图像重采样
        return T.Compose([
            T.RandomAffine(degrees=10, translate=(0.1, 0.1),   # 旋转 ±10° + 平移 ±10%
                          scale=(0.9, 1.1), fill=0),           # 缩放 90%~110% — 合并为一次重采样
            T.RandomPerspective(distortion_scale=0.15,          # 随机透视变换
                               p=0.3, fill=0),
            T.RandomErasing(p=0.2, scale=(0.02, 0.1),          # 随机擦除（Cutout）
                          ratio=(0.3, 3.3), value=0),
        ])
    else:
        return T.Compose([])   # 测试集不增强


class GPUBatchAugment(nn.Module):
    """GPU 批量数据增强 — 将旋转/平移/缩放合并为一次 grid_sample。

    相比 CPU 逐样本增强的优势：
    1. 旋转+平移+缩放合并为一次 GPU affine 变换（省 2 次图像重采样）
    2. RandomErasing 直接在 GPU 上批量完成
    3. 整体增强在 GPU 上执行，消除 CPU→GPU 的增强瓶颈
    4. GPU 算图像变换比 CPU 快 10-50x（取决于图像大小）

    用法：
        augment = GPUBatchAugment(rotation=10, translate=0.1).to(device)
        x_aug = augment(x)  # x: (B, C, H, W) on GPU，值域 [0, 1]

    注意：
        - 图像值域应为 [0, 1]（与 torchvision 一致），填充值为 0（黑色）
        - 不包含 RandomPerspective（GPU 批量 perspective 实现复杂，且 affine 已覆盖足够不变性）
        - 本模块是纯 GPU 操作，不需要 CPU→GPU 数据传输
    """

    def __init__(self, rotation=10, translate=0.1, scale_range=(0.9, 1.1),
                 erase_p=0.2, erase_scale=(0.02, 0.1), erase_ratio=(0.3, 3.3)):
        super().__init__()
        self.rotation = rotation
        self.translate = translate
        self.scale_range = scale_range
        self.erase_p = erase_p
        self.erase_scale = erase_scale
        self.erase_ratio = erase_ratio

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) Tensor on GPU，值域 [0, 1]
        Returns:
            augmented x, same shape & device
        """
        B, C, H, W = x.shape
        device = x.device

        # ============================================================
        # 1. 合并 Affine：旋转 + 平移 + 缩放 → 一次 grid_sample
        # ============================================================
        angles = (torch.rand(B, device=device) * 2 - 1) * self.rotation          # [-r, +r] 度
        tx = (torch.rand(B, device=device) * 2 - 1) * self.translate * 2.0       # 归一化平移
        ty = (torch.rand(B, device=device) * 2 - 1) * self.translate * 2.0
        s = (torch.rand(B, device=device)
             * (self.scale_range[1] - self.scale_range[0])
             + self.scale_range[0])                                                # [s_min, s_max]

        rad = torch.deg2rad(angles)
        cos_a, sin_a = torch.cos(rad), torch.sin(rad)

        # 构造 (B, 2, 3) 仿射矩阵
        theta = torch.zeros(B, 2, 3, device=device)
        theta[:, 0, 0] = cos_a / s
        theta[:, 0, 1] = -sin_a / s
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin_a / s
        theta[:, 1, 1] = cos_a / s
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')

        # ============================================================
        # 2. 批量 RandomErasing (Cutout)
        # ============================================================
        erase_mask = torch.rand(B, device=device) < self.erase_p
        n_erase = erase_mask.sum().item()
        if n_erase > 0:
            erase_indices = erase_mask.nonzero(as_tuple=True)[0]
            for idx in erase_indices:
                i = idx.item()
                area = H * W
                target_area = (torch.rand(1, device=device).item()
                               * (self.erase_scale[1] - self.erase_scale[0])
                               + self.erase_scale[0]) * area
                aspect = (torch.rand(1, device=device).item()
                          * (self.erase_ratio[1] - self.erase_ratio[0])
                          + self.erase_ratio[0])

                h_e = min(H, max(1, int(round(math.sqrt(target_area * aspect)))))
                w_e = min(W, max(1, int(round(math.sqrt(target_area / aspect)))))

                y1 = torch.randint(0, H - h_e + 1, (1,), device=device).item()
                x1 = torch.randint(0, W - w_e + 1, (1,), device=device).item()
                x[i, :, y1:y1 + h_e, x1:x1 + w_e] = 0.0

        return x


def visualize_augmentation(data, transform, num_samples=8, num_aug=4):
    """可视化数据增强效果。

    Args:
        data:      图像 Tensor，形状 (N, C, H, W)
        transform: torchvision transforms
        num_samples: 展示几个原始样本
        num_aug:    每个样本生成几个增强版本

    Returns:
        matplotlib fig
    """
    import matplotlib.pyplot as plt

    idxs = torch.randperm(len(data))[:num_samples]
    fig, axes = plt.subplots(num_samples, num_aug + 1,
                             figsize=(2 * (num_aug + 1), 2 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(idxs):
        original = data[idx]
        # 原始图像
        img = original.squeeze().numpy()
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title("原始" if i == 0 else "")
        axes[i, 0].axis("off")

        # 增强版本
        for j in range(num_aug):
            aug = transform(original)
            axes[i, j + 1].imshow(aug.squeeze().numpy(), cmap="gray")
            axes[i, j + 1].set_title(f"增强{j+1}" if i == 0 else "")
            axes[i, j + 1].axis("off")

    fig.suptitle("数据增强效果展示", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig
