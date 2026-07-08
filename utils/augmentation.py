"""数据增强工具 — 适用于 MNIST 的 numpy/Tensor 级变换。

本项目使用 fetch_openml 加载数据（numpy 数组），然后用 TensorDataset。
为了保持项目风格一致，这里提供两种增强方式：
1. numpy 级增强：在转 Tensor 之前应用（使用 scipy.ndimage）
2. Tensor 级增强：用 AugmentedDataset 包装 TensorDataset，应用 torchvision transforms

MNIST 增强注意事项：
- 可以用 RandomRotation（小角度）、RandomAffine（平移/缩放）、RandomPerspective
- **不能用 RandomHorizontalFlip**：数字翻转后语义改变（6→9）
- **不能用 RandomVerticalFlip**：同样改变语义
- RandomErasing/Cutout 对 CNN/ResNet 有效，对 MLP 效果有限
"""

import torch
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
        return T.Compose([
            T.RandomRotation(degrees=10, fill=0),           # 随机旋转 ±10°
            T.RandomAffine(degrees=0, translate=(0.1, 0.1),  # 随机平移 ±10%
                          scale=(0.9, 1.1)),                # 随机缩放 90%~110%
            T.RandomPerspective(distortion_scale=0.15,       # 随机透视变换
                               p=0.3, fill=0),
            T.RandomErasing(p=0.2, scale=(0.02, 0.1),        # 随机擦除（Cutout）
                          ratio=(0.3, 3.3), value=0),
        ])
    else:
        return T.Compose([])   # 测试集不增强


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
