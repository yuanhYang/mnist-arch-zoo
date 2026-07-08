"""
Grad-CAM（梯度加权类激活映射）可视化工具

原理：
1. 前向传播 → 获得目标类别的分数
2. 反向传播 → 计算目标卷积层激活图对分数的梯度
3. 全局平均池化梯度 → 每个通道的权重
4. 加权组合激活图 → ReLU → 热力图
5. 上采样叠加到原图

参考：Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image


class GradCAM:
    """
    Grad-CAM：对任意 CNN 模型计算类别激活热力图

    用法：
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam(input_tensor, target_class=None)
        # heatmap 是 numpy array, shape=(H, W)，值域 [0, 1]

    参数：
        model:        已训练好的 nn.Module
        target_layer: 目标卷积层（通常选最后一个卷积层）
                      例如 model.layer4[-1].conv2 或 model.conv_blocks[-1][-4]
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # 注册前向 hook：保存目标层的激活值
        self._forward_handle = target_layer.register_forward_hook(
            self._save_activation
        )
        # 注册反向 hook：保存目标层的梯度
        self._backward_handle = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, target_class=None):
        """
        计算 Grad-CAM 热力图

        参数：
            x:              输入张量，shape=(1, C, H, W) 或 (C, H, W)
            target_class:   目标类别索引，None 时自动取预测类别

        返回：
            heatmap:        numpy array, shape=(H, W)，值域 [0, 1]
            pred_class:     预测类别
        """
        # 确保是 batch 维度
        if x.dim() == 3:
            x = x.unsqueeze(0)

        self.model.eval()

        # ① 前向传播
        output = self.model(x)
        pred_class = output.argmax(dim=1).item()

        if target_class is None:
            target_class = pred_class

        # ② 反向传播：计算目标类别分数对激活图的梯度
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # ③ 全局平均池化梯度 → 通道权重
        # gradients: (1, C, H', W')
        gradients = self.gradients[0]  # (C, H', W')
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # ④ 加权组合激活图
        # activations: (1, C, H', W')
        activations = self.activations[0]  # (C, H', W')
        cam = torch.zeros(activations.shape[1:], dtype=activations.dtype,
                          device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ⑤ ReLU: 只保留对目标类别有正向贡献的区域
        cam = torch.relu(cam)

        # ⑥ 上采样到输入尺寸
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, H', W')
        cam = nn.functional.interpolate(
            cam, size=(x.shape[2], x.shape[3]),
            mode='bilinear', align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # ⑦ 归一化到 [0, 1]
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        return cam, pred_class

    def remove_hooks(self):
        """清理 hooks（释放资源）"""
        self._forward_handle.remove()
        self._backward_handle.remove()


# ============================================================
# 便捷函数：获取各模型的目标层
# ============================================================

def get_cnn_target_layer(model):
    """
    获取 SimpleCNN 的 Grad-CAM 目标层（最后一个卷积块的最后一个 Conv2d）
    SimpleCNN 结构：
        conv_blocks: ModuleList[
            Sequential[Conv→BN→ReLU→Conv→BN→ReLU→MaxPool→Dropout2d],   # block 0
            Sequential[Conv→BN→ReLU→Conv→BN→ReLU→MaxPool→Dropout2d],   # block 1 (target)
        ]
        classifier: Sequential[Flatten→Linear→...]
    """
    last_block = model.conv_blocks[-1]         # 最后一个 Sequential
    # Block 内部的倒数第 4 个是最后一个 Conv2d（结构：Conv→BN→ReLU→Conv→BN→ReLU→MaxPool→Dropout2d）
    # 索引 -5 是 Conv2d, -4 是 BN, -3 是 ReLU
    return last_block[-5]  # 最后一个 Conv2d


def get_resnet_target_layer(model):
    """
    获取 ResNet 的 Grad-CAM 目标层（layer3 最后一个 BasicBlock 的 conv2）
    ResNet 结构（默认 num_blocks=[2,2,2]）：
        conv1 → layer1(2 blocks) → layer2(2 blocks) → layer3(2 blocks) → avgpool → fc
    """
    return model.layer3[-1].conv2


# ============================================================
# 特征图可视化
# ============================================================

def get_feature_maps(model, x, layer_names=None):
    """
    提取指定层的中间特征图

    参数：
        model:       nn.Module（需要注册 forward hook）
        x:           输入张量
        layer_names: 层名称列表，如 ['conv1', 'layer1', 'layer2', 'layer3']
                     如果为 None，则使用默认 CNN 层名

    返回：
        dict: {layer_name: feature_map_tensor}
    """
    features = {}

    def hook_fn(name):
        def fn(module, input, output):
            features[name] = output.detach().cpu()
        return fn

    handles = []
    for name, module in model.named_modules():
        if layer_names and name in layer_names:
            handles.append(module.register_forward_hook(hook_fn(name)))

    model.eval()
    with torch.no_grad():
        _ = model(x)

    for h in handles:
        h.remove()

    return features


# ============================================================
# 可视化函数
# ============================================================

def plot_gradcam(original_img, heatmap, pred_class=None, target_class=None,
                 ax=None, alpha=0.5, cmap='jet', title=None):
    """
    在图片上叠加 Grad-CAM 热力图

    参数：
        original_img: numpy array, shape=(H, W) 灰度图（值域 [0, 1] 或 [0, 255]）
        heatmap:      numpy array, shape=(H, W)，值域 [0, 1]
        pred_class:   模型预测类别
        target_class: 目标类别
        ax:           matplotlib axis（None 时自动创建）
        alpha:        热力图透明度
        cmap:         热力图颜色映射
        title:        自定义标题

    返回：
        ax
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))

    # 显示原图
    ax.imshow(original_img, cmap='gray')

    # 叠加热力图
    ax.imshow(heatmap, cmap=cmap, alpha=alpha, interpolation='bilinear')

    # 标题
    if title:
        ax.set_title(title, fontsize=12)
    elif pred_class is not None:
        ax.set_title(f'Pred: {pred_class}', fontsize=14)

    ax.axis('off')
    return ax


def plot_multi_gradcam(images, heatmaps, labels=None, preds=None, ncols=4,
                       alpha=0.5, cmap='jet', figsize=None):
    """
    多张图片的 Grad-CAM 对比图

    参数：
        images:   list of numpy arrays, 每张灰度图
        heatmaps: list of numpy arrays, 对应的热力图
        labels:   真实标签列表
        preds:    预测类别列表
        ncols:    每行显示数量
    """
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (3 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n):
        ax = axes[i // ncols][i % ncols]
        ax.imshow(images[i], cmap='gray')
        ax.imshow(heatmaps[i], cmap=cmap, alpha=alpha, interpolation='bilinear')

        title_parts = []
        if labels is not None:
            title_parts.append(f'True: {labels[i]}')
        if preds is not None:
            title_parts.append(f'Pred: {preds[i]}')
        if title_parts:
            ax.set_title(' | '.join(title_parts), fontsize=10)
        ax.axis('off')

    # 隐藏多余子图
    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis('off')

    plt.tight_layout()
    return fig


def plot_feature_maps(feature_maps, layer_name, ncols=8, figsize=None):
    """
    可视化某一层的特征图（选取前 N 个通道）

    参数：
        feature_maps: dict，key=层名，value=特征图张量 (1, C, H, W)
        layer_name:   要可视化的层名
        ncols:        每行显示数量
        figsize:      图像尺寸
    """
    if layer_name not in feature_maps:
        raise KeyError(f"层 '{layer_name}' 不在 feature_maps 中。可用层: {list(feature_maps.keys())}")

    fm = feature_maps[layer_name]  # (1, C, H, W)
    n_channels = fm.shape[1]
    display_channels = min(n_channels, ncols * 4)  # 最多显示 ncols*4 个通道
    nrows = (display_channels + ncols - 1) // ncols

    if figsize is None:
        figsize = (2 * ncols, 2 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(display_channels):
        ax = axes[i // ncols][i % ncols]
        channel_img = fm[0, i].numpy()
        ax.imshow(channel_img, cmap='viridis')
        ax.set_title(f'Ch {i}', fontsize=8)
        ax.axis('off')

    for i in range(display_channels, nrows * ncols):
        axes[i // ncols][i % ncols].axis('off')

    fig.suptitle(f'特征图: {layer_name} ({n_channels} channels, 显示前 {display_channels})',
                 fontsize=12)
    plt.tight_layout()
    return fig


# ============================================================
# 工具：将灰度图转为 PIL Image 用于显示
# ============================================================

def tensor_to_img(tensor):
    """
    将 PyTorch 张量转为 numpy 灰度图

    参数：
        tensor: shape=(1, H, W) 或 (H, W)，值域任意

    返回：
        numpy array, shape=(H, W)，值域 [0, 1]
    """
    img = tensor.squeeze().detach().cpu().numpy()
    # 归一化到 [0, 1]
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return img


def apply_colormap(heatmap, cmap='jet'):
    """
    将单通道热力图转为 RGB 图像（用于导出/合成）

    参数：
        heatmap: numpy array, shape=(H, W), 值域 [0, 1]
        cmap:    颜色映射名称

    返回：
        numpy array, shape=(H, W, 3), 值域 [0, 255], dtype=uint8
    """
    cm = colormaps.get_cmap(cmap)
    colored = cm(heatmap)[:, :, :3]  # (H, W, 4) → (H, W, 3), 去掉 alpha
    return (colored * 255).astype(np.uint8)


def overlay_heatmap(img, heatmap, alpha=0.5, cmap='jet'):
    """
    将热力图叠加到灰度图上，输出 RGB 图像

    参数：
        img:     numpy array, shape=(H, W), 灰度图
        heatmap: numpy array, shape=(H, W), 值域 [0, 1]
        alpha:   热力图透明度
        cmap:    热力图颜色映射

    返回：
        numpy array, shape=(H, W, 3), 值域 [0, 255], dtype=uint8
    """
    # 灰度图转 RGB
    img_rgb = np.stack([img, img, img], axis=-1)
    if img_rgb.max() <= 1.0:
        img_rgb = (img_rgb * 255).astype(np.uint8)

    # 热力图着色
    heatmap_colored = apply_colormap(heatmap, cmap)

    # 混合
    blended = (img_rgb * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    return blended


# ============================================================
# 模型加载辅助
# ============================================================

def load_cnn_model(weights_path='../models/cnn/best.pth', device=None):
    """
    加载已训练的 SimpleCNN 模型

    注意：需要当前 namespace 中已有 SimpleCNN 类定义
    """
    import sys
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # SimpleCNN 的定义需要从 notebook 环境引入
    # 这里不做 import，由 notebook 调用时自行处理
    raise NotImplementedError(
        "请在 notebook 中先定义 SimpleCNN 类，然后直接使用：\n"
        "model = SimpleCNN(...).to(device)\n"
        "model.load_state_dict(torch.load(weights_path, map_location=device))\n"
        "# 然后: gradcam = GradCAM(model, get_cnn_target_layer(model))"
    )
