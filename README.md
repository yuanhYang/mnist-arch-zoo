# MLP_MNIST —— 深度学习入门：手写数字识别

> 一个面向**零基础初学者**的 PyTorch 实战项目。用 MNIST 手写数字数据集，对比 **MLP / CNN / ResNet / ViT / LSTM** 五种架构，并通过**知识蒸馏**让 MLP 学生"学"到 ResNet 教师的暗知识。

---

## 目录

- [这是什么项目？](#这是什么项目)
- [先备知识：你需要知道的 5 个概念](#先备知识你需要知道的-5-个概念)
- [项目结构](#项目结构)
- [入门指引：从零开始跑起来](#入门指引从零开始跑起来)
- [学习路线图](#学习路线图)
- [网络结构详解](#网络结构详解)
- [PyTorch 核心概念速查](#pytorch-核心概念速查)
- [五种架构：演进与对比](#五种架构演进与对比)
- [知识蒸馏](#知识蒸馏)
- [常见问题排查](#常见问题排查)
- [参考资料](#参考资料)

---

## 这是什么项目？

这个项目会用 PyTorch 一步步带你跑通**完整的深度学习流水线**：

```
数据加载 → 模型定义 → 前向传播 → 损失计算 → 反向传播 → 参数更新 → 模型评估
```

**你会学到什么？**

| 技能 | 在哪里学 |
|------|----------|
| 用 `fetch_openml` 下载并预处理 MNIST 数据集 | `01_data_exploration.ipynb` |
| 从零定义 MLP 网络（`nn.Module`）并理解每层作用 | `02_mlp_training.ipynb` |
| 写训练循环：zero_grad → forward → loss → backward → step | `02_mlp_training.ipynb` |
| 用 NumPy 纯手写 MLP：前向传播、反向传播、梯度下降 | `02_mlp_training_numpy.ipynb` |
| 用混淆矩阵和错误样本分析模型表现 | `03_evaluation.ipynb` |
| 多模型推理对比，交互式手写识别 | `04_predict_custom.ipynb` |
| 理解 CNN 的卷积/池化，ResNet 的残差连接 | `05_cnn_mnist.ipynb`、`06_resnet_mnist.ipynb` |
| 体验 ViT（Transformer 做视觉）和 LSTM（序列建模） | `07_vit_mnist.ipynb`、`08_lstm_mnist.ipynb` |
| 实践知识蒸馏：大模型教小模型 | `09_knowledge_distillation.ipynb` |

---

## 先备知识：你需要知道的 5 个概念

如果你是第一次接触深度学习，花 5 分钟看这 5 个概念就够了：

### 1. 什么是神经网络？

神经网络是一堆数学公式的堆叠。最简单的形式：

```
输入(x) → 线性变换(Wx+b) → 激活函数(ReLU) → 输出(y)
```

- **W（权重）**：网络要学习的参数，一开始随机，训练后逐渐变好
- **b（偏置）**：额外的可学习参数
- **ReLU 激活函数**：`max(0, x)`，把负数变成 0。没有它，100 层网络等于 1 层

### 2. 什么是"训练"？

训练 = 反复做 5 件事：

```
① 清零梯度 → ② 前向传播(算输出) → ③ 算损失(输出和正确答案差多远)
    → ④ 反向传播(算每个参数对损失的贡献) → ⑤ 更新参数(让损失变小)
```

循环几万次后，模型就学会了。

### 3. 损失函数 (Loss Function)

衡量"模型输出"和"正确答案"之间的差距。分类任务用 **交叉熵损失 (CrossEntropyLoss)**：

```python
# 模型预测: [0.1, 0.8, 0.1]  → 正确答案: 1
# 损失很小(0.22)：模型对了
# 模型预测: [0.8, 0.1, 0.1]  → 正确答案: 1
# 损失很大(2.30)：模型错了
```

### 4. 过拟合 (Overfitting)

训练集准确率 99.9%，测试集只有 98.6%？这就是过拟合——模型"背诵"了训练数据，但遇到新数据就不行了。解决方案：Dropout、BatchNorm、数据增强。

### 5. 梯度下降

想象你蒙着眼站在山顶，想走到山谷最低点。每次你感受脚下的坡度（梯度），朝最陡的方向走一小步（学习率）。重复几千次，就到了谷底。

**优化器 (Adam/SGD)** 就是帮你"下山"的算法。

---

## 项目结构

```
MLP_MNIST/
├── data/                              # MNIST 数据集缓存（自动下载，无需手动准备）
├── models/                            # 训练好的模型权重
│   ├── mlp/                           #   全连接网络
│   ├── cnn/                           #   卷积网络
│   ├── resnet/                        #   残差网络（蒸馏的教师模型）
│   ├── vit/                           #   Vision Transformer
│   ├── lstm/                          #   双向 LSTM
│   └── distill/                       #   蒸馏后的学生模型
├── notebooks/
│   ├── 01_data_exploration.ipynb       # 数据加载·预处理·可视化
│   ├── 02_mlp_training.ipynb           #  MLP 训练（入门必看！）
│   ├── 02_mlp_training_numpy.ipynb     #  NumPy 手写 MLP（理解反向传播原理）
│   ├── 03_evaluation.ipynb             #  评估·混淆矩阵·错误分析
│   ├── 04_predict_custom.ipynb         #  手写数字实时识别（五种模型对比）
│   ├── 05_cnn_mnist.ipynb              #  CNN 卷积网络
│   ├── 06_resnet_mnist.ipynb           #  ResNet 残差网络
│   ├── 07_vit_mnist.ipynb              #  ViT Vision Transformer
│   ├── 08_lstm_mnist.ipynb             #  LSTM 序列建模
│   └── 09_knowledge_distillation.ipynb  # 知识蒸馏
├── requirements.txt
└── README.md
```

---

## 入门指引：从零开始跑起来

### 第一步：安装环境

需要 **Python 3.10+**。打开终端（Windows 用 cmd 或 PowerShell，Mac/Linux 用 Terminal）：

```bash
# 1. 克隆或下载项目到本地
cd mnist-arch-zoo

# 2. （推荐）创建虚拟环境
python -m venv venv

# 3. 激活虚拟环境
#    Windows: venv\Scripts\activate
#    Mac/Linux: source venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt
```

> `requirements.txt` 包含：`torch`, `torchvision`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `pillow`, `pandas`, `jupyter`, `ipympl`

### 第二步：启动 Jupyter Notebook

```bash
cd notebooks
jupyter notebook
```

浏览器会自动打开 Jupyter 界面。

### 第三步：按推荐顺序运行

#### 🟢 入门路线（必做，约 40 分钟）

| 顺序 | Notebook | 做什么 | 你会学到 |
|------|----------|--------|----------|
| 1 | `01_data_exploration.ipynb` | 下载 MNIST，看数据长什么样 | 数据加载、归一化、DataLoader |
| 2 | `02_mlp_training.ipynb` | 定义并训练一个 MLP | **训练循环的完整流程** |
| 2b | `02_mlp_training_numpy.ipynb` | 纯 NumPy 手写 MLP | **反向传播的内部原理** |
| 3 | `03_evaluation.ipynb` | 评估模型，看混淆矩阵 | 模型分析、错误诊断 |
| 4 | `04_predict_custom.ipynb` | 自己写数字让模型识别 | 模型推理、多模型对比 + MODEL_CONFIG 注册表 |

> **运行方式**：在 Jupyter 中打开 notebook，按 `Shift+Enter` 逐个执行单元格。每个 cell 执行完会看到输出结果。

#### 🟡 进阶路线（选做，每种约 20 分钟）

| 顺序 | Notebook | 做什么 |
|------|----------|--------|
| 5 | `05_cnn_mnist.ipynb` | 训练 CNN，对比 MLP 的泛化能力 |
| 6 | `06_resnet_mnist.ipynb` | 训练 ResNet，体验残差连接 |
| 7 | `07_vit_mnist.ipynb` | 训练 ViT，理解 Transformer 做视觉 |
| 8 | `08_lstm_mnist.ipynb` | 训练 LSTM，用序列方式"读"图片 |

> 这些训练 notebook 都**可以独立运行**，不需要按顺序，随便挑你想了解的架构即可。

#### 🔴 高级路线（选做）

| 顺序 | Notebook | 前置条件 |
|------|----------|----------|
| 9 | `09_knowledge_distillation.ipynb` | 需要先训练 ResNet（06） |

### 预期运行效果

运行 `02_mlp_training.ipynb` 后，你应该看到：

```
Epoch [  1/100] Train Loss: 0.3521 | Train Acc: 0.8973 | Test Loss: 0.1523 | Test Acc: 0.9542
Epoch [ 10/100] Train Loss: 0.0512 | Train Acc: 0.9845 | Test Loss: 0.0647 | Test Acc: 0.9802
...
训练完成！最佳测试准确率: 0.9863 (Epoch 87)
```

一个 50 万参数的简单 MLP 就能达到约 **98.6%** 的准确率！

---

## 学习路线图

```
┌─────────────────────────────────────────────────────────────────┐
│                        入门者学习路径                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: 01_data_exploration  ← 理解数据长什么样                 │
│           ↓                                                     │
│  Step 2: 02_mlp_training      ← 理解训练循环（核心！）            │
│           ↓                                                     │
│  Step 2b: 02_mlp_training_numpy ← 纯 NumPy 手写反向传播（理解原理） │
│           ↓                                                     │
│  Step 3: 03_evaluation         ← 学会评估和分析模型               │
│           ↓                                                     │
│  Step 4: 04_predict_custom     ← 体验模型的实际应用               │
│           ↓                                                     │
│  ┌────────────────────────────────────────────────────┐         │
│  │  选择感兴趣的架构深入学习（可跳着学）                 │         │
│  │  05_cnn → 06_resnet → 07_vit → 08_lstm            │         │
│  └────────────────────────────────────────────────────┘         │
│           ↓                                                     │
│  Step 5: 09_knowledge_distillation ← 理解模型压缩和蒸馏          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**建议的学习方式**：
1. 先通读一遍 notebook 的 Markdown 说明文字
2. 按 `Shift+Enter` 逐个执行代码单元格
3. 每个 cell 执行后，看输出结果是否和预期一致
4. 尝试修改超参数（如学习率、隐藏层大小），观察变化
5. 遇到不懂的概念，回来看 README 中的"核心概念速查"

---

## 网络结构详解

### MLP（多层感知机）—— 最基础的前馈网络

```
输入层 (784 维向量)
    │     ← 28×28 图片被"拉直"
    ▼
全连接层 512 + BatchNorm1d + ReLU + Dropout(0.2)
    │     ← 512 个神经元，每个连接前一层所有 784 个输入
    ▼
全连接层 256 + BatchNorm1d + ReLU + Dropout(0.2)
    │     ← 256 个神经元
    ▼
输出层 (10 维) ──► CrossEntropyLoss
         ← 10 个分数，分别对应数字 0~9
```

| 组件 | 作用 | 通俗理解 |
|------|------|----------|
| `nn.Linear(784, 512)` | 线性变换 `y = Wx + b` | 每个输出是 784 个输入的加权求和 |
| `nn.BatchNorm1d(512)` | 批归一化 | 把数据"压"到均值 0 方差 1，让训练更稳定 |
| `nn.ReLU()` | 激活函数 `max(0, x)` | 负值变 0，引入非线性 |
| `nn.Dropout(0.2)` | 随机丢弃 20% 神经元 | 强迫网络不依赖特定路径，防止"死记硬背" |
| `nn.Linear(512, 256)` | 第二隐藏层 | 逐步压缩信息到更紧凑的表示 |
| `nn.Linear(256, 10)` | 输出层 | 10 个数字的原始分数（logits） |
| `CrossEntropyLoss` | 损失函数 | 内置 Softmax + 负对数似然，一步到位 |

### 参数量是怎么算出来的？

"537,354 个参数"不是拍脑袋的数字，而是逐层累加出来的。**全连接层的参数量** = `输入维度 × 输出维度 + 输出维度（偏置）`：

| 层 | 计算 | 参数量 |
|---|---|---|
| `Linear(784, 512)` | 784×512 + 512 = 401,408 + 512 | **401,920** |
| `BatchNorm1d(512)` | 512×2（γ+β） | **1,024** |
| `Linear(512, 256)` | 512×256 + 256 = 131,072 + 256 | **131,328** |
| `BatchNorm1d(256)` | 256×2（γ+β） | **512** |
| `Linear(256, 10)` | 256×10 + 10 = 2,560 + 10 | **2,570** |
| **合计** | | **537,354** |

> **通用公式**：`nn.Linear(in_features, out_features)` 的参数 = `in_features × out_features + out_features`。
> 前面的 `in_features × out_features` 是权重矩阵 W 的参数量，后面的 `+ out_features` 是偏置 b。

Dropout 和 ReLU **没有可学习参数**（它们只是操作，不是层）。BatchNorm 每个特征有两个可学习参数：缩放因子 γ 和偏移量 β。

**关键数字**：总参数量约 **537,354**，预期测试准确率 **97%~98%**。

### CNN（卷积神经网络）—— 用卷积核"看"图片

```
输入 (N, 1, 28, 28)                        ← 保持二维图像结构
    │
    ▼
Conv Block 1: Conv3×3(1→32)×2 → BN → ReLU → MaxPool(2) → Dropout2d(0.3)
    │     输出: (N, 32, 14, 14)
    ▼
Conv Block 2: Conv3×3(32→64)×2 → BN → ReLU → MaxPool(2) → Dropout2d(0.3)
    │     输出: (N, 64, 7, 7)
    ▼
Flatten → Linear(3136, 128) → BN → ReLU → Dropout(0.3)
    │
    ▼
Linear(128, 10) → 输出 logits
```

| 组件 | 作用 | 通俗理解 |
|------|------|----------|
| `Conv2d(in, out, 3, padding=1)` | 3×3 卷积核滑动扫描 | 每个输出像素只看 3×3 局部，提取边缘/纹理 |
| `BatchNorm2d` | 通道维度归一化 | 稳定训练，加速收敛 |
| `MaxPool2d(2)` | 2×2 窗口取最大值 | 尺寸减半，增大感受野，减少计算量 |
| `Dropout2d` | 随机丢弃整个通道 | 比普通 Dropout 更强的正则化，防止过拟合 |
| `Flatten` | 展平为向量 | 卷积特征图 → 全连接层输入 |

**参数量估算**（逐层拆解）：

| 层 | 计算 | 参数量 |
|---|---|---|
| Conv2d(1→32, 3×3) | 1×32×3×3=288(权重) + 32(偏置) | 320 |
| BatchNorm2d(32) | 32×2(γ+β) | 64 |
| Conv2d(32→32, 3×3) | 32×32×3×3=9,216 + 32 | 9,248 |
| BatchNorm2d(32) | 32×2 | 64 |
| **Conv Block 1 小计** | | **9,696** |
| Conv2d(32→64, 3×3) | 32×64×3×3=18,432 + 64 | 18,496 |
| BatchNorm2d(64) | 64×2 | 128 |
| Conv2d(64→64, 3×3) | 64×64×3×3=36,864 + 64 | 36,928 |
| BatchNorm2d(64) | 64×2 | 128 |
| **Conv Block 2 小计** | | **55,680** |
| Linear(3136→128) | 3136×128=401,408(权重) + 128(偏置) | 401,536 |
| BatchNorm1d(128) | 128×2 | 256 |
| **FC 层小计** | | **401,792** |
| Linear(128→10) | 128×10=1,280 + 10 | 1,290 |
| **合计** | | **468,458** |

> **卷积核参数量公式**：`Conv2d(in, out, K×K)` 的参数量 = `in × out × K × K + out`（权重 + 偏置）。
> 为什么 CNN 参数比 MLP 少？因为卷积核在所有位置共享——一个 3×3 卷积核只有 9 个权重，不管图片多大。而 MLP 的 `Linear(784, 512)` 第一层就 40 万个参数。

---

### ResNet（残差网络）—— 加一条"短路"让深层网络可训练

```
输入 (N, 1, 28, 28)
    │
    ▼
初始卷积: Conv3×3(1→32) → BN → ReLU        ← 尺寸不变，通道 1→32
    │
    ▼
Layer1: 2×BasicBlock(32→32, stride=1)       ← 28×28 不变，残差学习
    │
    ▼
Layer2: 2×BasicBlock(32→64, stride=2)       ← 14×14，首个 block 用 1×1 conv 对齐
    │
    ▼
Layer3: 2×BasicBlock(64→128, stride=2)      ← 7×7，首个 block 用 1×1 conv 对齐
    │
    ▼
AdaptiveAvgPool2d(1,1) → Flatten → Linear(128, 10)
```

**BasicBlock 内部结构**：

| 组件 | 作用 |
|------|------|
| 主路 `Conv→BN→ReLU→Conv→BN` | 两层 3×3 卷积学习残差 F(x) |
| 短路（shortcut） | 直接把输入 x 加到主路输出上 |
| 恒等映射 shortcut | 输入输出维度相同时，什么都不做 |
| 1×1 卷积 shortcut | 维度不匹配时，用 1×1 卷积投影对齐 |

**关键设计**：
- **无 Dropout**：只靠 BatchNorm 做正则化，残差连接本身就是一种正则化
- **Kaiming 初始化**：针对 ReLU 设计的权重初始化方法，让梯度在深层网络中稳定传播
- **bias=False**：卷积后紧跟 BN，bias 的效果会被 BN 抵消

**参数量估算**（逐层拆解）：

| 层 | 计算 | 参数量 |
|---|---|---|
| Conv2d(1→32, 3×3, bias=False) | 1×32×3×3=288 | 288 |
| BatchNorm2d(32) | 32×2 | 64 |
| **初始卷积层小计** | | **352** |
| Layer1 Blk1 (32→32, 恒等shortcut) | 32×32×9×2=18,432 + BN×2=128 | 18,560 |
| Layer1 Blk2 (32→32, 恒等shortcut) | 同上 | 18,560 |
| **Layer1 小计** | | **37,120** |
| Layer2 Blk1 (32→64, 1×1 shortcut) | 主路: 55,552 + shortcut: 32×64×1=2,048+BN(64)128 | 57,728 |
| Layer2 Blk2 (64→64, 恒等shortcut) | 64×64×9×2=73,728 + BN×2=256 | 73,984 |
| **Layer2 小计** | | **131,712** |
| Layer3 Blk1 (64→128, 1×1 shortcut) | 主路: 221,696 + shortcut: 64×128×1=8,192+BN(128)256 | 230,144 |
| Layer3 Blk2 (128→128, 恒等shortcut) | 128×128×9×2=294,912 + BN×2=512 | 295,424 |
| **Layer3 小计** | | **525,568** |
| Linear(128→10) | 128×10=1,280 + 10(偏置) | 1,290 |
| **合计** | | **696,042** |

> `bias=False` 是因为卷积后紧跟 BatchNorm，偏置的效果会被 BN 抵消。ResNet 比 MLP 多 ~160K 参数，但准确率高出近 1%。

> **核心直觉**：普通网络学的是 `output = F(x)`，ResNet 学的是 `output = F(x) + x`。最坏情况下 F(x) 学到 0，output 至少等于 x——"不退化"。

---

### ViT（Vision Transformer）—— 把图片当文字"读"

```
输入 (N, 1, 28, 28)
    │
    ▼
Patch Embedding: Conv2d(1, 64, kernel=4, stride=4)
    │     图片切成 7×7=49 个 4×4 patch，每个投影为 64 维向量
    ▼
[CLS] + 49 个 patch tokens → 50 个 token × 64 维
    │     加入可学习的 class token 和位置编码
    ▼
Dropout(0.1)
    │
    ▼
Transformer Encoder × 4 层
    │  Pre-LN → Multi-Head Self-Attention(4 head) → Add
    │  → Pre-LN → FFN(64→128→64, GELU) → Add
    ▼
LayerNorm → 取 [CLS] token → Linear(64, 10)
```

| 组件 | 作用 | 通俗理解 |
|------|------|----------|
| Patch Embedding | 图片切块 + 向量化 | 把图片拆成"单词"，每个 patch 就是一个 token |
| Class Token `[CLS]` | 可学习的分类令牌 | 聚合全局信息，最终用它做分类（类似 BERT） |
| Position Embedding | 位置编码 | 告诉模型每个 patch 在图片的哪个位置 |
| Multi-Head Self-Attention | 每个 patch 关注所有 patch | 全局建模——左上角可以"看到"右下角 |
| FFN（前馈网络） | 两层 MLP + GELU | 对每个 token 独立做非线性变换 |

**关键设计**：
- **Pre-LN**：LayerNorm 放在 Attention/FFN 前面，训练更稳定
- **GELU 激活**：比 ReLU 更平滑，Transformer 标配
- **无卷积归纳偏置**：ViT 不假设"相邻像素有关联"，全靠数据学

**参数量估算**（逐层拆解）：

| 层 | 计算 | 参数量 |
|---|---|---|
| PatchEmbed: Conv2d(1→64, 4×4, stride=4) | 1×64×4×4=1,024(权重) + 64(偏置) | 1,088 |
| Class Token | 1×1×64=64 | 64 |
| Position Embedding | (49+1)×64=3,200 | 3,200 |
| **每层 Transformer Encoder** | | |
| — Multi-Head Self-Attention | in_proj(192×64+192=12,480) + out_proj(64×64+64=4,160) | 16,640 |
| — LayerNorm × 2 | 64×2 ×2 = 256 | 256 |
| — FFN: Linear(64→128) + Linear(128→64) | 8,320 + 8,256 | 16,576 |
| **每层小计** | | **33,472** |
| Transformer Encoder × 4 层 | 33,472 × 4 | 133,888 |
| 最终 LayerNorm | 64×2 | 128 |
| Linear(64→10) | 64×10=640 + 10(偏置) | 650 |
| **合计** | | **139,018** |

> **Self-Attention 参数量公式**：`MultiheadAttention(d_model=64)` 将 Q/K/V 投影合并为一个大矩阵 `in_proj(3×64, 64)`，权重 = `192×64=12,288`，偏置 = `192`。

> **核心直觉**：CNN 靠卷积核"扫"，ViT 靠自注意力"看全局"。小数据集上 CNN 更好（有归纳偏置），大数据集上 ViT 能反超。

---

### LSTM（长短期记忆网络）—— 逐行"阅读"图片

```
输入 (N, 28, 28)                            ← 28 个时间步，每步一行 28 像素
    │
    ▼
双向 LSTM: 2 层 × 128 隐层，层间 Dropout=0.3
    │  LSTM 内部: 遗忘门 + 输入门 + 输出门 控制信息流
    │  正向: 第1行→第28行扫描   反向: 第28行→第1行扫描
    │  拼接: 256 维 (128×2 方向)
    ▼
取最后隐状态: 正向 h_n[-2] 拼接 反向 h_n[-1] → (N, 256)
    │
    ▼
Dropout(0.3) → Linear(256, 64) → ReLU → Dropout(0.3) → Linear(64, 10)
```

| 组件 | 作用 | 通俗理解 |
|------|------|----------|
| 遗忘门 | 决定丢弃哪些旧信息 | "上一行不重要，忘了" |
| 输入门 | 决定存储哪些新信息 | "这一行有关键笔画，记住" |
| 输出门 | 决定输出哪些信息 | "现在看到的内容中，哪些对分类有用" |
| 双向扫描 | 正向+反向各跑一遍 | 既看到"从左到右"也看到"从右到左" |
| 隐状态拼接 | 正反向 128 维拼接为 256 维 | 同时拥有前后文信息 |

**参数量估算**（逐层拆解）：

| 层 | 计算 | 参数量 |
|---|---|---|
| **LSTM 第 1 层（input=28→hidden=128）** | | |
| — 正向：`4×128×(28+128+2)` | `512×158` | 80,896 |
| — 反向：同上 | | 80,896 |
| **LSTM 第 2 层（input=256→hidden=128）** | | |
| — 正向：`4×128×(256+128+2)` | `512×386` | 197,632 |
| — 反向：同上 | | 197,632 |
| **LSTM 层小计** | | **557,056** |
| Linear(256→64) | 256×64=16,384 + 64(偏置) | 16,448 |
| Linear(64→10) | 64×10=640 + 10(偏置) | 650 |
| **合计** | | **574,154** |

> **LSTM 参数量公式**：每层每方向 = `4 × hidden × (input + hidden + 2)`。其中 4 对应遗忘门、输入门、候选记忆、输出门四套权重；`+2` 是每套的偏置。
> 虽然参数量不小（~574K，接近 MLP 的 537K），但大部分参数集中在 4 个门控矩阵中，这些参数是**跨时间步共享**的——28 个时间步用同一套参数。

> **核心直觉**：把 28×28 图片当成 28 行文字逐行阅读，LSTM 能"记住"前几行的笔画信息来判断后续笔画。虽然准确率不如 CNN，但证明了序列模型也能做图像分类。

---

## PyTorch 核心概念速查

如果你第一次用 PyTorch，这些概念需要了解：

| 概念 | 代码 | 一句话解释 |
|------|------|-----------|
| **Tensor** | `torch.tensor([1,2,3])` | 多维数组，能在 GPU 上运算，自动追踪梯度 |
| **nn.Module** | `class MyModel(nn.Module):` | 所有网络的基类，只需实现 `__init__` 和 `forward` |
| **DataLoader** | `DataLoader(dataset, batch_size=64)` | 自动切分数据为 mini-batch，支持打乱和多线程 |
| **自动求导** | `loss.backward()` | 自动计算所有参数的梯度（链式法则），无需手写 |
| **optimizer** | `optim.Adam(model.parameters())` | 根据梯度自动更新参数 |
| **device** | `model.to('cuda')` | 一键将模型/数据迁移到 GPU |
| **state_dict** | `model.state_dict()` | 模型的所有权重（可以保存和加载） |
| **model.train()** | 启用 Dropout 和 BN 的训练行为 | 训练时调用 |
| **model.eval()** | 关闭 Dropout，BN 用全局统计 | 评估/推理时调用 |
| **torch.no_grad()** | 不构建计算图 | 推理时使用，省显存 |

### 训练循环的标准模板

```python
model.train()                          # ① 训练模式
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()              # ② 清零梯度（否则会累加！）
    outputs = model(images)            # ③ 前向传播
    loss = criterion(outputs, labels)  # ④ 计算损失
    loss.backward()                    # ⑤ 反向传播（自动求梯度）
    optimizer.step()                   # ⑥ 更新参数
```

---

### `__init__`、`forward` 和参数更新的区别

初学者最容易混淆的三个概念：

| | `__init__` | `forward` | 参数更新 |
|---|---|---|---|
| **是什么** | 定义模型结构（搭积木） | 定义数据流向（怎么走） | 调节参数（让模型变好） |
| **做什么** | 创建层对象，注册为 `nn.Parameter` | 描述输入 → 输出经过哪些层的顺序 | `W = W - lr × ∂loss/∂W` |
| **是否修改参数** | ❌ 只初始化，不修改 | ❌ 纯计算，不修改 | ✅ 这里才真正改变参数 |
| **调用时机** | 实例化时执行一次 | 每次 `model(x)` 自动调用 | 每个 batch 执行一次 |

```python
# __init__: 定义"有哪些可训练的参数"（W, b, γ, β...）
model = MLP()              # 只执行一次，所有权重随机初始化

# forward: 定义"数据怎么算"
output = model(x)          # 每批数据都执行，纯计算，不修改参数

# 参数更新：在 forward 之外，由 optimizer 完成
loss = criterion(output, y)   # ① 算损失
optimizer.zero_grad()         # ② 清空旧梯度
loss.backward()               # ③ 反向传播：算出每个参数的梯度 ∂loss/∂W
optimizer.step()              # ④ 更新参数：W = W - lr × ∂loss/∂W
```

> **记忆技巧**：`__init__` 搭水管，`forward` 让水流，`loss.backward()` + `optimizer.step()` 调阀门。只有调阀门才改变水流大小。

---

### 常用层详解

PyTorch 提供了丰富的 `nn` 模块，理解每一层的作用、输入输出形状和参数量是搭建网络的基础。以下按功能分类介绍本项目用到的所有层。

#### 一、线性层

##### `nn.Linear(in_features, out_features, bias=True)`

全连接层，`y = xW^T + b`。**最基础也是最常用的层。**

| 参数 | 含义 |
|---|---|
| `in_features` | 输入维度 |
| `out_features` | 输出维度 |
| `bias` | 是否学偏置（默认 True） |

```python
nn.Linear(784, 512)    # 输入 784 维 → 输出 512 维
```

| 输入形状 | 输出形状 | 参数量 |
|---|---|---|
| `(N, in_features)` | `(N, out_features)` | `in_features × out_features + out_features` |

> `(N, *)` 也支持，`Linear` 只对最后一维做变换。比如 `(N, 28, 784)` 也能过 `Linear(784, 512)`，输出 `(N, 28, 512)`。

---

#### 二、卷积层

##### `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)`

二维卷积，用 `out_channels` 个 `kernel_size × kernel_size` 的卷积核在输入上滑动，提取局部特征。

| 参数 | 含义 | 常用值 |
|---|---|---|
| `in_channels` | 输入通道数 | 灰度图=1，中间层=32/64/128 |
| `out_channels` | 输出通道数（=卷积核个数） | 逐层翻倍：32→64→128 |
| `kernel_size` | 卷积核大小 | 3（最常用），1（1×1 投影），5/7 |
| `stride` | 滑动步长 | 1（保持尺寸），2（尺寸减半） |
| `padding` | 边缘补零圈数 | `kernel//2`（same padding，保持尺寸） |
| `bias` | 是否学偏置 | 跟 BN 连用时设为 `False`（BN 会抵消偏置） |

```python
nn.Conv2d(1, 32, 3, padding=1)              # 32 个 3×3 卷积核，same padding
nn.Conv2d(64, 128, 1, bias=False)           # 1×1 卷积，只做通道投影
nn.Conv2d(1, 64, 4, stride=4)               # 步长 4，用于 ViT patch 切分
```

| 输入形状 | 输出形状 | 参数量 |
|---|---|---|
| `(N, C_in, H, W)` | `(N, C_out, H_out, W_out)` | `C_in × C_out × K × K + C_out` |

> 输出尺寸公式：`H_out = (H + 2×padding - kernel) / stride + 1`（向下取整）。

---

#### 三、归一化层

##### `nn.BatchNorm1d(num_features)`

对 `(N, D)` 的每一维做归一化：减去当前 batch 的均值，除以标准差，再缩放平移。**让每层输入的分布稳定，加速收敛。**

```python
nn.BatchNorm1d(256)     # 256 维特征，每个特征独立归一化
```

| 输入形状 | 输出形状 | 参数量 | 训练/推理行为 |
|---|---|---|---|
| `(N, D)` | `(N, D)` | `D × 2`（γ+β） | 训练：用 batch 统计；推理：用全局滑动平均 |

##### `nn.BatchNorm2d(num_features)`

同 `BatchNorm1d`，但对 `(N, C, H, W)` 的每个通道独立归一化。

```python
nn.BatchNorm2d(64)      # 64 通道，每个通道独立归一化
```

| 输入形状 | 输出形状 | 参数量 |
|---|---|---|
| `(N, C, H, W)` | `(N, C, H, W)` | `C × 2` |

> **重要**：`model.train()` 时 BN 用当前 batch 的均值/方差并更新滑动平均；`model.eval()` 时用训练期累积的全局统计量。切换模式是必须的。

##### `nn.LayerNorm(normalized_shape)`

对**每个样本内部**做归一化，不依赖 batch。Transformer 标配（不依赖 batch 统计，推理时行为一致）。

```python
nn.LayerNorm(64)        # 对最后一维 64 做归一化
```

| 输入形状 | 输出形状 | 参数量 |
|---|---|---|
| `(N, ..., D)` | 同输入 | `normalized_shape × 2` |

> **BN vs LN**：BN 沿 batch 维度归一化（CNN 常用），LN 沿特征维度归一化（Transformer 常用）。BN 在小 batch 时不稳定，LN 不受 batch 大小影响。

---

#### 四、正则化层

##### `nn.Dropout(p=0.5)`

训练时每次前向传播**随机将 `p` 比例的元素置零**，其余元素放大 `1/(1-p)` 倍。推理时不做任何操作。

```python
nn.Dropout(0.3)         # 每次前向随机丢弃 30% 的神经元
```

- **参数量**：0（无学习参数）
- **为什么有效**：强迫网络不依赖某一个神经元，相当于每次训练一个"子网络"，集成学习效果
- **放在哪**：通常跟在激活函数后面

##### `nn.Dropout2d(p=0.5)`

和 `Dropout` 类似，但**按整个通道丢弃**（而非单个元素），对卷积特征图更有效。

```python
nn.Dropout2d(0.3)       # 随机丢弃 30% 的通道
```

---

#### 五、激活函数

##### `nn.ReLU()`

ReLU（Rectified Linear Unit）：`f(x) = max(0, x)`。最常用的激活函数。

```python
nn.ReLU()               # 负值→0，正值→原样输出
```

- **参数量**：0
- **优点**：计算快，缓解梯度消失（正区间梯度恒为 1）
- **缺点**：负区间梯度为 0，"死神经元"问题（一旦进入负区间就不更新了）

##### `nn.GELU()`

GELU（Gaussian Error Linear Unit）：比 ReLU 更平滑，Transformer 标配。公式近似为 `x × Φ(x)`（Φ 是高斯累积分布）。

```python
# 在 TransformerEncoderLayer 中以字符串形式使用
nn.TransformerEncoderLayer(d_model=64, nhead=4, activation='gelu', ...)
```

- **参数量**：0
- **ReLU vs GELU**：ReLU 一刀切，GELU 平滑过渡。小负值不会被完全压为 0。

---

#### 六、池化层

##### `nn.MaxPool2d(kernel_size, stride=None)`

在 `kernel_size × kernel_size` 窗口内取最大值。**降采样、增大感受野。**

```python
nn.MaxPool2d(2)         # 2×2 窗口取最大值，默认 stride=kernel_size=2，尺寸减半
```

| 输入形状 | 输出形状 | 参数量 |
|---|---|---|
| `(N, C, H, W)` | `(N, C, H/2, W/2)` | 0 |

##### `nn.AdaptiveAvgPool2d(output_size)`

**自适应**平均池化：无论输入多大，输出固定大小。ResNet 用 `(1,1)` 把特征图池化到一个点。

```python
nn.AdaptiveAvgPool2d((1, 1))   # 任意大小特征图 → (N, C, 1, 1)
```

- **参数量**：0
- **为什么用这个**：不需要手动计算 `in_features`，改输入尺寸也不用改代码。同时全局平均池化有正则化效果。

---

#### 七、循环层

##### `nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=False, bidirectional=False, dropout=0)`

长短期记忆网络。通过遗忘门、输入门、输出门控制信息流，解决普通 RNN 的长期依赖问题。

| 参数 | 含义 |
|---|---|
| `input_size` | 每个时间步的特征维度 |
| `hidden_size` | 隐状态维度 |
| `num_layers` | 堆叠层数（≥2 时层间可用 dropout） |
| `batch_first` | `True` 时输入为 `(B, seq, feat)`，强烈建议设为 True |
| `bidirectional` | 是否双向（输出维度翻倍） |

```python
nn.LSTM(input_size=28, hidden_size=128, num_layers=2,
        batch_first=True, bidirectional=True, dropout=0.3)
```

| 输入形状 | 输出形状 | 参数量公式 |
|---|---|---|
| `(B, L, input_size)` | `output: (B, L, D×hidden)` | 每层每方向：`4 × hidden × (input + hidden + 2)` |

> 参数量中的 `4` 对应 4 个门（遗忘、输入、候选记忆、输出），`+2` 是每套的偏置。参数量虽然大，但**跨时间步共享**。

---

#### 八、Transformer 层

##### `nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu', batch_first=False, norm_first=False)`

单个 Transformer 编码器层 = Self-Attention + FFN，带残差连接和 LayerNorm。

| 参数 | 含义 |
|---|---|
| `d_model` | 模型维度（= token 的向量长度） |
| `nhead` | 多头注意力头数（必须能整除 d_model） |
| `dim_feedforward` | FFN 隐藏层维度（通常 `d_model × 2` 或 `×4`） |
| `batch_first` | 建议设为 True，输入 `(B, seq, d_model)` |
| `norm_first` | True=Pre-LN（LN 在前），False=Post-LN。Pre-LN 训练更稳定 |

```python
nn.TransformerEncoderLayer(
    d_model=64, nhead=4, dim_feedforward=128,
    dropout=0.1, activation='gelu', batch_first=True, norm_first=True
)
```

**内部结构**（Pre-LN 模式）：
```
x → LN → MultiHeadSelfAttention → +x(残差) → LN → FFN(GELU) → +x(残差) → out
```

##### `nn.TransformerEncoder(encoder_layer, num_layers)`

将多个 `TransformerEncoderLayer` 堆叠成完整编码器。

```python
encoder_layer = nn.TransformerEncoderLayer(d_model=64, ...)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
```

---

#### 九、容器层

##### `nn.Sequential(*layers)`

按顺序执行各层，输入流经每一层后自动传给下一层。

```python
self.net = nn.Sequential(
    nn.Linear(784, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 10)
)
# 等价于: x = dropout(relu(bn(linear1(x)))); x = linear2(x)
```

- **适用场景**：层是线性管道（无分支/跳跃连接）
- **不适用场景**：ResNet（有 shortcut）、多输入多输出

##### `nn.ModuleList(modules)`

像 Python 列表一样存多个子模块，但**会被 PyTorch 自动识别**（参数会被追踪）。

```python
self.conv_blocks = nn.ModuleList()
for out_ch in [32, 64]:
    self.conv_blocks.append(nn.Sequential(...))
# 前向中手动遍历: for block in self.conv_blocks: x = block(x)
```

- **vs `Sequential`**：`ModuleList` 不会自动前向，需要手动在 `forward` 里写循环
- **vs Python `list`**：`ModuleList` 里的模块会被注册到 `model.parameters()` 中

---

#### 十、形状变换层

##### `nn.Flatten(start_dim=1)`

把多维张量展平，默认保留第 0 维（batch）。

```python
nn.Flatten()            # (N, 64, 7, 7) → (N, 3136)
```

- **参数量**：0
- **等价于** `x.view(x.size(0), -1)`，但作为 `nn.Module` 可以放进 `Sequential`

---

#### 速查总表

| 层 | 参数量 | 典型输入 | 典型输出 | 何时用 |
|---|---|---|---|---|
| `Linear(d_in, d_out)` | `d_in×d_out + d_out` | `(N, d_in)` | `(N, d_out)` | 全连接，做分类/回归 |
| `Conv2d(C_in, C_out, 3)` | `C_in×C_out×9 + C_out` | `(N, C_in, H, W)` | `(N, C_out, H, W)` | 提取空间特征 |
| `BatchNorm1d(D)` | `2D` | `(N, D)` | `(N, D)` | 全连接后稳定训练 |
| `BatchNorm2d(C)` | `2C` | `(N, C, H, W)` | `(N, C, H, W)` | 卷积后稳定训练 |
| `LayerNorm(D)` | `2D` | `(N, ..., D)` | 同输入 | Transformer/不依赖 batch |
| `Dropout(p)` | 0 | 任意 | 同输入 | 防过拟合 |
| `Dropout2d(p)` | 0 | `(N, C, H, W)` | 同输入 | 卷积特征图防过拟合 |
| `ReLU()` | 0 | 任意 | 同输入 | 引入非线性 |
| `MaxPool2d(2)` | 0 | `(N, C, H, W)` | `(N, C, H/2, W/2)` | 降采样 |
| `AdaptiveAvgPool2d((1,1))` | 0 | `(N, C, H, W)` | `(N, C, 1, 1)` | 全局池化→FC 输入 |
| `LSTM(28, 128, 2, bidir=True)` | `4×128×(28+128+2)×2×2` | `(B, 28, 28)` | `(B, 28, 256)` | 序列建模 |
| `Flatten()` | 0 | `(N, C, H, W)` | `(N, C×H×W)` | 卷积→全连接前的展平 |
| `Sequential(...)` | 看内部 | — | — | 无分支的线形管道 |
| `ModuleList([...])` | 看内部 | — | — | 需要手动遍历的模块列表 |

---

## 五种架构：演进与对比

### 为什么 MLP 不够好？

MLP 把 28×28 图片展平为 784 维向量，**完全丢失了空间结构**：

| 问题 | 表现 |
|------|------|
| **丢失空间关系** | 相邻像素和远距离像素在展平后"一视同仁" |
| **无平移不变性** | 数字"5"移动几个像素，MLP 的输出就会剧烈变化 |
| **参数冗余** | 53 万参数全连接，大量参数花在记忆位置而非特征 |

证据：**train acc ~99.9%，test acc 仅 ~98.6%**，train-test gap 高达 1.3%——明显过拟合。

### 五种架构对比表

| 特性 | MLP | CNN | ResNet | ViT | LSTM |
|------|-----|-----|--------|-----|------|
| **核心思想** | 全连接层堆叠 | 卷积 + 池化 | 卷积 + 残差连接 | 自注意力 + Transformer | 门控循环单元 |
| **空间归纳偏置** | 无 | 平移不变性 | 同 CNN + 恒等映射 | 几乎无 | 逐行时序扫描 |
| **输入格式** | `(784,)` 向量 | `(1,28,28)` 张量 | `(1,28,28)` 张量 | `(1,28,28)→patches` | `(28,28)` 序列 |
| **参数量** | ~537K | ~468K | ~696K | ~280K | ~85K |
| **过拟合倾向** | 最高 | 低 | 最低 | 小数据上偏高 | 中等 |
| **预期准确率** | ~98.6% | ~99.3%+ | ~99.5%+ | ~98.5%~99% | ~98.5% |
| **训练速度** | 最快 | 快 | 中等 | 慢 | 中等 |

### 各架构一句话总结

- **CNN**：用 3×3 小卷积核滑动扫描图片，天然提取边缘纹理，参数共享极大减少参数量
- **ResNet**：在 CNN 基础上加"跳跃连接"（`output = F(x) + x`），让 100 层网络也能训练
- **ViT**：把图片切成 16×16 的小块（patches），当作文本 token 扔进 Transformer，用自注意力全局建模
- **LSTM**：把 28×28 图片当成 28 个时间步的序列，逐行"阅读"，用门控机制控制信息流

### CNN 核心机制详解

```
输入图片 (1×28×28)
    │
    ▼
Conv2d(1→32, kernel=3)  ← 32个 3×3 卷积核，提取 32 种特征图
    │  每个卷积核在图片上滑动，每次计算 3×3=9 个像素的加权和
    ▼
BatchNorm2d + ReLU
    │
    ▼
MaxPool2d(2)  ← 2×2 窗口取最大值，尺寸减半 (14×14)
    │
    ▼
Conv2d(32→64, kernel=3)  ← 64个卷积核，提取更高层特征
    │
    ▼
MaxPool2d(2)  ← 尺寸再减半 (7×7)
    │
    ▼
Flatten → Linear(64×7×7, 128) → Linear(128, 10)
```

**为什么 CNN 比 MLP 好？**
- 卷积核只关注 3×3 局部区域 → 天然提取边缘、纹理
- 参数共享：同一个卷积核扫过整张图 → 平移不变性
- MaxPooling 降采样 → 减少计算量，增大感受野

### ResNet 核心机制详解

```
普通网络:  x → Conv → BN → ReLU → Conv → BN → ReLU → 输出
                   梯度传到这里已经很小了（梯度消失）

ResNet:    x → Conv → BN → ReLU → Conv → BN → (+) → ReLU → 输出
                    ↑                           ↑
                    └─── 跳跃连接（shortcut）─────┘
                   梯度可以"抄近路"直达浅层，不会消失
```

公式：`output = ReLU( F(x) + x )`，其中 `F(x)` 是两层卷积的输出。当 `F(x)=0` 时，output=x，相当于什么都没做——这让网络可以"选择"跳过不重要的层。

### ViT 核心机制详解

```
28×28 图片
    │ 切分为 7×7=49 个 4×4 的小块(patches)
    ▼
[P1] [P2] [P3] ... [P49]    ← 每个 patch 投影为 64 维向量
    │
    ▼ 加上 class_token（特殊的分类令牌）和位置编码
[CLS] [P1] [P2] ... [P49]
    │
    ▼ 送入 4 层 Transformer Encoder
    │  每层: 自注意力（每个patch关注所有patch）+ 前馈网络
    ▼
取 [CLS] 的输出 → Linear(64, 10) → 分类结果
```

**自注意力的核心**：每个 patch 计算自己和其他所有 patch 的相似度，然后加权聚合信息。这让 ViT 能看到全局关系，但也需要更多数据来学习。

### LSTM 核心机制详解

```
     时间步1      时间步2           时间步28
    ┌────────┐  ┌────────┐       ┌────────┐
    │ 第 1 行 │  │ 第 2 行 │  ...  │ 第28行 │
    │ 28 像素 │  │ 28 像素 │       │ 28 像素 │
    └───┬────┘  └───┬────┘       └───┬────┘
        │   ┌──────┘                │
        ▼   ▼                       ▼
    ┌──────────────────────────────────┐
    │       双向 LSTM (2层×128)        │
    │  前向: 从左到右扫描               │
    │  后向: 从右到左扫描               │
    │  拼接: 256维隐状态               │
    └──────────────┬───────────────────┘
                   ▼
            FC(256→64→10)
```

**三个门控**：
- **遗忘门**：决定丢弃哪些旧信息
- **输入门**：决定存储哪些新信息
- **输出门**：决定输出哪些信息

---

## 知识蒸馏

### 核心思想

让一个大的"教师"模型（ResNet，99.5%准确率）教一个小的"学生"模型（MLP，537K参数）。

**传统训练 vs 蒸馏训练**：

```
传统训练:
  图片 → MLP学生 → [0.01, 0.02, 0.90, ...] → 对比真实标签"2" → 只知道"这是2"

蒸馏训练:
  图片 → ResNet教师 → [0.00, 0.05, 0.85, 0.03, 0.07, ...]
                            ↓ 软标签：不仅告诉学生"这是2"，
                            ↓ 还告诉学生"有点像3"，"有点像5"
  图片 → MLP学生   → [0.02, 0.08, 0.75, 0.06, 0.09, ...]
                            ↓ 学生同时学习: ①真实标签 ②教师的软标签
```

### 蒸馏损失公式

```
Loss = α × CrossEntropy(学生输出, 真实标签)         ← 硬标签损失
     + (1-α) × T² × KL(softmax(学生/T), softmax(教师/T))  ← 软标签损失

其中:
  α: 硬标签权重（通常 0.5~0.9）
  T: 温度（通常 2~10）
  T²: 缩放因子，保持梯度大小不变
```

### 温度 T 的作用

```
T=1:  softmax → [0.001, 0.002, 0.990, ...]   ← 输出很"尖锐"，接近 one-hot
T=5:  softmax → [0.02,  0.05,  0.80,  ...]   ← 输出变"平滑"
T=10: softmax → [0.05,  0.08,  0.50,  ...]   ← 更平滑，"暗知识"更明显
```

T 越高，教师输出的概率分布越"平滑"，类别间的相似性信息（暗知识）越容易被学生学到。

---

## 常见问题排查

| 问题 | 可能原因 | 解决方法 |
|------|----------|----------|
| `ModuleNotFoundError: No module named 'torch'` | PyTorch 未安装 | `pip install torch torchvision` |
| `FileNotFoundError: ../models/xxx/best.pth` | 模型未训练 | 先运行对应的训练 notebook |
| 训练很慢（每个 epoch 超过 30 秒） | 没有 GPU | 正常，CPU 也能跑，只是慢一点 |
| GPU 显存不足 (CUDA out of memory) | batch_size 太大 | 减小 `BATCH_SIZE` 到 32 或 16 |
| `load_state_dict` 报错 size mismatch | 模型结构与保存时不符 | 确保模型定义代码完全一致 |
| 测试准确率很低（<90%） | 训练不够或数据未归一化 | 增加 epochs 或检查像素是否除以 255 |
| `.pth` 文件不存在，显示 SKIP | 该模型尚未训练 | 正常现象，评估 notebook 会自动跳过 |

### 环境问题快速诊断

```bash
# 检查 Python 版本（需要 3.8+）
python --version

# 检查 PyTorch 是否安装并支持 GPU
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 参考资料

### 入门教程
- [PyTorch 官方 60 分钟教程](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) — 强烈推荐先看这个
- [动手学深度学习 (D2L)](https://d2l.ai/) — 李沐的经典教材，有中文版

### 论文
- [Batch Normalization (2015)](https://arxiv.org/abs/1502.03167) — BN 的原始论文
- [Dropout (2014)](https://jmlr.org/papers/v15/srivastava14a.html) — 理解为什么随机丢弃能防过拟合
- [ResNet (2015)](https://arxiv.org/abs/1512.03385) — 残差连接的里程碑工作
- [ViT (2020)](https://arxiv.org/abs/2010.11929) — "一张图值 16×16 个词"
- [LSTM (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf) — 经典中的经典
- [知识蒸馏 (2015)](https://arxiv.org/abs/1503.02531) — Hinton 的蒸馏论文

### 数据集
- [MNIST 数据集](https://www.openml.org/search?type=data&status=active&id=554) — 手写数字识别的"Hello World"

## NumPy 手写 MLP 教学（理解反向传播"黑盒"内部）

PyTorch 的 `loss.backward()` 太方便了，初学者往往不知道它背后发生了什么。

`02_mlp_training_numpy.ipynb` 用**纯 NumPy** 手写了一个 MLP，逐步展示：

```
前向传播 → 交叉熵损失 → 链式法则逐层求梯度 → 小批量 SGD 更新
```

| 操作 | NumPy 手动 | PyTorch 自动 |
|------|-----------|-------------|
| 前向传播 | 手写矩阵乘法 + ReLU + Softmax | `model(x)` |
| 损失 | 手写 `-sum(y*log(p))/N` | `criterion(output, label)` |
| 反向传播 | 逐层手推链式法则 | `loss.backward()` |
| 参数更新 | `W -= lr * dW` | `optimizer.step()` |

跑完这个 notebook 后，你会对**梯度是什么、怎么算、怎么用**有清晰的认识。强烈推荐在 `02_mlp_training.ipynb` 之后运行。
