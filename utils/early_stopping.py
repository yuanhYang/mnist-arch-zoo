"""
EarlyStopping + Checkpoint 工具类
=================================
可在任何训练脚本中直接导入使用。

使用方式:
    from utils.early_stopping import EarlyStopping

    early_stop = EarlyStopping(patience=5, mode='min')
    for epoch in range(epochs):
        val_loss = train_and_eval()
        early_stop(val_loss, model, optimizer, epoch)
        if early_stop.early_stop:
            break

    # 训练结束后加载最优模型
    checkpoint = torch.load('best_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
"""
import torch
import numpy as np
import os


class EarlyStopping:
    """
    Early Stopping 机制 — 自动在验证集指标不再改善时停止训练。

    参数:
        patience:   连续 N 个 epoch 指标不改善则停止（推荐 5~10）
        min_delta:  多少变化算"改善"（默认 0，即任意改善都算）
        mode:       'min'（监控 loss 越小越好）或 'max'（监控 acc 越大越好）
        verbose:    是否打印停止信息
        path:       最优模型 checkpoint 保存路径

    使用方式:
        early_stop = EarlyStopping(patience=5, mode='min',
                                   path='best_checkpoint.pth')

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(...)
            test_loss, test_acc = evaluate(...)

            early_stop(test_loss, model, optimizer, epoch)
            if early_stop.early_stop:
                break

        # 加载最优模型
        checkpoint = torch.load('best_checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
    """

    def __init__(self, patience=5, min_delta=0.0, mode='min',
                 verbose=True, path='best_checkpoint.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.path = path

        self.counter = 0             # 计数器：连续多少轮没有改善
        self.best_score = None       # 当前最佳得分（统一为越高越好）
        self.early_stop = False      # 是否触发停止
        self.val_loss_min = np.inf   # 记录最小验证 loss
        self.best_epoch = 0          # 最优 epoch

        # 确保保存目录存在
        save_dir = os.path.dirname(self.path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def __call__(self, score, model, optimizer, epoch):
        """
        每个 epoch 结束后调用。

        参数:
            score:      当前 epoch 的验证指标（loss 或 acc）
            model:      模型（保存其 state_dict）
            optimizer:  优化器（保存其状态，用于断点续训）
            epoch:      当前 epoch 编号
        """
        # 根据 mode 转换 score：统一成"越大越好"
        if self.mode == 'min':
            current_score = -score   # loss 越小 → -loss 越大
        else:
            current_score = score    # acc 越大越好

        # 第一轮：直接设为最优
        if self.best_score is None:
            self.best_score = current_score
            self._save_checkpoint(score, model, optimizer, epoch)
            return

        # 检查是否有显著改善
        if current_score > self.best_score + self.min_delta:
            # 有改善 → 保存 checkpoint，重置计数器
            self.best_score = current_score
            self.counter = 0
            self._save_checkpoint(score, model, optimizer, epoch)
        else:
            # 无改善 → 计数器 +1
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping: {self.counter}/{self.patience} '
                      f'(best: {self.val_loss_min:.6f})')

            # 达到 patience → 停止训练
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'\n  >>> EarlyStopping 在第 {epoch} 个 epoch 触发！')
                    print(f'  >>> 最优 epoch: {self.best_epoch}, '
                          f'最优验证 loss: {self.val_loss_min:.6f}')
                    print(f'  >>> 最优模型已保存至: {self.path}')

    def _save_checkpoint(self, score, model, optimizer, epoch):
        """保存当前最优 checkpoint（含模型权重 + 优化器状态）"""
        if self.mode == 'min':
            self.val_loss_min = score
        self.best_epoch = epoch

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.val_loss_min if self.mode == 'min' else score,
        }, self.path)

        if self.verbose:
            print(f'  [checkpoint] epoch {epoch}: {score:.6f}')
