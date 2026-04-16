import json
import random
import sys
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


def trainer(model, optimizer, data_loader, epoch, device):
    """单轮训练"""
    model.train()  # 将模型设置为训练模式（启用 DropOut / BatchNorm 等特性）

    criterion = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵（适合分类任务，内部包含Softmax）
    accu_loss = torch.zeros(1, device=device)  # 记录这一个 Epoch 的累计损失
    accu_num = torch.zeros(1, device=device)   # 记录这一个 Epoch 猜对的样本总数
    sample_num = 0                             # 记录当前已处理的样本总条数

    optimizer.zero_grad()  # 在开始新一轮的训练前，先清空优化器里上一步残留的梯度（防止累积）

    progress_bar = tqdm(data_loader, file=sys.stdout)  # 包装 data_loader，在控制台打印进度条
    for step, data in enumerate(progress_bar):
        images, labels = data
        images, labels = images.to(device), labels.to(device)  # 将图片和标签移动到指定设备（如 GPU）上

        output = model(images)            # 前向传播：把图片喂给网络，得到初步预测概率
        loss = criterion(output, labels)  # 计算损失：对比“网络的预测”与“真实标签”，算出扣分差距

        pred_classes = torch.max(output, dim=1)[1]        # 从预测输出中取出概率最大的那个类别的编号
        accu_num += torch.eq(pred_classes, labels).sum()  # 将预测正确的照片数量相加并累计
        sample_num += images.shape[0]                     # 累计当前批次包含的照片张数

        loss.backward()             # 反向传播：根据损失大小，自动计算参数更新所需的梯度
        accu_loss += loss.detach()  # 累计这一个批次的损失（detach 可以切断计算图，节省显存）

        # 实时更新进度条后面显示的日志信息
        progress_bar.desc = (
            f"[train epoch {epoch}] "
            f"loss: {accu_loss.item() / (step + 1):.3f}, "
            f"acc: {accu_num.item() / sample_num:.3f}"
        )

        # 异常阻断：如果损失变成无限大/无效值（梯度爆炸），立刻停止程序
        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training", loss)
            sys.exit(1)

        optimizer.step()       # 优化器工作：根据刚才计算出的梯度，正式调整神经元的权重参数
        optimizer.zero_grad()  # 再次清空梯度，以免污染下一个批次

    # 循环结束后，返回这一轮训练的“平均损失”和“整体准确率”
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()  # 禁用梯度计算（验证阶段不需要更新参数，这样设置跑得快、省内存）
def evaluate(model, data_loader, epoch, device):
    """单轮验证/测试"""
    model.eval()  # 将模型设置为验证模式（冻结权重更新，稳定网络）

    criterion = nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1, device=device)
    accu_num = torch.zeros(1, device=device)
    sample_num = 0

    progress_bar = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(progress_bar):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        output = model(images)            # 让模型针对没见过的图片做考题
        loss = criterion(output, labels)  # 评判模型这次做得怎么样（算错题分）

        pred_classes = torch.max(output, dim=1)[1]        # 获取考得最具信心的答案
        accu_num += torch.eq(pred_classes, labels).sum()  # 累计答对的题数
        sample_num += images.shape[0]
        accu_loss += loss

        progress_bar.desc = (
            f"[val epoch {epoch}] "
            f"loss: {accu_loss.item() / (step + 1):.3f}, "
            f"acc: {accu_num.item() / sample_num:.3f}"
        )

    # 返回这一轮“考试”的平均分及正确率
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def save_curves(train_loss_list, train_acc_list, val_loss_list, val_acc_list, save_path):
    """绘制训练过程中的日志曲线图，便于监控是否过拟合"""
    plt.figure(figsize=(10, 4))  # 申请一块宽高比例 10:4 的画布

    # 绘制左侧的 Loss 视图：展现模型误差是怎么按轮次逐渐下降的
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label="train_loss")
    plt.plot(val_loss_list, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss Curve")

    # 绘制右侧的 Accuracy 视图：展现模型答题率怎么升高的
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label="train_acc")
    plt.plot(val_acc_list, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.tight_layout()               # 自动适应间距，防止字三重叠
    plt.savefig(save_path, dpi=300)  # 高清保存为图片
    plt.close()                      # 收起画板（释放内存）


@torch.no_grad()
def show_predictions(model, val_images_path: List[str], transform, class_json_path: str, device, num=6):
    """在界面上弹出小图，直观展示模型对这几张图片的预测结果与自信程度"""
    model.eval()

    # 打开之前存好的索引文件，把数字标签 `0/1` 转回文本 `cow/sheep`
    with open(class_json_path, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)

    # 决定要抽取多少张：默认为6张，如果你的图片总数不足6，就取全集
    num_samples = min(num, len(val_images_path))
    samples = random.sample(val_images_path, num_samples)  # 随机在验证集中抽图片路径

    plt.figure(figsize=(10, 6))
    for i, img_path in enumerate(samples):
        image = Image.open(img_path).convert("RGB")  # 真正在硬盘里打开这张图
        img_tensor = transform(image)                # 做和训练阶段一样的预处理（裁剪/转张量等）
        
        # 神经网络不吃单张图，它吃“批次”。所以用 unsqueeze 加个空维度，强行把形状变为： [1批, C通道, H宽, W高]
        img_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

        logits = model(img_tensor)            # 跑前向得到没章法的输出分
        probs = torch.softmax(logits, dim=1)  # 利用 Softmax 函数，把输出转为加一起等于 100% 的概率分布
        
        # 从输出里揪出两个信息：预测结论（预测它是啥）、预测概率（有 % 多少的把握它是这个结论）
        pred_idx = int(torch.argmax(probs, dim=1).item())
        pred_name = idx_to_class[str(pred_idx)]
        pred_prob = float(probs[0, pred_idx].item())

        # 画到界面上
        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.title(f"Pred: {pred_name} (Prob: {pred_prob:.2f})")
        plt.axis("off")  # 关掉 X 轴和 Y 轴刻度，避免干扰看图

    plt.tight_layout()
    plt.show()  # 弹出窗体输出画面