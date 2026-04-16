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
    model.train()

    criterion = nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1, device=device)
    accu_num = torch.zeros(1, device=device)
    sample_num = 0

    optimizer.zero_grad()

    progress_bar = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(progress_bar):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        pred_classes = torch.max(output, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()
        sample_num += images.shape[0]

        loss.backward()
        accu_loss += loss.detach()

        progress_bar.desc = (
            f"[train epoch {epoch}] "
            f"loss: {accu_loss.item() / (step + 1):.3f}, "
            f"acc: {accu_num.item() / sample_num:.3f}"
        )

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training", loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, epoch, device):
    """单轮验证/测试"""
    model.eval()

    criterion = nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1, device=device)
    accu_num = torch.zeros(1, device=device)
    sample_num = 0

    progress_bar = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(progress_bar):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        pred_classes = torch.max(output, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()
        sample_num += images.shape[0]
        accu_loss += loss

        progress_bar.desc = (
            f"[val epoch {epoch}] "
            f"loss: {accu_loss.item() / (step + 1):.3f}, "
            f"acc: {accu_num.item() / sample_num:.3f}"
        )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def save_curves(train_loss_list, train_acc_list, val_loss_list, val_acc_list, save_path):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label="train_loss")
    plt.plot(val_loss_list, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label="train_acc")
    plt.plot(val_acc_list, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


@torch.no_grad()
def show_predictions(model, val_images_path: List[str], transform, class_json_path: str, device, num=6):
    model.eval()

    with open(class_json_path, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)

    num_samples = min(num, len(val_images_path))
    samples = random.sample(val_images_path, num_samples)

    plt.figure(figsize=(10, 6))
    for i, img_path in enumerate(samples):
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image)
        img_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        pred_name = idx_to_class[str(pred_idx)]
        pred_prob = float(probs[0, pred_idx].item())

        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.title(f"Pred: {pred_name} (Prob: {pred_prob:.2f})")
        plt.axis("off")

    plt.tight_layout()
    plt.show()