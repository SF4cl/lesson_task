import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# 从当前目录新拆分的多个模块导入关联功能
from dataset import MyDataSet, read_split_data
from net_model import Net
from utils import evaluate, save_curves, show_predictions, trainer


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 读取数据集划分
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
        args.data_root, args.val_rate
    )

    data_transform = transforms.Compose([
        transforms.Resize([args.img_size, args.img_size]),
        transforms.ToTensor(),
    ])

    train_dataset = MyDataSet(
        images_path=train_images_path,
        images_class=train_images_label,
        transform=data_transform,
    )
    val_dataset = MyDataSet(
        images_path=val_images_path,
        images_class=val_images_label,
        transform=data_transform,
    )

    nw = min([os.cpu_count() if os.cpu_count() is not None else 1, args.batch_size, 8])
    print(f"Using {nw} dataloader workers every process")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn,
    )

    # 模型实例化与优化器
    model = Net(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 结果保存路径准备
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, "best_model.pth")
    last_model_path = os.path.join(args.output_dir, "last_model.pth")
    curves_path = os.path.join(args.output_dir, "training_curves.png")

    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []

    best_acc = 0.0

    # 开始多轮训练
    for epoch in range(args.epochs):
        train_loss, train_acc = trainer(model, optimizer, train_loader, epoch, device)
        val_loss, val_acc = evaluate(model, val_loader, epoch, device)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print(
            f"Epoch[{epoch + 1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    # 保存最后一轮权重和学习过程曲线
    torch.save(model.state_dict(), last_model_path)
    save_curves(train_loss_list, train_acc_list, val_loss_list, val_acc_list, curves_path)

    print(f"Training finished. Best val acc: {best_acc:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Curves saved to: {curves_path}")

    # 模拟实际运用：随机抽取验证集样本绘制预测结果
    if len(val_images_path) > 0:
        print("\n展示部分样本的预测结果可视化图表...")
        show_predictions(
            model,
            val_images_path,
            data_transform,
            class_json_path="class_indices.json",
            device=device,
            num=6
        )


def parse_args():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_root = os.path.join(current_dir, "..", "Exp3数据", "Ch3", "sample")
    default_output_dir = os.path.join(current_dir, "..", "runs_exp3")

    parser = argparse.ArgumentParser(description="实验3：牛羊分类")
    parser.add_argument(
        "--data-root",
        type=str,
        default=default_data_root,
        help="数据集根目录（内含 cow/ sheep 子文件夹）",
    )
    parser.add_argument("--img-size", type=int, default=128, help="输入图像尺寸")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--val-rate", type=float, default=0.2, help="验证集比例")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help="输出目录（模型、曲线）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)