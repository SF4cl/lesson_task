# 3.1 导入必要的模块
import argparse
import os

import torch  # 导入构建网络模块
import torch.nn.functional as F  # 导入激活函数模块
import torch.optim as optim  # 导入优化器模块
from torchvision import transforms  # 导入转换操作模块
from torchvision import datasets  # 导入数据集模块
from torch.utils.data import DataLoader  # 导入打包模块
import matplotlib.pyplot as plt  # 导入可视化模块

# 3.3 定义数据的预处理方式
transform = transforms.Compose([
    transforms.ToTensor(),  # 输入除以255,归一化,将输入归一化到(0,1)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 使用“(x-mean)/std”,将每个元素分布到(-1,1)
])


def resolve_data_root(input_root: str = ""):
    """解析 CIFAR-10 根路径，优先使用命令行参数。"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if input_root:
        candidates.append(input_root)
    candidates.extend([
        os.path.join(current_dir, "Ch5", "CIFAR10"),
        os.path.join(current_dir, "Ch5", "Ch5", "CIFAR10"),
        os.path.join(current_dir, "CIFAR10"),
    ])

    for path in candidates:
        full_path = os.path.abspath(path)
        if os.path.isdir(full_path):
            return full_path

    return os.path.abspath(candidates[0]) if candidates else os.path.join(current_dir, "CIFAR10")


def build_dataloaders(data_root, batch_size=64, download=False, num_workers=2):
    """构建训练/测试 DataLoader，必要时下载数据集。"""
    os.makedirs(data_root, exist_ok=True)

    try:
        trainset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=download,
            transform=transform,
        )
        testset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=download,
            transform=transform,
        )
    except RuntimeError as e:
        if "Dataset not found or corrupted" in str(e):
            raise RuntimeError(
                f"数据集未找到: {data_root}\n"
                "请使用 --download 自动下载，或将 CIFAR-10 放到该目录后重试。"
            ) from e
        raise

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return trainloader, testloader


# ==========================================
# 4. 构建网络模型
# ==========================================
class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch1x1 = self.branch1x1(x)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(2200, 10)

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net()
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

trainloader = None
testloader = None


# ==========================================
# 5. 定义训练网络 (增加 Train Accuracy 统计)
# ==========================================
def train(epoch):
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, data in enumerate(trainloader, 0):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()

        # 统计训练集的正确率
        predicted = torch.max(outputs.data, dim=1)[1]
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 300 == 299:
            print('[Epoch %d, Batch %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

    train_acc = 100.0 * correct / total
    train_loss = total_loss / len(trainloader)
    return train_loss, train_acc


# ==========================================
# 6. 定义测试网络 (增加 Test Loss 统计)
# ==========================================
def test():
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # 计算测试集的误差
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            predicted = torch.max(outputs.data, dim=1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100.0 * correct / total
    avg_test_loss = test_loss / len(testloader)
    print('Accuracy on test set: %.2f %% | Loss: %.3f' % (test_acc, avg_test_loss))
    return avg_test_loss, test_acc


# ==========================================
# 7. 开始训练和测试网络及高级双指标可视化
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GoogLeNet CIFAR-10 训练脚本")
    parser.add_argument("--data-root", type=str, default="", help="CIFAR-10 根目录")
    parser.add_argument("--download", action="store_true", help="若数据不存在则自动下载")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader 进程数")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    print(f"Using device: {device}")

    trainloader, testloader = build_dataloaders(
        data_root=data_root, batch_size=args.batch_size, download=args.download, num_workers=args.workers)

    # 记录 4 个核心指标的列表
    history_train_loss = []
    history_test_loss = []
    history_train_acc = []
    history_test_acc = []

    print("Start Training GoogLeNet on CIFAR-10...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test()

        history_train_loss.append(train_loss)
        history_train_acc.append(train_acc)
        history_test_loss.append(test_loss)
        history_test_acc.append(test_acc)

    print("Finished Training.")

    # ==========================================
    # 8. 绘制并保存四指标双子图 (学术标准格式)
    # ==========================================
    print("Generating comprehensive plots...")
    epochs_range = range(1, args.epochs + 1)

    plt.figure(figsize=(12, 5))

    # --- 子图1：Loss 走势 (Train vs Test) ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_train_loss, marker='o', color='#1f77b4', linewidth=2, label='Train Loss')
    plt.plot(epochs_range, history_test_loss, marker='s', color='#ff7f0e', linewidth=2, label='Test Loss')
    plt.xticks(epochs_range)
    plt.title('Loss Curve (Train vs Test)')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # --- 子图2：Accuracy 走势 (Train vs Test) ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_train_acc, marker='o', color='#1f77b4', linewidth=2, label='Train Accuracy')
    plt.plot(epochs_range, history_test_acc, marker='s', color='#ff7f0e', linewidth=2, label='Test Accuracy')

    # 标注最后一个 Epoch 的测试集准确率
    final_acc = history_test_acc[-1]
    plt.annotate(f'Final Test Acc: {final_acc:.2f}%',
                 xy=(args.epochs, final_acc),
                 xytext=(-60, 15), textcoords='offset points',
                 fontweight='bold', color='#d62728',
                 arrowprops=dict(arrowstyle="->", color='#d62728', lw=1.5))

    plt.xticks(epochs_range)
    plt.title('Accuracy Curve (Train vs Test)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('googlenet_cifar10_full_metrics.png', dpi=300)
    print("图表已生成并保存为 googlenet_cifar10_full_metrics.png")
    plt.show()