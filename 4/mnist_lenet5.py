import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# 获取当前脚本所在目录的绝对路径，这可以避免脚本在工作区根目录执行时产生相对路径错误
current_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(current_dir, 'Ch4')

# 设置相应的超参数
lr = 0.01          # 学习率
momentum = 0.5     # 动量
log_interval = 10  # 跑多少次 batch 进行一次日志记录
epochs = 10        # 训练轮数
batch_size = 64
test_batch_size = 1000

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 导入数据集
# 加载训练数据
train_loader = DataLoader(
    datasets.MNIST(root=data_root, train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

# 加载测试数据集
test_loader = DataLoader(
    datasets.MNIST(root=data_root, train=False, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=False)

# 2. 模型构建 (LeNet-5)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),    # padding=2 保证输入输出尺寸相同 input_size=(1*28*28)
            nn.ReLU(),                   # ReLU 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),         # output_size=(16*10*10)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)           # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 展平
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = LeNet().to(device)

# 定义优化器和损失函数
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
criterion = nn.CrossEntropyLoss()

# 3. 训练网络
def train(epoch):
    model.train()  # 设置为训练模式
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    avg_loss = train_loss / len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    return avg_loss, accuracy

# 4. 测试模型
def test():
    model.eval()   # 设置为测试模式
    test_loss = 0  # 初始化测试损失值为0
    correct = 0    # 初始化预测正确的数据个数为0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 把所有loss值进行累加
            test_loss += criterion(output, target).item() * data.size(0)
            
            # 获得最大概率的下标
            pred = output.argmax(dim=1, keepdim=True)
            
            # 对预测正确的数据个数进行累加
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 所有loss值进行过累加，除以总的数据长度得平均loss
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy

if __name__ == '__main__':
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train(epoch)
        te_loss, te_acc = test()
        
        train_losses.append(tr_loss)
        train_accuracies.append(tr_acc)
        test_losses.append(te_loss)
        test_accuracies.append(te_acc)

    # 绘制训练和测试的 Loss 曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # 绘制训练和测试的 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    # 随机取几张测试集照片进行预测结果的可视化
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images_device = images.to(device)
    
    with torch.no_grad():
        outputs = model(images_device)
        _, predicted = torch.max(outputs, 1)

    fig = plt.figure(figsize=(12, 6))
    for i in range(10):  # 显示前10张图片
        ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        # 将归一化后的数据还原
        img = images[i].numpy().squeeze()
        img = img * 0.3081 + 0.1307 # 反归一化
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Pred: {predicted[i].item()}\nTrue: {labels[i].item()}", 
                     color=("green" if predicted[i]==labels[i] else "red"))
    plt.tight_layout()
    plt.show()
