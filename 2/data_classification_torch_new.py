import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以保证结果可复现
torch.manual_seed(42)

# ==========================
# 1. 构造样本集
# ==========================
# 生成100个数据点，初始化为1
data = torch.ones(100, 2)

# 第一类数据：正态分布，均值为2，方差为1
x0 = torch.normal(2 * data, 1)
# 第二类数据：正态分布，均值为-2，方差为1
x1 = torch.normal(-2 * data, 1)

# 将两类数据按列合并，并转换为Float类型的Tensor变量
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)

# 标签生成
y0 = torch.zeros(100)  # 第一类标签设置为0
y1 = torch.ones(100)   # 第二类标签设置为1

# 将两类标签合并，并转换为Long类型的Tensor变量
y = torch.cat((y0, y1)).type(torch.LongTensor)

# ==========================
# 2. 生成测试集 (用于实验任务)
# ==========================
test_data = torch.ones(50, 2)
test_x0 = torch.normal(2 * test_data, 1)
test_x1 = torch.normal(-2 * test_data, 1)
x_test = torch.cat((test_x0, test_x1), 0).type(torch.FloatTensor)

test_y0 = torch.zeros(50)
test_y1 = torch.ones(50)
y_test = torch.cat((test_y0, test_y1)).type(torch.LongTensor)

# ==========================
# 3. 构建网络
# ==========================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(2, 15),   # 输入层2个节点，隐含层15个节点
            nn.ReLU(),          # 激活函数ReLU
            nn.Linear(15, 2),   # 隐含层15个节点，输出层2个节点
            nn.Softmax(dim=1)   # Softmax函数输出概率值
        )

    def forward(self, x):
        classification = self.classify(x)
        return classification

# ==========================
# 4. 训练网络
# ==========================
net = Net()
# 采用SGD算法进行优化，学习率0.03
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
# 分类问题采用交叉熵函数作为损失函数
loss_func = nn.CrossEntropyLoss()

plt.ion() # 打开交互模式
plt.figure(figsize=(12, 5))

print("开始训练...")
for epoch in range(100):
    out = net(x)
    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 动态可视化
    if epoch % 5 == 0 or epoch == 99:
        plt.clf()
        plt.subplot(1, 2, 1)
        
        # 返回每一行中最大值的下标
        classification = torch.max(out, 1)[1]
        
        # 将张量转换成numpy数组便于绘图和计算
        class_y = classification.data.numpy()
        target_y = y.data.numpy()

        # 绘制散点图
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=class_y, s=100, cmap='RdYlGn')
        
        # 计算准确率
        accuracy = sum(class_y == target_y) / 200.0
        
        # 显示准确率
        plt.text(1.5, -4, f'Accuracy={accuracy:.2f}', fontdict={'size': 15, 'color': 'red'})
        plt.title(f"Training - Epoch {epoch}")
        plt.pause(0.1)

plt.ioff()
print("训练结束。")

# ==========================
# 5. 测试程序 (实验任务)
# ==========================
print("开始测试...")
net.eval()
with torch.no_grad():
    test_out = net(x_test)
    test_prediction = torch.max(test_out, 1)[1]
    
    test_class_y = test_prediction.data.numpy()
    test_target_y = y_test.data.numpy()
    
    test_accuracy = sum(test_class_y == test_target_y) / 100.0

plt.subplot(1, 2, 2)
plt.scatter(x_test.data.numpy()[:, 0], x_test.data.numpy()[:, 1], c=test_class_y, s=100, cmap='RdYlGn')
plt.text(1.5, -4, f'Test Acc={test_accuracy:.2f}', fontdict={'size': 15, 'color': 'blue'})
plt.title("Testing Result")

plt.tight_layout()
plt.show()
print(f"测试集准确率: {test_accuracy*100:.2f}%")
