import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子，保证每次生成的散点数据和网络初始权重一致，便于结果复现
torch.manual_seed(42)


# ==========================
# 1. 构造训练样本集与测试样本集
# ==========================
def generate_data(num_samples=100):
    """
    根据给定的样本数量生成两类服从正态分布的数据点
    """
    data = torch.ones(num_samples, 2)  # [cite: 150]

    # 第一类数据：均值为 2，方差为 1 [cite: 133]
    x0 = torch.normal(2 * data, 1)  # [cite: 151]
    y0 = torch.zeros(num_samples)  # 第一类标签设置为 0 [cite: 193, 195]

    # 第二类数据：均值为 -2，方差为 1 [cite: 133]
    x1 = torch.normal(-2 * data, 1)  # [cite: 154]
    y1 = torch.ones(num_samples)  # 第二类标签设置为 1 [cite: 194, 196]

    # 将两类数据按列合并 (维度 0 代表按列合并)，并转换为 FloatTensor [cite: 175, 190]
    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)

    # 将两类标签合并，并按 PyTorch 分类要求转换为 LongTensor 变量 [cite: 198, 200]
    y = torch.cat((y0, y1)).type(torch.LongTensor)

    return x, y


# 生成训练集 (每类 100 个点，共 200 个) 和 测试集 (每类 50 个点，共 100 个)
x_train, y_train = generate_data(100)
x_test, y_test = generate_data(50)  # 为完成实验任务特别追加的测试数据 [cite: 279]


# ==========================
# 2. 构建网络模型
# ==========================
class Net(nn.Module):  # 继承 torch.nn.Module 基类 [cite: 216]
    def __init__(self):
        super(Net, self).__init__()  # 继承父类初始化函数 [cite: 218]
        # 构建具有三层结构的分类网络 [cite: 203, 219]
        self.classify = nn.Sequential(
            nn.Linear(2, 15),  # 输入层 2 个节点，隐含层 15 个节点 [cite: 211, 212, 220]
            nn.ReLU(),  # 隐含层激活函数采用 ReLU [cite: 204, 221]
            nn.Linear(15, 2),  # 隐含层 15 个节点，输出层 2 个节点 [cite: 212, 213, 222]
            nn.Softmax(dim=1)  # 输出层采用 Softmax 函数输出概率值 [cite: 204, 223]
        )

    def forward(self, x):  # 前向传播过程 [cite: 225]
        classification = self.classify(x)  # [cite: 226]
        return classification  # [cite: 227]


net = Net()  # 实例化网络 [cite: 232]

# ==========================
# 3. 训练配置
# ==========================
# 采用 SGD 算法进行优化，学习率为 0.03
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)

# 采用交叉熵函数作为分类问题的损失函数 [cite: 230, 234]
loss_func = nn.CrossEntropyLoss()

# ==========================
# 4. 训练网络与动态可视化
# ==========================
plt.ion()  # 打开交互模式用于动态绘图
plt.figure(figsize=(12, 5))

for epoch in range(100):  # 设定循环训练 100 次 [cite: 235]
    out = net(x_train)  # 前向传播 [cite: 236]
    loss = loss_func(out, y_train)  # 计算损失 [cite: 237]

    optimizer.zero_grad()  # 梯度清零 [cite: 238]
    loss.backward()  # 反向传播 [cite: 239]
    optimizer.step()  # 参数优化更新 [cite: 240]

    # 动态显示分类效果 (每 10 次循环刷新一次画面)
    if epoch % 10 == 0 or epoch == 99:
        plt.clf()
        plt.subplot(1, 2, 1)

        # 获取每一行中概率最大值的下标，作为最终预测的分类标签 [cite: 263]
        classification = torch.max(out, 1)[1]

        # 把预测标签和真实标签转换成 numpy 数组以便计算和绘图 [cite: 264, 266]
        class_y = classification.data.numpy()  # [cite: 265]
        target_y = y_train.data.numpy()  # [cite: 267]

        # 绘制训练集散点图，采用 RdYlGn 颜色模式根据分类标签着色 [cite: 269, 273]
        plt.scatter(x_train.data.numpy()[:, 0], x_train.data.numpy()[:, 1], c=class_y, s=100, cmap='RdYlGn')

        # 计算当前轮次的准确率
        accuracy = sum(class_y == target_y) / 200.0

        # 在图中指定位置显示当前的准确率
        plt.text(1.5, -4, f'Accuracy={accuracy:.2f}', fontdict={'size': 15, 'color': 'red'})
        plt.title(f"Training Phase - Epoch: {epoch}")

        plt.pause(0.1)

print("训练程序执行完毕！")
plt.ioff()  # 关闭动态交互模式

# ==========================
# 5. 测试程序 (验证泛化能力)
# ==========================
print("开始执行测试程序...")
net.eval()  # 切换至评估模式
with torch.no_grad():  # 测试阶段无需追踪计算图的梯度
    test_out = net(x_test)
    # 获取测试集的预测标签
    test_pred = torch.max(test_out, 1)[1]
    test_class_y = test_pred.data.numpy()
    test_target_y = y_test.data.numpy()

    # 计算测试集准确率
    test_accuracy = sum(test_class_y == test_target_y) / 100.0

print(f"测试集最终准确率: {test_accuracy * 100:.2f}%")

# 绘制测试结果对比图
plt.subplot(1, 2, 2)
# 使用测试集特征数据与测试集预测出来的标签进行绘图
plt.scatter(x_test.data.numpy()[:, 0], x_test.data.numpy()[:, 1], c=test_class_y, s=100, cmap='RdYlGn')
plt.text(1.5, -4, f'Test Acc={test_accuracy:.2f}', fontdict={'size': 15, 'color': 'blue'})
plt.title("Testing Phase Result")

plt.tight_layout()
plt.show()  # 定格显示最终同时包含训练和测试结果的完整图像