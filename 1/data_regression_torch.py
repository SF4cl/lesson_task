import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ==========================
# 1) 构造样本数据
# ==========================
# 为了结果可复现
torch.manual_seed(42)
np.random.seed(42)

# x: [-pi, 2*pi] 区间，200个点，转换成二维 (N, 1)
x = torch.unsqueeze(torch.linspace(-np.pi, np.pi * 2, 200), dim=1)
# y: cos(x) 周围带噪声散点
y = torch.cos(x) + 0.3 * torch.rand(x.size())


# ==========================
# 2) 定义网络
# 输入1维 -> 隐藏层20 -> 输出1维
# ==========================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.predict = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, x_input):
        out = self.predict(x_input)
        return out


# ==========================
# 3) 训练配置
# ==========================
net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_func = nn.MSELoss()

num_epochs = 6000
show_every = 200


# ==========================
# 4) 动态显示训练过程
# ==========================
plt.ion()  # 打开交互模式
fig = plt.figure(figsize=(8, 5))

for epoch in range(num_epochs):
    # 前向传播
    out = net(x)
    loss = loss_func(out, y)

    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每隔 show_every 次可视化一次
    if epoch % show_every == 0:
        plt.cla()
        plt.scatter(
            x.detach().numpy(),
            y.detach().numpy(),
            s=15,
            c="steelblue",
            alpha=0.7,
            label="noisy samples",
        )
        plt.plot(
            x.detach().numpy(),
            out.detach().numpy(),
            "r-",
            lw=2,
            label="network fit",
        )
        plt.title("Data Regression with Neural Network")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc="best")
        plt.text(
            0.02,
            0.95,
            f"Epoch: {epoch}/{num_epochs}  Loss: {loss.item():.6f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        plt.pause(0.05)

print(f"训练完成，最终损失 Loss = {loss.item():.6f}")


# ==========================
# 5) 简单测试：与真实 cos 曲线对比
# ==========================
net.eval()
with torch.no_grad():
    x_test = torch.unsqueeze(torch.linspace(-np.pi, np.pi * 2, 400), dim=1)
    y_pred = net(x_test)
    y_true = torch.cos(x_test)
    test_mse = torch.mean((y_pred - y_true) ** 2).item()

print(f"测试集(与 cos 真值对比) MSE = {test_mse:.6f}")

plt.ioff()  # 关闭交互模式
plt.figure(figsize=(8, 5))
plt.scatter(x.numpy(), y.numpy(), s=12, alpha=0.5, label="train noisy samples")
plt.plot(x_test.numpy(), y_true.numpy(), "g--", lw=2, label="true cos(x)")
plt.plot(x_test.numpy(), y_pred.numpy(), "r", lw=2, label="predicted")
plt.title("Final Regression Result")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("regression_result.png", dpi=300) # 保存最终结果图片
plt.show()
