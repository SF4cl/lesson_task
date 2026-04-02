import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ==========================
# 实验扩展版参数配置
# ==========================
TARGET_FUNC = "sin"  # 选 "cos" 或 "sin"
NUM_EPOCHS = 6000
LEARNING_RATE = 0.05
HIDDEN_UNITS = [32, 32]  # 多层隐藏节点数的配置，如 [32, 32] 表示两层32节点
NOISE_STD = 0.3  # 噪声系数

# 再做些设置使结果可重复一致
torch.manual_seed(100)
np.random.seed(100)


def target_function(x):
    if TARGET_FUNC == "sin":
        return torch.sin(x)
    return torch.cos(x)


# ==========================
# 1) 构造样本数据
# ==========================
x = torch.unsqueeze(torch.linspace(-np.pi, np.pi * 2, 200), dim=1)
y = target_function(x) + NOISE_STD * torch.rand(x.size())
# ==========================
# 2) 自由定制层数、节点数的网络
# ==========================
class AdvancedNet(nn.Module):
    def __init__(self, in_features=1, hidden_dims=[20], out_features=1):
        super(AdvancedNet, self).__init__()

        layers = []
        prev_dim = in_features

        # 逐层添加 Linear + ReLU
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        # 图最后输出
        layers.append(nn.Linear(prev_dim, out_features))

        self.predict = nn.Sequential(*layers)

    def forward(self, x_input):
        return self.predict(x_input)


# ==========================
# 3) 训练配置
# ==========================
net = AdvancedNet(hidden_dims=HIDDEN_UNITS)
print(f"当前网络结构：\n{net}")

# 如果网络层数深或拟合较难，也可将 SGD 换为 Adam 尝试，如 `torch.optim.Adam(net.parameters(), lr=0.01)`
optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)
loss_func = nn.MSELoss()
show_every = 300


# ==========================
# 4) 动态显示训练过程
# ==========================
plt.ion()
fig = plt.figure(figsize=(9, 6))

for epoch in range(NUM_EPOCHS):
    out = net(x)
    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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
        plt.title(f"Data Regression - Func: {TARGET_FUNC}, Hidden: {HIDDEN_UNITS}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc="upper right")
        plt.text(
            0.02,
            0.95,
            f"Epoch: {epoch:04d}/{NUM_EPOCHS}\nLoss: {loss.item():.6f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        plt.pause(0.01)

print(f"训练完成，最终损失 Loss = {loss.item():.6f}")


# ==========================
# 5) 简单测试：与真值对比
# ==========================
net.eval()
with torch.no_grad():
    x_test = torch.unsqueeze(torch.linspace(-np.pi, np.pi * 2, 400), dim=1)
    y_pred = net(x_test)
    y_true = target_function(x_test)
    test_mse = torch.mean((y_pred - y_true) ** 2).item()

print(f"在 {TARGET_FUNC} 测试集上 MSE = {test_mse:.6f}")

plt.ioff()
plt.figure(figsize=(9, 6))
plt.scatter(x.numpy(), y.numpy(), s=12, alpha=0.3, label="train noisy samples")
plt.plot(x_test.numpy(), y_true.numpy(), "g--", lw=3, label=f"true {TARGET_FUNC}(x)")
plt.plot(x_test.numpy(), y_pred.numpy(), "r", lw=2, label="predicted")
plt.title(f"Final Regression | HIDDEN: {HIDDEN_UNITS} | Test MSE: {test_mse:.6f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()