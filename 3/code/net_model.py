import torch.nn as nn


class Net(nn.Module):
    """
    按文档结构构建网络：
    Conv(3->8, 3x3, s=1, p=1)
    Conv(8->8, 3x3, s=1, p=1)
    MaxPool(2x2)
    FC(64*64*8 -> 1000)
    FC(1000 -> 2)
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 64 * 8, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x