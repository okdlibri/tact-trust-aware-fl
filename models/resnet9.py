"""ResNet-9 for CIFAR-scale images (32x32x3, 10-100 classes).

Standard fast-training ResNet-9 (conv-bn-relu stem + two residual blocks),
as commonly used in FL literature for CIFAR-10/100 experiments.
"""
import torch
import torch.nn as nn


def conv_bn(in_ch, out_ch, pool=False):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = conv_bn(channels, channels)
        self.conv2 = conv_bn(channels, channels)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class ResNet9(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.prep = conv_bn(in_channels, 64)
        self.layer1 = conv_bn(64, 128, pool=True)
        self.res1 = Residual(128)
        self.layer2 = conv_bn(128, 256, pool=True)
        self.layer3 = conv_bn(256, 512, pool=True)
        self.res2 = Residual(512)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.res1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.res2(x)
        x = self.pool(x)
        return self.classifier(x)


if __name__ == "__main__":
    m = ResNet9()
    x = torch.randn(4, 3, 32, 32)
    y = m(x)
    assert y.shape == (4, 10)
    print("ResNet9 OK, output shape:", y.shape, "params:", sum(p.numel() for p in m.parameters()))
