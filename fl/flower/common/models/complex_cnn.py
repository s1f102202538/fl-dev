import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ComplexCNN(nn.Module):
  """複雑なCNNモデル - より深いアーキテクチャ（元のMiniCNNの複雑版）"""

  def __init__(self, dropout_rate: float = 0.25) -> None:
    super().__init__()

    self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
    self.bn2 = nn.BatchNorm2d(32)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.dropout1 = nn.Dropout2d(dropout_rate)

    self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
    self.bn3 = nn.BatchNorm2d(64)
    self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
    self.bn4 = nn.BatchNorm2d(64)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.dropout2 = nn.Dropout2d(dropout_rate)

    self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
    self.bn5 = nn.BatchNorm2d(128)
    self.pool3 = nn.AdaptiveAvgPool2d((4, 4))

    # 全結合層
    self.fc1 = nn.Linear(128 * 4 * 4, 512)
    self.bn_fc1 = nn.BatchNorm1d(512)
    self.dropout3 = nn.Dropout(dropout_rate * 2)
    self.fc2 = nn.Linear(512, 256)
    self.bn_fc2 = nn.BatchNorm1d(256)
    self.dropout4 = nn.Dropout(dropout_rate)
    self.fc3 = nn.Linear(256, 10)

  def forward(self, x: Tensor) -> Tensor:
    # 第1ブロック
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = self.pool1(x)
    x = self.dropout1(x)

    # 第2ブロック
    x = F.relu(self.bn3(self.conv3(x)))
    x = F.relu(self.bn4(self.conv4(x)))
    x = self.pool2(x)
    x = self.dropout2(x)

    # 第3ブロック
    x = F.relu(self.bn5(self.conv5(x)))
    x = self.pool3(x)

    # 全結合層
    x = x.view(x.size(0), -1)
    x = F.relu(self.bn_fc1(self.fc1(x)))
    x = self.dropout3(x)
    x = F.relu(self.bn_fc2(self.fc2(x)))
    x = self.dropout4(x)
    x = self.fc3(x)

    return x


class ResidualBlock(nn.Module):
  """残差ブロック - さらに高度なアーキテクチャ用"""

  def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride), nn.BatchNorm2d(out_channels))

  def forward(self, x: Tensor) -> Tensor:
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out


class ResNetCNN(nn.Module):
  """ResNet風のCNNモデル - 最も高度なアーキテクチャ"""

  def __init__(self, dropout_rate: float = 0.2) -> None:
    super().__init__()

    self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
    self.bn1 = nn.BatchNorm2d(16)

    self.layer1 = self._make_layer(16, 16, 2, stride=1)
    self.layer2 = self._make_layer(16, 32, 2, stride=2)
    self.layer3 = self._make_layer(32, 64, 2, stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(64, 10)
    self.dropout = nn.Dropout(dropout_rate)

  def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
    layers = []
    layers.append(ResidualBlock(in_channels, out_channels, stride))
    for _ in range(1, num_blocks):
      layers.append(ResidualBlock(out_channels, out_channels))
    return nn.Sequential(*layers)

  def forward(self, x: Tensor) -> Tensor:
    x = F.relu(self.bn1(self.conv1(x)))

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.dropout(x)
    x = self.fc(x)

    return x
