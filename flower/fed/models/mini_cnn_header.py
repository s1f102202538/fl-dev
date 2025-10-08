import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MiniCNN_header(nn.Module):
  def __init__(self, dropout_rate: float = 0.2) -> None:
    super(MiniCNN_header, self).__init__()

    self.num_ftrs = 128

    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.dropout = nn.Dropout2d(dropout_rate)

    self.fc1 = nn.Linear(128 * 4 * 4, self.num_ftrs)

  def forward(self, x: Tensor) -> Tensor:
    # 畳み込み層
    x = F.relu(self.conv1(x))
    x = self.pool(x)  # 32x32 -> 16x16

    x = F.relu(self.conv2(x))
    x = self.pool(x)  # 16x16 -> 8x8

    x = F.relu(self.conv3(x))
    x = self.pool(x)  # 8x8 -> 4x4
    x = self.dropout(x)

    # 特徴量抽出
    x = x.view(x.size(0), -1)
    features = F.relu(self.fc1(x))

    return features


class MiniCNNMNIST_header(nn.Module):
  def __init__(self, dropout_rate: float = 0.2) -> None:
    super(MiniCNNMNIST_header, self).__init__()

    self.num_ftrs = 128

    self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.dropout = nn.Dropout2d(dropout_rate)

    # 特徴量抽出用の全結合層
    self.fc1 = nn.Linear(32 * 7 * 7, self.num_ftrs)

  def forward(self, x: Tensor) -> Tensor:
    """特徴量を抽出して返す（128次元）"""
    # 畳み込み層
    x = F.relu(self.conv1(x))
    x = self.pool(x)  # 28x28 -> 14x14

    x = F.relu(self.conv2(x))
    x = self.pool(x)  # 14x14 -> 7x7
    x = self.dropout(x)

    # 特徴量抽出
    x = x.view(x.size(0), -1)
    features = F.relu(self.fc1(x))

    return features
