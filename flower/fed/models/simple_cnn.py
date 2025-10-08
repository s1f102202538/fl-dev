import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimpleCNN(nn.Module):
  """軽量なCNNモデル - FashionMNIST用のシンプルなアーキテクチャ"""

  def __init__(self, dropout_rate: float = 0.2) -> None:
    super().__init__()

    # 軽量な畳み込み層
    self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.dropout = nn.Dropout2d(dropout_rate)

    # 軽量な全結合層
    self.fc1 = nn.Linear(32 * 7 * 7, 128)
    self.fc2 = nn.Linear(128, 10)
    self.dropout_fc = nn.Dropout(dropout_rate)

  def forward(self, x: Tensor) -> Tensor:
    # 第1畳み込みブロック
    x = F.relu(self.conv1(x))
    x = self.pool(x)  # 28x28 -> 14x14

    # 第2畳み込みブロック
    x = F.relu(self.conv2(x))
    x = self.pool(x)  # 14x14 -> 7x7
    x = self.dropout(x)

    # 全結合層
    x = x.view(x.size(0), -1)  # フラット化
    x = F.relu(self.fc1(x))
    x = self.dropout_fc(x)
    x = self.fc2(x)

    return x


class LightCNN(nn.Module):
  """さらに軽量なCNNモデル - 最小限のパラメータ"""

  def __init__(self, dropout_rate: float = 0.1) -> None:
    super().__init__()

    # 最小限の畳み込み層
    self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
    self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
    self.pool = nn.MaxPool2d(2, 2)

    # 最小限の全結合層
    self.fc1 = nn.Linear(16 * 7 * 7, 64)
    self.fc2 = nn.Linear(64, 10)
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, x: Tensor) -> Tensor:
    # 第1畳み込みブロック
    x = F.relu(self.conv1(x))
    x = self.pool(x)  # 28x28 -> 14x14

    # 第2畳み込みブロック
    x = F.relu(self.conv2(x))
    x = self.pool(x)  # 14x14 -> 7x7

    # 全結合層
    x = x.view(x.size(0), -1)  # フラット化
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)

    return x
