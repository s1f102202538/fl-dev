from typing import override

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_model import BaseModel


class MiniCNN(BaseModel):
  """軽量なMiniCNNモデル - CIFAR-10対応"""

  def __init__(self, dropout_rate: float = 0.2) -> None:
    super().__init__()

    # CIFAR-10用畳み込み層（3チャンネル入力、32x32画像）
    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.dropout = nn.Dropout2d(dropout_rate)

    # 全結合層
    self.fc1 = nn.Linear(128 * 4 * 4, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x: Tensor):
    # 畳み込み層
    x = F.relu(self.conv1(x))
    x = self.pool(x)  # 32x32 -> 16x16

    x = F.relu(self.conv2(x))
    x = self.pool(x)  # 16x16 -> 8x8

    x = F.relu(self.conv3(x))
    x = self.pool(x)  # 8x8 -> 4x4
    x = self.dropout(x)

    # 全結合層
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)

    return x


class MiniCNNMNIST(BaseModel):
  """軽量なMiniCNNモデル - MNIST/FashionMNIST対応"""

  def __init__(self, dropout_rate: float = 0.2) -> None:
    super().__init__()

    # MNIST用畳み込み層（1チャンネル入力、28x28画像）
    self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.dropout = nn.Dropout2d(dropout_rate)

    # 全結合層
    self.fc1 = nn.Linear(32 * 7 * 7, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x: Tensor) -> Tensor:
    # 畳み込み層
    x = F.relu(self.conv1(x))
    x = self.pool(x)  # 28x28 -> 14x14

    x = F.relu(self.conv2(x))
    x = self.pool(x)  # 14x14 -> 7x7
    x = self.dropout(x)

    # 全結合層
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)

    return x


# 特徴量抽出用のヘッダークラス
class MiniCNN_header(nn.Module):
  """CIFAR-10用のMiniCNNヘッダー（特徴量抽出のみ）"""

  def __init__(self, dropout_rate: float = 0.2) -> None:
    super(MiniCNN_header, self).__init__()

    # CIFAR-10用畳み込み層（3チャンネル入力、32x32画像）
    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.dropout = nn.Dropout2d(dropout_rate)

    # 特徴量抽出のための全結合層（最終分類層は除く）
    self.fc1 = nn.Linear(128 * 4 * 4, 128)

    # 特徴量次元（MOONで使用）
    self.num_ftrs = 128

  def forward(self, x: Tensor) -> Tensor:
    # 畳み込み層
    x = F.relu(self.conv1(x))
    x = self.pool(x)  # 32x32 -> 16x16

    x = F.relu(self.conv2(x))
    x = self.pool(x)  # 16x16 -> 8x8

    x = F.relu(self.conv3(x))
    x = self.pool(x)  # 8x8 -> 4x4
    x = self.dropout(x)

    # 特徴量抽出（分類層は除く）
    x = x.view(x.size(0), -1)
    features = F.relu(self.fc1(x))

    return features


class MiniCNNMNIST_header(nn.Module):
  """MNIST用のMiniCNNヘッダー（特徴量抽出のみ）"""

  def __init__(self, dropout_rate: float = 0.2) -> None:
    super(MiniCNNMNIST_header, self).__init__()

    # MNIST用畳み込み層（1チャンネル入力、28x28画像）
    self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.dropout = nn.Dropout2d(dropout_rate)

    # 特徴量抽出のための全結合層（最終分類層は除く）
    self.fc1 = nn.Linear(32 * 7 * 7, 128)

    # 特徴量次元（MOONで使用）
    self.num_ftrs = 128

  def forward(self, x: Tensor) -> Tensor:
    # 畳み込み層
    x = F.relu(self.conv1(x))
    x = self.pool(x)  # 28x28 -> 14x14

    x = F.relu(self.conv2(x))
    x = self.pool(x)  # 14x14 -> 7x7
    x = self.dropout(x)

    # 特徴量抽出（分類層は除く）
    x = x.view(x.size(0), -1)
    features = F.relu(self.fc1(x))

    return features


# MOON論文で使用された軽量なSimpleCNNモデル
class SimpleCNN(BaseModel):
  """MOON論文準拠のSimpleCNN - CIFAR-10対応"""

  def __init__(self, input_dim: int = 16 * 5 * 5, hidden_dims: list = [120, 84], output_dim: int = 10):
    super(SimpleCNN, self).__init__()

    # CIFAR-10用畳み込み層
    self.conv1 = nn.Conv2d(3, 6, 5)  # kernel_size=5, no padding
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)  # kernel_size=5, no padding

    # 全結合層
    self.fc1 = nn.Linear(input_dim, hidden_dims[0])  # 16*5*5 -> 120
    self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])  # 120 -> 84
    self.fc3 = nn.Linear(hidden_dims[1], output_dim)  # 84 -> 10

  def forward(self, x: Tensor) -> Tensor:
    # 畳み込み層（元論文に準拠）
    x = self.pool(self.relu(self.conv1(x)))  # 32x32 -> 28x28 -> 14x14
    x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 10x10 -> 5x5
    x = x.view(-1, 16 * 5 * 5)

    # 全結合層
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  @override
  def predict(self, x: Tensor) -> Tensor:
    return self.forward(x)


class SimpleCNNMNIST(BaseModel):
  """MOON論文準拠のSimpleCNN - MNIST対応"""

  def __init__(self, input_dim: int = 16 * 4 * 4, hidden_dims: list = [120, 84], output_dim: int = 10):
    super(SimpleCNNMNIST, self).__init__()

    # MNIST用畳み込み層
    self.conv1 = nn.Conv2d(1, 6, 5)  # kernel_size=5, no padding
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)  # kernel_size=5, no padding

    # 全結合層
    self.fc1 = nn.Linear(input_dim, hidden_dims[0])  # 16*4*4 -> 120
    self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])  # 120 -> 84
    self.fc3 = nn.Linear(hidden_dims[1], output_dim)  # 84 -> 10

  def forward(self, x: Tensor) -> Tensor:
    # 畳み込み層
    x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 24x24 -> 12x12
    x = self.pool(self.relu(self.conv2(x)))  # 12x12 -> 8x8 -> 4x4
    x = x.view(-1, 16 * 4 * 4)

    # 全結合層
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  @override
  def predict(self, x: Tensor) -> Tensor:
    return self.forward(x)


# 論文準拠の特徴量抽出用ヘッダークラス
class SimpleCNN_header(nn.Module):
  """MOON論文準拠のSimpleCNNヘッダー - CIFAR-10用（特徴量抽出のみ）"""

  def __init__(self, input_dim: int = 16 * 5 * 5, hidden_dims: list = [120, 84], output_dim: int = 10):
    super(SimpleCNN_header, self).__init__()

    # CIFAR-10用畳み込み層
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)

    # 特徴量抽出用の全結合層（分類層は除く）
    self.fc1 = nn.Linear(input_dim, hidden_dims[0])
    self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

    # 特徴量次元
    self.num_ftrs = hidden_dims[1]  # 84

  def forward(self, x: Tensor) -> Tensor:
    x = self.pool(self.relu(self.conv1(x)))
    x = self.pool(self.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)

    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    return x


class SimpleCNNMNIST_header(nn.Module):
  """MOON論文準拠のSimpleCNNヘッダー - MNIST用（特徴量抽出のみ）"""

  def __init__(self, input_dim: int = 16 * 4 * 4, hidden_dims: list = [120, 84], output_dim: int = 10):
    super(SimpleCNNMNIST_header, self).__init__()

    # MNIST用畳み込み層
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)

    # 特徴量抽出用の全結合層（分類層は除く）
    self.fc1 = nn.Linear(input_dim, hidden_dims[0])
    self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

    # 特徴量次元
    self.num_ftrs = hidden_dims[1]  # 84

  def forward(self, x: Tensor) -> Tensor:
    x = self.pool(self.relu(self.conv1(x)))
    x = self.pool(self.relu(self.conv2(x)))
    x = x.view(-1, 16 * 4 * 4)

    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    return x
