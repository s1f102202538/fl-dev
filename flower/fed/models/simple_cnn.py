from typing import override

import torch.nn as nn
from torch import Tensor

from .base_model import BaseModel


# MOON論文で使用された軽量なSimpleCNNモデル
class SimpleCNN(BaseModel):
  def __init__(self, output_dim: int = 10, input_dim: int = 16 * 5 * 5, hidden_dims: list = [120, 84]):
    super().__init__()

    # CIFAR-10用畳み込み層
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)

    # 全結合層
    self.fc1 = nn.Linear(input_dim, hidden_dims[0])  # 16*5*5 -> 120
    self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])  # 120 -> 84
    self.fc3 = nn.Linear(hidden_dims[1], output_dim)  # 84 -> 10

  def forward(self, x: Tensor) -> Tensor:
    # 畳み込み層
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
  def __init__(self, output_dim: int = 10, input_dim: int = 16 * 4 * 4, hidden_dims: list = [120, 84]):
    super().__init__()

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
  def __init__(self, input_dim: int = 16 * 5 * 5, hidden_dims: list = [120, 84]):
    super().__init__()

    # CIFAR-10用畳み込み層
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)

    # 特徴量抽出用の全結合層
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
  def __init__(self, input_dim: int = 16 * 4 * 4, hidden_dims: list = [120, 84], output_dim: int = 10):
    super().__init__()

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
