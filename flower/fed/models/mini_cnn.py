from typing import override

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_model import BaseModel


class MiniCNN(BaseModel):
  def __init__(self, n_classes: int = 10, dropout_rate: float = 0.2) -> None:
    super().__init__()

    # CIFAR-10用畳み込み層（3チャンネル入力、32x32画像）
    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.dropout = nn.Dropout2d(dropout_rate)

    # 全結合層
    self.fc1 = nn.Linear(128 * 4 * 4, 128)
    self.fc2 = nn.Linear(128, n_classes)

  def forward(self, x: Tensor) -> Tensor:
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
  def __init__(self, n_classes: int = 10, dropout_rate: float = 0.2) -> None:
    super().__init__()

    # MNIST用畳み込み層（1チャンネル入力、28x28画像）
    self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.dropout = nn.Dropout2d(dropout_rate)

    # 全結合層
    self.fc1 = nn.Linear(32 * 7 * 7, 128)
    self.fc2 = nn.Linear(128, n_classes)

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
  def __init__(self, dropout_rate: float = 0.2) -> None:
    super().__init__()

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
  def __init__(self, dropout_rate: float = 0.2) -> None:
    super().__init__()

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
