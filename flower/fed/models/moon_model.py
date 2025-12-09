from typing import override

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_model import BaseModel
from .mini_cnn import MiniCNN, MiniCNN_header, MiniCNNMNIST, MiniCNNMNIST_header
from .simple_cnn import SimpleCNN, SimpleCNN_header, SimpleCNNMNIST, SimpleCNNMNIST_header


class ModelFedCon(BaseModel):
  def __init__(self, base_model: str, out_dim: int = 256, n_classes: int = 10):
    super().__init__()

    if base_model == "mini-cnn":
      self.features = MiniCNN_header()
      num_ftrs = self.features.num_ftrs  # 128
    elif base_model == "mini-cnn-mnist":
      self.features = MiniCNNMNIST_header()
      num_ftrs = self.features.num_ftrs  # 128
    elif base_model == "simple-cnn":
      self.features = SimpleCNN_header()
      num_ftrs = self.features.num_ftrs  # 84
    elif base_model == "simple-cnn-mnist":
      self.features = SimpleCNNMNIST_header()
      num_ftrs = self.features.num_ftrs  # 84
    else:
      raise ValueError(f"Unsupported base_model: {base_model}")

    # projection MLP
    self.l1 = nn.Linear(num_ftrs, out_dim)
    self.l2 = nn.Linear(out_dim, out_dim)

    # last layer
    self.l3 = nn.Linear(num_ftrs, n_classes)

  def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    h = self.features(x)
    h = h.squeeze()

    proj = self.l1(h)
    proj = F.relu(proj)
    proj = self.l2(proj)

    y = self.l3(h)
    return h, proj, y

  @override
  def predict(self, x: Tensor) -> Tensor:
    _, _, y = self.forward(x)
    return y


class ModelFedCon_noheader(BaseModel):
  def __init__(self, base_model: str, n_classes: int = 10):
    super().__init__()

    if base_model == "mini-cnn":
      self.backbone = MiniCNN(n_classes)
    elif base_model == "mini-cnn-mnist":
      self.backbone = MiniCNNMNIST(n_classes)
    elif base_model == "simple-cnn":
      self.backbone = SimpleCNN(n_classes)
    elif base_model == "simple-cnn-mnist":
      self.backbone = SimpleCNNMNIST(n_classes)
    else:
      raise ValueError(f"Unsupported base_model: {base_model}")

  def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # backbone（完全なモデル）を使用
    y = self.backbone(x)

    # projection headがない場合の特徴量抽出
    # 最終層の前の特徴量を取得するのが理想だが、ここでは単純化して最終出力を特徴量として使用
    h = y
    proj = y

    return h, proj, y

  @override
  def predict(self, x: Tensor) -> Tensor:
    y = self.backbone(x)
    return y


# 後方互換性のためのエイリアス
MoonModel = ModelFedCon
