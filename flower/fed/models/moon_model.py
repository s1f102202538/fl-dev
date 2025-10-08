from typing import override

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_model import BaseModel
from .mini_cnn_header import MiniCNN_header, MiniCNNMNIST_header


class MoonModel(BaseModel):
  def __init__(self, base_model: str, out_dim: int = 256, n_classes: int = 10):
    super().__init__()

    if base_model == "mini-cnn":
      self.features = MiniCNN_header()
      self.num_ftrs = self.features.num_ftrs
    elif base_model == "mini-cnn-minist":
      self.features = MiniCNNMNIST_header()
      self.num_ftrs = self.features.num_ftrs

    # 投影ヘッド
    self.l1 = nn.Linear(self.num_ftrs, self.num_ftrs)
    self.l2 = nn.Linear(self.num_ftrs, out_dim)

    # 分類層
    self.l3 = nn.Linear(out_dim, n_classes)

  def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # 特徴量抽出
    h = self.features(x)
    h = h.squeeze()

    # 投影ヘッドを通す
    proj = self.l1(h)
    proj = F.relu(proj)
    proj = self.l2(proj)

    # 分類出力
    y = self.l3(proj)

    return h, proj, y

  @override
  def predict(self, x: Tensor) -> Tensor:
    _, _, y = self.forward(x)
    return y
