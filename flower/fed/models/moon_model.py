"""MOON (Model-Contrastive Federated Learning) モデル実装

MOON論文の公式実装に準拠した投影ヘッド付きモデル
Reference: https://github.com/Xtra-Computing/MOON/blob/main/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mini_cnn import MiniCNNFeatures


class MoonModel(nn.Module):
  """MOON公式実装準拠の投影ヘッド付きモデル"""

  def __init__(self, out_dim: int = 256, n_classes: int = 10):
    """
    Args:
        out_dim: 投影ヘッドの出力次元（論文準拠で256）
        n_classes: 分類クラス数
    """
    super().__init__()

    # MOON公式実装準拠：特徴量抽出器を直接保持
    # 注意：蒸留でもMoonModelを一貫使用するため、base_modelの複雑な処理は不要
    self.features = MiniCNNFeatures()

    # 特徴量次元（MiniCNNFeatures の fc1 出力）
    num_ftrs = 128

    # 投影ヘッド
    self.l1 = nn.Linear(num_ftrs, num_ftrs)  # 128 -> 128
    self.l2 = nn.Linear(num_ftrs, out_dim)  # 128 -> 256

    # 分類ヘッド（投影特徴量から分類）
    self.l3 = nn.Linear(out_dim, n_classes)  # 256 -> 10

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MOON公式実装準拠のフォワードパス
    Reference: https://github.com/Xtra-Computing/MOON/blob/main/model.py#L577-587

    Returns:
        h: ベース特徴量 (128次元)
        proj: 投影後特徴量 (256次元) - 対比学習用
        y: 分類出力 (10次元) - 投影特徴量から分類
    """
    # 特徴量抽出
    h = self.features(x)
    # バッチ次元を保持しながら不要な次元のみを削除
    if h.dim() > 2:
      h = h.view(h.size(0), -1)  # (batch_size, features)

    # 投影ヘッドを通す
    proj = self.l1(h)
    proj = F.relu(proj)
    proj = self.l2(proj)

    # 分類出力（投影特徴量から分類）
    y = self.l3(proj)

    return h, proj, y
