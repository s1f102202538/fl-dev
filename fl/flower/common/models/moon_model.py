"""MOON用の投影ヘッド付きモデル"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoonModel(nn.Module):
  """MOON論文準拠の投影ヘッド付きモデル（state_dict互換性確保）"""

  def __init__(self, base_model: nn.Module, out_dim: int = 256, n_classes: int = 10):
    """
    Args:
        base_model: ベースとなるMiniCNNモデル
        out_dim: 投影ヘッドの出力次元
        n_classes: 分類クラス数
    """
    super().__init__()

    # ベースモデルを保持
    self.base_model = base_model

    # MiniCNNのfc1出力は128次元
    self.feature_dim = 128

    # 投影ヘッド
    self.l1 = nn.Linear(self.feature_dim, self.feature_dim)
    self.l2 = nn.Linear(self.feature_dim, out_dim)

    # 分類ヘッド
    self.l3 = nn.Linear(out_dim, n_classes)

  def forward_features(self, x: torch.Tensor) -> torch.Tensor:
    """ベースモデルから特徴量を抽出"""
    # MiniCNNのConv層を通す
    x = F.relu(self.base_model.conv1(x))  # type: ignore
    x = self.base_model.pool(x)  # type: ignore
    x = F.relu(self.base_model.conv2(x))  # type: ignore
    x = self.base_model.pool(x)  # type: ignore
    x = self.base_model.dropout(x)  # type: ignore

    # 平坦化して最初の全結合層を通す
    x = x.view(x.size(0), -1)
    features = F.relu(self.base_model.fc1(x))  # type: ignore
    features = self.base_model.dropout_fc(features)  # type: ignore

    return features

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    論文実装に準拠したフォワードパス

    Returns:
        h: ベース特徴量 (feature_dim次元)
        x_proj: 投影後特徴量 (out_dim次元) - 対比学習で使用
        y: 分類出力 (n_classes次元)
    """
    # ベース特徴量を取得
    h = self.forward_features(x)
    h = h.squeeze()

    # 投影ヘッドを通す
    projected = self.l1(h)
    projected = F.relu(projected)
    x_proj = self.l2(projected)

    # 分類出力
    y = self.l3(x_proj)

    return h, x_proj, y
