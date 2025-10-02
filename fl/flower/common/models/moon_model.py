"""MOON (Model-Contrastive Federated Learning) モデル実装

MOON論文の公式実装に準拠した投影ヘッド付きモデル
Reference: https://github.com/Xtra-Computing/MOON/blob/main/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mini_cnn import MiniCNN, MiniCNNFeatures


class MoonModel(nn.Module):
  """MOON公式実装準拠の投影ヘッド付きモデル"""

  def __init__(self, base_model: nn.Module | None = None, out_dim: int = 256, n_classes: int = 10):
    """
    Args:
        base_model: ベースとなるモデル（蒸留済みモデル等）
        out_dim: 投影ヘッドの出力次元（論文準拠で256）
        n_classes: 分類クラス数
    """
    super().__init__()

    # ベースモデルが提供された場合はその特徴量抽出器を使用
    if base_model is not None:
      print("MoonModel: 提供されたbase_modelから特徴量抽出器を作成")
      # MiniCNNの場合、特徴量抽出部分を複製
      try:
        if isinstance(base_model, MiniCNN):
          # MiniCNNから特徴量抽出器を作成
          features_extractor = MiniCNNFeatures()

          # 学習済みMiniCNNの重みを特徴量抽出器にコピー
          features_extractor.conv1.load_state_dict(base_model.conv1.state_dict())
          features_extractor.conv2.load_state_dict(base_model.conv2.state_dict())
          features_extractor.fc1.load_state_dict(base_model.fc1.state_dict())

          # Dropoutレイヤーは新しいインスタンスを使用（パラメータなし）
          # features_extractor.dropout と features_extractor.dropout_fc は既に初期化済み

          self.features = features_extractor
          print("MiniCNNの学習済み重みを特徴量抽出器にコピーしました")

          # 重みコピーの確認
          print(f"Conv1重み確認: {torch.allclose(features_extractor.conv1.weight, base_model.conv1.weight)}")
          print(f"FC1重み確認: {torch.allclose(features_extractor.fc1.weight, base_model.fc1.weight)}")

        else:
          # その他のモデル形式
          print(f"非MiniCNNモデル: {type(base_model)}")
          self.features = base_model
      except Exception as e:
        print(f"base_model処理に失敗、デフォルトを使用: {e}")
        self.features = MiniCNNFeatures()
    else:
      print("MoonModel: 新しいMiniCNNFeaturesを使用")
      # MOON公式実装準拠：特徴量抽出器を直接保持
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

  def predict(self, x: torch.Tensor) -> torch.Tensor:
    """評価用：分類出力のみを返す（CNNTaskとの互換性のため）

    Args:
        x: 入力テンソル

    Returns:
        y: 分類出力のみ (10次元)
    """
    _, _, y = self.forward(x)
    return y
