"""対比学習のためのFedMoonアルゴリズム実装"""

import copy
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class MoonContrastiveLearning:
  """FedMoon対比学習実装"""

  def __init__(
    self,
    mu: float = 5.0,
    temperature: float = 0.5,
    adaptive_mu: bool = True,
    min_mu: float = 1.0,
    max_mu: float = 10.0,
    device: torch.device | None = None,
  ):
    """Moon対比学習を初期化

    Args:
        mu: 対比損失の重み
        temperature: 対比損失の温度
        adaptive_mu: 適応的muを使用するかどうか
        min_mu: muの最小値
        max_mu: muの最大値
        device: 計算に使用するデバイス
    """
    self.mu = mu
    self.temperature = temperature
    self.adaptive_mu = adaptive_mu
    self.min_mu = min_mu
    self.max_mu = max_mu
    self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 対比学習のためのモデル状態
    self.global_model = None
    self.previous_model = None

    # 適応学習のための性能追跡
    self.performance_history = []
    self.max_history_length = 5

  def update_models(self, previous_model: nn.Module, global_model: nn.Module) -> None:
    """グローバルモデルと前回モデルの状態を更新

    Args:
        previous_model: 前回のローカルモデル
        global_model: グローバルモデル
    """
    # 前回のローカルモデルを保存
    if self.previous_model is None:
      self.previous_model = copy.deepcopy(previous_model)
    else:
      self.previous_model.load_state_dict(previous_model.state_dict())

    # グローバルモデルを保存
    self.global_model = copy.deepcopy(global_model)

  def forward_with_features(self, model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """特徴量と出力の両方を返すフォワードパス

    Args:
        model: ニューラルネットワークモデル
        x: 入力テンソル

    Returns:
        (特徴量, 出力)のタプル
    """
    # MiniCNN構造に従った最終層前の特徴量抽出
    x = F.relu(model.conv1(x))  # type: ignore
    x = model.pool(x)  # type: ignore  # 28x28 -> 14x14

    x = F.relu(model.conv2(x))  # type: ignore
    x = model.pool(x)  # type: ignore  # 14x14 -> 7x7
    x = model.dropout(x)  # type: ignore

    # 平坦化して最終層前の特徴量を取得
    x = x.view(x.size(0), -1)
    features = F.relu(model.fc1(x))  # type: ignore
    features = model.dropout_fc(features)  # type: ignore
    outputs = model.fc2(features)  # type: ignore

    return features, outputs

  def compute_contrastive_loss(self, local_features: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """ローカル、グローバル、前回モデル間の対比損失を計算

    Args:
        local_features: ローカルモデルからの特徴量
        images: 入力画像

    Returns:
        対比損失テンソル
    """
    if self.global_model is None or self.previous_model is None:
      return torch.tensor(0.0, device=self.device, requires_grad=True)

    # グローバルモデルと前回モデルから特徴量を取得
    with torch.no_grad():
      global_features, _ = self.forward_with_features(self.global_model, images)
      prev_features, _ = self.forward_with_features(self.previous_model, images)

    # 特徴量を正規化
    local_features = F.normalize(local_features, dim=1)
    global_features = F.normalize(global_features, dim=1)
    prev_features = F.normalize(prev_features, dim=1)

    # 類似度を計算
    pos_sim = torch.sum(local_features * global_features, dim=1) / self.temperature  # 正例ペア
    neg_sim = torch.sum(local_features * prev_features, dim=1) / self.temperature  # 負例ペア

    # 対比損失: -log(exp(pos) / (exp(pos) + exp(neg)))
    logits = torch.stack([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
    contrastive_loss = F.cross_entropy(logits, labels)

    return contrastive_loss

  def compute_enhanced_contrastive_loss(self, local_features: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """数値安定性を向上させた拡張対比損失

    Args:
        local_features: ローカルモデルからの特徴量
        images: 入力画像

    Returns:
        拡張対比損失テンソル
    """
    if self.global_model is None or self.previous_model is None:
      return torch.tensor(0.0, device=self.device, requires_grad=True)

    # グローバルモデルと前回モデルから特徴量を取得
    with torch.no_grad():
      global_features, _ = self.forward_with_features(self.global_model, images)
      prev_features, _ = self.forward_with_features(self.previous_model, images)

    # 数値安定性を考慮した拡張正規化
    eps = 1e-8
    local_features = F.normalize(local_features + eps, dim=1)
    global_features = F.normalize(global_features + eps, dim=1)
    prev_features = F.normalize(prev_features + eps, dim=1)

    # 温度スケーリングによる類似度計算
    pos_sim = torch.sum(local_features * global_features, dim=1) / self.temperature
    neg_sim = torch.sum(local_features * prev_features, dim=1) / self.temperature

    # 数値安定性を考慮した拡張対比損失
    pos_sim = torch.clamp(pos_sim, min=-10, max=10)
    neg_sim = torch.clamp(neg_sim, min=-10, max=10)

    logits = torch.stack([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
    contrastive_loss = F.cross_entropy(logits, labels)

    return contrastive_loss

  def update_adaptive_parameters(self, current_round: int) -> None:
    """ラウンドと性能履歴に基づく適応パラメータ更新

    Args:
        current_round: 現在の訓練ラウンド
    """
    if self.adaptive_mu and len(self.performance_history) >= 2:
      # 性能トレンドを計算
      recent_losses = [h["train_loss"] for h in self.performance_history[-3:]]
      if len(recent_losses) >= 2:
        loss_trend = recent_losses[-1] - recent_losses[0]

        # 損失が増加している場合はmuを増加（より強い正則化が必要）
        if loss_trend > 0:
          self.mu = min(self.max_mu, self.mu * 1.1)
        else:
          self.mu = max(self.min_mu, self.mu * 0.95)

    # ラウンドベースの適応
    if current_round > 10:
      # 後期ラウンドでは対比学習の重みを徐々に減少
      decay_factor = 0.99
      self.mu = max(self.min_mu, self.mu * decay_factor)

  def track_performance(self, train_loss: float, distillation_performed: bool, current_round: int) -> None:
    """適応学習のための性能追跡

    Args:
        train_loss: このラウンドの訓練損失
        distillation_performed: 蒸留が実行されたかどうか
        current_round: 現在の訓練ラウンド
    """
    self.performance_history.append({"train_loss": train_loss, "distillation_performed": distillation_performed, "round": current_round})

    # 最近の履歴のみを保持
    if len(self.performance_history) > self.max_history_length:
      self.performance_history.pop(0)

  def get_adaptive_mu(self, distillation_performed: bool) -> float:
    """現在の状態に基づく適応mu値を取得

    Args:
        distillation_performed: 蒸留が実行されたかどうか

    Returns:
        適応mu値
    """
    adaptive_mu = self.mu
    if distillation_performed:
      # 蒸留実行時は対比学習の重みを削減
      adaptive_mu *= 0.7
    return adaptive_mu


class MoonTrainer:
  """FedMoon訓練実装"""

  def __init__(
    self,
    moon_learner: MoonContrastiveLearning,
    device: torch.device | None = None,
  ):
    """Moonトレーナーを初期化

    Args:
        moon_learner: Moon対比学習インスタンス
        device: 計算に使用するデバイス
    """
    self.moon_learner = moon_learner
    self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def train_with_moon(
    self,
    model: nn.Module,
    train_loader,
    lr: float,
    local_epochs: int,
  ) -> float:
    """FedMoon対比学習による訓練

    Args:
        model: ニューラルネットワークモデル
        train_loader: 訓練データローダー
        lr: 学習率
        local_epochs: ローカルエポック数

    Returns:
        平均訓練損失
    """
    model.to(self.device)
    criterion = nn.CrossEntropyLoss().to(self.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)

    model.train()
    running_loss = 0.0
    total_batches = 0

    for epoch in range(local_epochs):
      for batch in train_loader:
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)

        optimizer.zero_grad()

        # フォワードパス
        features, outputs = self.moon_learner.forward_with_features(model, images)

        # 標準クロスエントロピー損失
        ce_loss = criterion(outputs, labels)

        # 対比損失（FedMoon）
        contrastive_loss = 0.0
        if self.moon_learner.global_model is not None and self.moon_learner.previous_model is not None:
          contrastive_loss = self.moon_learner.compute_contrastive_loss(features, images)

        # 総損失
        total_loss = ce_loss + self.moon_learner.mu * contrastive_loss

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        total_batches += 1

    avg_train_loss = running_loss / total_batches if total_batches > 0 else 0.0
    return avg_train_loss

  def train_with_enhanced_moon(
    self,
    model: nn.Module,
    train_loader,
    lr: float,
    local_epochs: int,
    distillation_performed: bool = False,
  ) -> float:
    """適応FedMoon対比学習による拡張訓練

    Args:
        model: ニューラルネットワークモデル
        train_loader: 訓練データローダー
        lr: 学習率
        local_epochs: ローカルエポック数
        distillation_performed: 蒸留が実行されたかどうか

    Returns:
        平均訓練損失
    """
    model.to(self.device)
    criterion = nn.CrossEntropyLoss().to(self.device)

    # 蒸留実行の有無に基づく異なるオプティマイザパラメータ
    if distillation_performed:
      # 蒸留後のファインチューニングのため学習率を下げる
      optimizer = torch.optim.SGD(model.parameters(), lr=lr * 0.7, momentum=0.9, weight_decay=1e-5)
    else:
      optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)

    model.train()
    running_loss = 0.0
    total_batches = 0
    contrastive_loss_sum = 0.0

    for epoch in range(local_epochs):
      for batch in train_loader:
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)

        optimizer.zero_grad()

        # フォワードパス
        features, outputs = self.moon_learner.forward_with_features(model, images)

        # 標準クロスエントロピー損失
        ce_loss = criterion(outputs, labels)

        # 拡張対比損失（FedMoon）
        contrastive_loss = 0.0
        if self.moon_learner.global_model is not None and self.moon_learner.previous_model is not None:
          contrastive_loss = self.moon_learner.compute_enhanced_contrastive_loss(features, images)

        # 適応総損失
        adaptive_mu = self.moon_learner.get_adaptive_mu(distillation_performed)
        total_loss = ce_loss + adaptive_mu * contrastive_loss

        total_loss.backward()

        # 安定性のためのグラデーションクリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += total_loss.item()
        contrastive_loss_sum += contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss
        total_batches += 1

    avg_train_loss = running_loss / total_batches if total_batches > 0 else 0.0
    avg_contrastive_loss = contrastive_loss_sum / total_batches if total_batches > 0 else 0.0

    print(f"拡張MOON訓練 - CE損失: {avg_train_loss - avg_contrastive_loss * self.moon_learner.mu:.4f}, 対比損失: {avg_contrastive_loss:.4f}")

    return avg_train_loss
