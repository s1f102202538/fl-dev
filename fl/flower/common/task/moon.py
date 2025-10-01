import copy
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class MoonContrastiveLearning:
  """FedMoon対比学習"""

  def __init__(
    self,
    mu: float = 1.0,
    temperature: float = 0.5,
    device: torch.device | None = None,
  ):
    """Moon対比学習を初期化

    Args:
        mu: 対比損失の重み
        temperature: 対比損失の温度
        device: 計算に使用するデバイス
    """
    self.mu = mu
    self.temperature = temperature
    self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 対比学習のためのモデル状態
    self.global_model = None
    self.previous_model = None

  def update_models(self, previous_model: nn.Module, global_model: nn.Module) -> None:
    """グローバルモデルと前回モデルの状態を更新（論文準拠）

    Args:
        previous_model: 前回のローカルモデル
        global_model: グローバルモデル
    """
    # 前回のローカルモデルを保存
    if self.previous_model is None:
      self.previous_model = copy.deepcopy(previous_model)
    else:
      self.previous_model.load_state_dict(previous_model.state_dict())

    # 前回モデルを評価モードに設定し、勾配を無効化
    self.previous_model.eval()
    for param in self.previous_model.parameters():
      param.requires_grad = False

    # グローバルモデルを保存
    self.global_model = copy.deepcopy(global_model)

    # グローバルモデルを評価モードに設定し、勾配を無効化
    self.global_model.eval()
    for param in self.global_model.parameters():
      param.requires_grad = False

  def forward_with_features(self, model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """特徴量と出力の両方を返すフォワードパス（論文準拠）

    Args:
        model: MoonModel（投影ヘッド付き）
        x: 入力テンソル

    Returns:
        (投影特徴量, 出力)のタプル
    """
    # 論文実装に準拠: (h, x_proj, y) を返すMoonModelを使用
    h, x_proj, outputs = model(x)
    return x_proj, outputs  # 対比学習には投影後特徴量(x_proj)を使用

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

    # CosineSimilarityを使用
    cos = torch.nn.CosineSimilarity(dim=-1)

    # 正例：local-global similarity
    posi = cos(local_features, global_features)
    logits = posi.reshape(-1, 1)

    # 負例：local-previous similarity
    nega = cos(local_features, prev_features)
    logits = torch.cat([logits, nega.reshape(-1, 1)], dim=1)

    # Temperature scaling
    logits /= self.temperature

    # ラベル：正例が0番目
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)

    # Cross entropy loss
    contrastive_loss = F.cross_entropy(logits, labels)

    return contrastive_loss

  def compute_enhanced_contrastive_loss(self, local_features: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """対比損失計算（論文準拠のメモリ最適化）

    Args:
        local_features: ローカルモデルからの特徴量
        images: 入力画像

    Returns:
        対比損失テンソル
    """
    if self.global_model is None or self.previous_model is None:
      return torch.tensor(0.0, device=self.device, requires_grad=True)

    # デバイス最適化: GPUの場合のみメモリ管理を行う
    if self.device.type == "cuda":
      # GPUの場合：メモリ最適化のためCUDA ↔ CPU移動
      self.global_model.to(self.device)
      global_features, _ = self.forward_with_features(self.global_model, images)
      self.global_model.to("cpu")  # メモリ節約のためCPUに戻す

      self.previous_model.to(self.device)
      with torch.no_grad():
        prev_features, _ = self.forward_with_features(self.previous_model, images)
      self.previous_model.to("cpu")  # メモリ節約のためCPUに戻す
    else:
      # CPUの場合：不要な移動を避けて直接計算
      with torch.no_grad():
        global_features, _ = self.forward_with_features(self.global_model, images)
        prev_features, _ = self.forward_with_features(self.previous_model, images)

    # CosineSimilarityを使用（論文準拠）
    cos = torch.nn.CosineSimilarity(dim=-1)

    # 正例：local-global similarity
    posi = cos(local_features, global_features)
    logits = posi.reshape(-1, 1)

    # 負例：local-previous similarity
    nega = cos(local_features, prev_features)
    logits = torch.cat([logits, nega.reshape(-1, 1)], dim=1)

    # Temperature scaling
    logits /= self.temperature

    # ラベル：正例が0番目
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)

    # Cross entropy loss
    contrastive_loss = F.cross_entropy(logits, labels)

    return contrastive_loss


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
    epochs: int,
    current_round: int = 1,
  ) -> float:
    """FedMoon対比学習による訓練

    Args:
        model: ニューラルネットワークモデル
        train_loader: 訓練データローダー
        lr: ベース学習率
        local_epochs: ローカルエポック数
        current_round: 現在のラウンド
        distillation_performed: 蒸留が実行されたかどうか

    Returns:
        平均訓練損失
    """
    model.to(self.device)
    criterion = nn.CrossEntropyLoss().to(self.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)

    print(f"[MOON] Round {current_round}: Using LR = {lr:.6f} (base={lr:.6f})")

    model.train()
    running_loss = 0.0
    total_batches = 0

    for epoch in range(epochs):
      for batch in train_loader:
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)

        optimizer.zero_grad()

        # フォワードパス（論文準拠の投影ヘッド使用）
        h, features, outputs = model(images)  # MoonModel: (h, x_proj, y)

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
    epochs: int,
    current_round: int = 1,
  ) -> float:
    """適応FedMoon対比学習による拡張訓練

    Args:
        model: ニューラルネットワークモデル
        train_loader: 訓練データローダー
        lr: ベース学習率
        local_epochs: ローカルエポック数
        current_round: 現在のラウンド
        distillation_performed: 蒸留が実行されたかどうか

    Returns:
        平均訓練損失
    """
    model.to(self.device)
    criterion = nn.CrossEntropyLoss().to(self.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)

    print(f"[Enhanced MOON] Round {current_round}: Using LR = {lr:.6f} (base={lr:.6f})")

    model.train()
    running_loss = 0.0
    total_batches = 0

    for epoch in range(epochs):
      epoch_loss_collector = []
      epoch_loss1_collector = []  # CE損失
      epoch_loss2_collector = []  # 対比損失

      for batch in train_loader:
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)

        optimizer.zero_grad()

        # フォワードパス（論文準拠の投影ヘッド使用）
        h, features, outputs = model(images)  # MoonModel: (h, x_proj, y)

        # 標準クロスエントロピー損失
        loss1 = criterion(outputs, labels)

        # 拡張対比損失（FedMoon）
        loss2 = torch.tensor(0.0, device=self.device)
        if self.moon_learner.global_model is not None and self.moon_learner.previous_model is not None:
          contrastive_loss = self.moon_learner.compute_enhanced_contrastive_loss(features, images)
          loss2 = self.moon_learner.mu * contrastive_loss

        # 総損失（論文準拠）
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss_collector.append(loss.item())
        epoch_loss1_collector.append(loss1.item())
        epoch_loss2_collector.append(loss2.item())
        total_batches += 1

      # エポックごとの損失ログ
      epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
      epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
      epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
      print(f"Epoch: {epoch} Loss: {epoch_loss:.6f} Loss1: {epoch_loss1:.6f} Loss2: {epoch_loss2:.6f}")

    avg_train_loss = running_loss / total_batches if total_batches > 0 else 0.0
    return avg_train_loss
