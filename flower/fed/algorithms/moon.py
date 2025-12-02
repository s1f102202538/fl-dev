import copy

import torch
from torch import nn

from ..models.base_model import BaseModel


class MoonContrastiveLearning:
  """FedMoon対比学習"""

  def __init__(
    self,
    mu: float = 5.0,
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

    # 対比学習のためのモデル
    self.global_model = None
    self.previous_model = None  # 単一の前回モデルを保持

  def update_models(self, previous_model: BaseModel, global_model: BaseModel) -> None:
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

    # ローカルを評価モードに設定し、勾配を無効化
    self.previous_model.eval()
    for param in self.previous_model.parameters():
      param.requires_grad = False
    self.previous_model.to(self.device)  # デバイスに移動

    # グローバルモデルを保存
    if self.global_model is None:
      self.global_model = copy.deepcopy(global_model)
    else:
      self.global_model.load_state_dict(global_model.state_dict())

    # グローバルモデルを評価モードに設定し、勾配を無効化
    self.global_model.eval()
    for param in self.global_model.parameters():
      param.requires_grad = False
    self.global_model.to(self.device)  # デバイスに移動

    has_previous = self.previous_model is not None
    print(f"MOON models updated: mu={self.mu}, temperature={self.temperature}, has_previous_model={has_previous}")

  def compute_contrastive_loss(self, features: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """対比損失計算

    Args:
        features: ローカルモデルからの投影特徴量
        images: 入力画像

    Returns:
        対比損失テンソル
    """
    if self.global_model is None or self.previous_model is None:
      return torch.tensor(0.0, device=self.device, requires_grad=True)

    # グローバルモデルから特徴量を取得
    with torch.no_grad():
      _, global_features, _ = self.global_model(images)

    # 負例用の前回モデルから特徴量を取得
    with torch.no_grad():
      _, prev_features, _ = self.previous_model(images)

    # L2正規化（対比学習の安定性向上）
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    global_features = torch.nn.functional.normalize(global_features, p=2, dim=1)
    prev_features = torch.nn.functional.normalize(prev_features, p=2, dim=1)

    # CosineSimilarityを使用
    cos = torch.nn.CosineSimilarity(dim=-1)

    # 正例：local-global similarity
    posi = cos(features, global_features)
    logits = posi.reshape(-1, 1)

    # 負例：local-previous similarity
    nega = cos(features, prev_features)
    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

    # Temperature scaling
    logits /= self.temperature

    # NaN/Inf check
    if torch.isnan(logits).any() or torch.isinf(logits).any():
      print("Warning: NaN/Inf detected in logits. Returning zero tensor.")
      return torch.tensor(0.0, device=self.device, requires_grad=True)

    # ラベル：正例が0番目
    labels = torch.zeros(images.size(0), dtype=torch.long, device=self.device)

    # Cross entropy loss
    criterion = nn.CrossEntropyLoss().to(self.device)
    contrastive_loss = criterion(logits, labels)

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
    model: BaseModel,
    train_loader,
    lr: float,
    epochs: int,
    args_optimizer: str = "sgd",
  ) -> float:
    """FedMoon対比学習による訓練

    Args:
        model: ニューラルネットワークモデル
        train_loader: 訓練データローダー
        lr: 学習率
        epochs: エポック数
        args_optimizer: オプティマイザータイプ
        weight_decay: 重み減衰

    Returns:
        平均訓練損失
    """
    model.to(self.device)
    criterion = nn.CrossEntropyLoss().to(self.device)

    # 元論文準拠：オプティマイザーの設定
    if args_optimizer == "adam":
      optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:  # SGD (default)
      optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)

    print(f"[MOON] Using optimizer={args_optimizer}, LR={lr:.6f}")

    model.train()
    running_loss = 0.0
    total_batches = 0

    for epoch in range(epochs):
      epoch_loss_collector = []
      epoch_loss1_collector = []  # CE損失
      epoch_loss2_collector = []  # 対比損失

      for batch in train_loader:
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device).long()

        optimizer.zero_grad()

        # フォワードパス
        h, proj, outputs = model(images)

        # 標準クロスエントロピー損失
        loss1 = criterion(outputs, labels)

        # 対比損失
        loss2 = torch.tensor(0.0, device=self.device)
        if self.moon_learner.global_model is not None and self.moon_learner.previous_model is not None:
          contrastive_loss = self.moon_learner.compute_contrastive_loss(proj, images)
          loss2 = self.moon_learner.mu * contrastive_loss

        # 総損失
        total_loss = loss1 + loss2

        # NaN detection and handling
        if torch.isnan(total_loss) or torch.isinf(total_loss):
          print(f"Warning: Loss became NaN/Inf. loss1={loss1.item()}, loss2={loss2.item()}")
          continue  # Skip this batch

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += total_loss.item()
        epoch_loss_collector.append(total_loss.item())
        epoch_loss1_collector.append(loss1.item())
        epoch_loss2_collector.append(loss2.item())
        total_batches += 1

      # エポックごとの損失ログ
      if len(epoch_loss_collector) > 0:
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        print(f"Epoch: {epoch + 1}/{epochs}, Total Loss: {epoch_loss:.6f}, CE Loss: {epoch_loss1:.6f}, Contrastive Loss: {epoch_loss2:.6f}")

    avg_train_loss = running_loss / total_batches if total_batches > 0 else 0.0
    return avg_train_loss
