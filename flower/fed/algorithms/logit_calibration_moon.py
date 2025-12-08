import copy

import torch
from torch import nn

from ..models.base_model import BaseModel


class LogitCalibrationMoonContrastiveLearning:
  def __init__(
    self,
    mu: float = 3.0,
    temperature: float = 0.3,
    device: torch.device | None = None,
  ):
    self.mu = mu
    self.temperature = temperature
    self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 対比学習のためのモデル
    self.global_model = None
    self.previous_model = None  # 単一の前回モデルを保持

  def update_models(self, previous_model: BaseModel, global_model: BaseModel) -> None:
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
    print(f"[FedLC-MOON] Models updated: mu={self.mu}, temperature={self.temperature}, has_previous_model={has_previous}")

  def compute_contrastive_loss(self, features: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
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

  def _apply_fedlc_calibration(self, outputs: torch.Tensor, labels: torch.Tensor, class_counts: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    device = outputs.device
    n_classes = outputs.size(1)
    margin = torch.zeros_like(outputs)

    # FedLC の pairwise margin: Δ(y,i) = τ * (n_y^{-1/4} - n_i^{-1/4})
    class_counts = class_counts.float().to(device)
    class_counts_pow = class_counts.pow(-0.25)  # n_c^{-1/4}

    for i in range(n_classes):
      margin[:, i] = tau * (class_counts_pow[labels] - class_counts_pow[i])

    # correct class を補正
    outputs_calibrated = outputs + margin
    return outputs_calibrated

  def _compute_class_counts(self, train_loader, num_classes: int = 10):
    class_counts = torch.zeros(num_classes, dtype=torch.long)

    for batch in train_loader:
      labels = batch["label"]
      counts = torch.bincount(labels, minlength=num_classes)
      class_counts += counts

    class_counts[class_counts == 0] = 1
    return class_counts


class LogitCalibrationMoonTrainer:
  def __init__(
    self,
    moon_learner: LogitCalibrationMoonContrastiveLearning,
    device: torch.device | None = None,
  ):
    self.moon_learner = moon_learner
    self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def train_with_moon(
    self,
    model: BaseModel,
    train_loader,
    lr: float,
    epochs: int,
    args_optimizer: str = "sgd",
    weight_decay: float = 1e-5,
  ) -> float:
    model.to(self.device)
    criterion = nn.CrossEntropyLoss().to(self.device)

    # 元論文準拠：オプティマイザーの設定
    if args_optimizer == "adam":
      optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    else:  # SGD (default)
      optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=weight_decay)

    print(f"[FedLC-MOON] Using optimizer={args_optimizer}, LR={lr:.6f}, weight_decay={weight_decay}")

    model.train()
    running_loss = 0.0
    total_batches = 0

    class_counts = self.moon_learner._compute_class_counts(train_loader, 10)

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
        outputs_calibrated = self.moon_learner._apply_fedlc_calibration(outputs, labels, class_counts, tau=3.0)
        loss1 = criterion(outputs_calibrated, labels)

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

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
