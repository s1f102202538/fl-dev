import copy

import torch
from torch import nn

from ..models.base_model import BaseModel


class CsdBasedMoonContrastiveLearning:
  def __init__(
    self,
    mu: float = 3.0,
    temperature: float = 0.3,
    lambda_csd: float = 0.5,
    csd_temperature: float = 0.3,
    csd_margin: float = 0.1,
    device: torch.device | None = None,
  ):
    self.mu = mu
    self.temperature = temperature
    self.lambda_csd = lambda_csd
    self.csd_temperature = csd_temperature
    self.csd_margin = csd_margin
    self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    self.global_model = None
    self.previous_model = None

    self.class_prototypes: torch.Tensor | None = None

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
    self.previous_model.to(self.device)

    # グローバルモデルを保存
    if self.global_model is None:
      self.global_model = copy.deepcopy(global_model)
    else:
      self.global_model.load_state_dict(global_model.state_dict())

    # グローバルモデルを評価モードに設定し、勾配を無効化
    self.global_model.eval()
    for param in self.global_model.parameters():
      param.requires_grad = False
    self.global_model.to(self.device)

    has_previous = self.previous_model is not None
    print(f"[CSD-MOON] Models updated: mu={self.mu}, temperature={self.temperature}, lambda_csd={self.lambda_csd}, has_previous_model={has_previous}")

  def set_class_prototypes(self, class_prototypes: torch.Tensor) -> None:
    self.class_prototypes = class_prototypes.to(self.device)
    print(f"[CSD] Class prototypes set: shape={self.class_prototypes.shape}")

  def compute_csd_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if self.class_prototypes is None:
      return torch.tensor(0.0, device=self.device, requires_grad=True)

    # ロジットを正規化
    logits_normalized = torch.nn.functional.normalize(logits, p=2, dim=1)
    prototypes_normalized = torch.nn.functional.normalize(self.class_prototypes, p=2, dim=1)

    # コサイン類似度を計算 [batch_size, n_classes]
    similarity = torch.mm(logits_normalized, prototypes_normalized.t())
    similarity = similarity / self.csd_temperature

    # similarity: [B, C]
    B = similarity.size(0)
    C = similarity.size(1)

    # 正例クラスにマージンを追加
    pos_mask = torch.zeros_like(similarity, dtype=torch.bool)
    pos_mask.scatter_(1, labels.unsqueeze(1), True)
    similarity[pos_mask] += self.csd_margin

    # 正例クラス類似度
    pos_sim = similarity.gather(1, labels.unsqueeze(1))  # [B, 1]

    # 負例マスク：正解クラス以外
    neg_mask = torch.arange(C, device=self.device).unsqueeze(0) != labels.unsqueeze(1)  # [B, C]
    neg_sim = similarity[neg_mask].view(B, C - 1)  # [B, C-1]

    # 負例を遠ざける：負例の平均類似度を下げたい
    neg_loss = torch.mean(torch.logsumexp(neg_sim, dim=1))

    # 正例を近づける
    pos_loss = -torch.mean(pos_sim)

    csd_loss = pos_loss + neg_loss

    return csd_loss

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


class CsdBasedMoonTrainer:
  def __init__(
    self,
    moon_learner: CsdBasedMoonContrastiveLearning,
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
    class_prototypes: torch.Tensor | None = None,
  ) -> float:
    # クラスプロトタイプを設定
    if class_prototypes is not None:
      self.moon_learner.set_class_prototypes(class_prototypes)
    model.to(self.device)
    criterion = nn.CrossEntropyLoss().to(self.device)

    # 元論文準拠：オプティマイザーの設定
    if args_optimizer == "adam":
      optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    else:  # SGD (default)
      optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=weight_decay)

    print(f"[CSD-MOON] Using optimizer={args_optimizer}, LR={lr:.6f}, weight_decay={weight_decay}")

    model.train()
    running_loss = 0.0
    total_batches = 0

    for epoch in range(epochs):
      epoch_loss_collector = []
      epoch_loss1_collector = []  # CE損失
      epoch_loss2_collector = []  # 対比損失
      epoch_loss3_collector = []  # CSD損失

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

        # CSD損失（クラスプロトタイプとの類似度）
        loss3 = torch.tensor(0.0, device=self.device)
        if self.moon_learner.class_prototypes is not None:
          csd_loss = self.moon_learner.compute_csd_loss(outputs, labels)
          loss3 = self.moon_learner.lambda_csd * csd_loss

        # 総損失
        total_loss = loss1 + loss2 + loss3

        # NaN detection and handling
        if torch.isnan(total_loss) or torch.isinf(total_loss):
          print(f"Warning: Loss became NaN/Inf. loss1={loss1.item()}, loss2={loss2.item()}, loss3={loss3.item()}")
          continue  # Skip this batch

        total_loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += total_loss.item()
        epoch_loss_collector.append(total_loss.item())
        epoch_loss1_collector.append(loss1.item())
        epoch_loss2_collector.append(loss2.item())
        epoch_loss3_collector.append(loss3.item())
        total_batches += 1

      # エポックごとの損失ログ
      if len(epoch_loss_collector) > 0:
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        epoch_loss3 = sum(epoch_loss3_collector) / len(epoch_loss3_collector)

        # CSD損失のログ出力
        if self.moon_learner.class_prototypes is not None:
          print(
            f"Epoch: {epoch + 1}/{epochs}, Total Loss: {epoch_loss:.6f}, CE Loss: {epoch_loss1:.6f}, Contrastive Loss: {epoch_loss2:.6f}, CSD Loss: {epoch_loss3:.6f}"
          )
        else:
          print(f"Epoch: {epoch + 1}/{epochs}, Total Loss: {epoch_loss:.6f}, CE Loss: {epoch_loss1:.6f}, Contrastive Loss: {epoch_loss2:.6f}")

    avg_train_loss = running_loss / total_batches if total_batches > 0 else 0.0
    return avg_train_loss
