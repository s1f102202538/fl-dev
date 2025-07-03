from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.adam import Adam
from torch.utils.data import DataLoader


class Distillation:
  def __init__(self, studentModel: nn.Module, public_data: DataLoader, soft_targets: List[torch.Tensor]) -> None:
    self.studentModel = studentModel
    self.public_data = public_data  # 公開データセット
    self.soft_targets = soft_targets  # ソフトターゲット

    # ソフトターゲットがリストかどうかを判定
    self.is_batch_list = isinstance(soft_targets, list) and len(soft_targets) > 0 and isinstance(soft_targets[0], torch.Tensor)

  def train_knowledge_distillation(
    self, epochs: int, learning_rate: float, T: float, soft_target_loss_weight: float, ce_loss_weight: float, device: torch.device
  ) -> nn.Module:
    # 損失関数と最適化アルゴリズム
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    optimizer = Adam(self.studentModel.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    self.studentModel.train()  # 生徒モデルを学習モードに設定

    if self.is_batch_list:
      self.soft_targets = [batch.to(device) for batch in self.soft_targets]

      for epoch in range(epochs):
        running_loss = 0.0

        # データローダーとソフトターゲットバッチを同時にイテレート
        for batch_data, soft_batch in zip(self.public_data, self.soft_targets):
          inputs = batch_data["image"].to(device)
          labels = batch_data["label"].to(device)

          # print("Soft targets stats: mean", soft_batch.mean().item(), "std", soft_batch.std().item())
          # print("Soft targets softmax (first sample):", F.softmax(soft_batch[0] / T, dim=0))

          optimizer.zero_grad()

          student_logits = self.studentModel(inputs)

          # ロジットのクリッピング
          soft_batch_clipped = torch.clamp(soft_batch, min=-50, max=50)
          student_logits_clipped = torch.clamp(student_logits, min=-50, max=50)

          soft_targets_temp = F.softmax(soft_batch_clipped / T, dim=1)
          soft_prob = F.log_softmax(student_logits_clipped / T, dim=1)

          # softmaxの後にepsでクリップ
          eps = 1e-10
          soft_targets_temp = torch.clamp(soft_targets_temp, min=eps, max=1.0 - eps)

          # KL損失とCE損失の組み合わせ
          distillation_loss = kl_loss(soft_prob, soft_targets_temp) * (T**2)
          student_loss = ce_loss(student_logits, labels)

          # NaN/Inf の確認
          if torch.isnan(distillation_loss) or torch.isinf(distillation_loss):
            print("Warning: Invalid distillation loss detected, using only CE loss")
            loss = student_loss
          elif torch.isnan(student_loss) or torch.isinf(student_loss):
            print("Warning: Invalid student loss detected, using only distillation loss")
            loss = distillation_loss
          else:
            loss = soft_target_loss_weight * distillation_loss + ce_loss_weight * student_loss

          # 損失を逆伝搬
          loss.backward()

          # 勾配クリッピング追加
          clip_grad_norm_(self.studentModel.parameters(), max_norm=1.0)

          optimizer.step()
          running_loss += loss.item()

        epoch_loss = running_loss / len(self.public_data)
        scheduler.step(epoch_loss)
        print(f"Knowledge Distillation (Batch List) Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}")

    return self.studentModel
