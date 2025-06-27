from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.adam import Adam
from torch.utils.data import DataLoader


class Distillation:
  def __init__(self, studentModel: nn.Module, train_data: DataLoader, soft_target_losses: List[torch.Tensor]) -> None:
    self.studentModel = studentModel
    self.train_data = train_data
    self.soft_target_losses = soft_target_losses

  def train_knowledge_distillation(
    self, epochs: int, learning_rate: float, T: float, soft_target_loss_weight: float, ce_loss_weight: float, device: torch.device
  ) -> nn.Module:
    # 損失関数と最適化アルゴリズム
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    optimizer = Adam(self.studentModel.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    self.studentModel.train()  # 生徒モデルを学習モードに設定

    for epoch in range(epochs):
      running_loss = 0.0
      for (inputs, labels), soft_targets in zip(self.train_data, self.soft_target_losses):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        student_logits = self.studentModel(inputs)

        soft_targets = F.softmax(soft_targets / T, dim=1)
        soft_prob = F.log_softmax(student_logits / T, dim=1)

        # KL損失とCE損失の組み合わせ
        distillation_loss = kl_loss(soft_prob, soft_targets) * (T**2)
        student_loss = ce_loss(student_logits, labels)
        loss = soft_target_loss_weight * distillation_loss + ce_loss_weight * student_loss

        # 損失を逆伝搬
        loss.backward()

        # 勾配クリッピング追加
        clip_grad_norm_(self.studentModel.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item()

      epoch_loss = running_loss / len(self.train_data)
      scheduler.step(epoch_loss)

    return self.studentModel
