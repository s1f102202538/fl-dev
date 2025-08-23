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

    # バッチ数の検証
    if self.is_batch_list:
      expected_batches = len(self.public_data)
      actual_batches = len(self.soft_targets)
      if expected_batches != actual_batches:
        print(f"Warning: Batch count mismatch. Public data: {expected_batches}, Soft targets: {actual_batches}")
        # より少ない方に合わせる
        min_batches = min(expected_batches, actual_batches)
        print(f"Using first {min_batches} batches for distillation")

  def train_knowledge_distillation(
    self, epochs: int, learning_rate: float, T: float, soft_target_loss_weight: float, ce_loss_weight: float, device: torch.device
  ) -> nn.Module:
    # 損失関数と最適化アルゴリズム
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    optimizer = Adam(self.studentModel.parameters(), lr=learning_rate)

    # 短いエポック数の場合はスケジューラを使わない
    use_scheduler = epochs > 5
    scheduler = None
    if use_scheduler:
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    self.studentModel.train()  # 生徒モデルを学習モードに設定

    if self.is_batch_list:
      # ソフトターゲットをデバイスに移動
      self.soft_targets = [batch.to(device) for batch in self.soft_targets]

      # バッチ数の確認
      min_batches = min(len(self.public_data), len(self.soft_targets))

      for epoch in range(epochs):
        running_loss = 0.0
        batch_count = 0

        # データローダーとソフトターゲットバッチを同時にイテレート
        for batch_idx, batch_data in enumerate(self.public_data):
          if batch_idx >= min_batches:
            break

          # データの形式を柔軟に処理
          if isinstance(batch_data, dict):
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
          elif isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
            inputs, labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)
          else:
            print(f"Warning: Unexpected batch data format: {type(batch_data)}")
            continue

          soft_batch = self.soft_targets[batch_idx]

          # バッチサイズの確認
          if inputs.size(0) != soft_batch.size(0):
            print(f"Warning: Batch size mismatch. Input: {inputs.size(0)}, Soft targets: {soft_batch.size(0)}")
            # より小さいバッチサイズに合わせる
            min_batch_size = min(inputs.size(0), soft_batch.size(0))
            inputs = inputs[:min_batch_size]
            labels = labels[:min_batch_size]
            soft_batch = soft_batch[:min_batch_size]

          optimizer.zero_grad()

          student_logits = self.studentModel(inputs)

          # ロジットのクリッピング
          soft_batch_clipped = torch.clamp(soft_batch, min=-20, max=20)
          student_logits_clipped = torch.clamp(student_logits, min=-20, max=20)

          # 温度スケーリングされたソフトマックス
          soft_targets_temp = F.softmax(soft_batch_clipped / T, dim=1)
          soft_prob = F.log_softmax(student_logits_clipped / T, dim=1)

          # 数値安定性のためのクリッピング
          eps = 1e-8
          soft_targets_temp = torch.clamp(soft_targets_temp, min=eps, max=1.0 - eps)

          # KL損失とCE損失の計算
          distillation_loss = kl_loss(soft_prob, soft_targets_temp) * (T**2)
          student_loss = ce_loss(student_logits, labels)

          # NaN/Inf の確認と処理
          if torch.isnan(distillation_loss) or torch.isinf(distillation_loss):
            print("Warning: Invalid distillation loss detected, using only CE loss")
            loss = student_loss
          elif torch.isnan(student_loss) or torch.isinf(student_loss):
            print("Warning: Invalid student loss detected, using only distillation loss")
            loss = distillation_loss
          else:
            loss = soft_target_loss_weight * distillation_loss + ce_loss_weight * student_loss

          # 最終的な損失の検証
          if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: Final loss is invalid, skipping this batch")
            continue

          # 損失を逆伝搬
          loss.backward()

          # 勾配クリッピング
          clip_grad_norm_(self.studentModel.parameters(), max_norm=1.0)

          optimizer.step()
          running_loss += loss.item()
          batch_count += 1

        if batch_count > 0:
          epoch_loss = running_loss / batch_count
          if use_scheduler and scheduler is not None:
            scheduler.step(epoch_loss)
          print(f"Knowledge Distillation Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}, Processed batches: {batch_count}")
        else:
          print(f"Knowledge Distillation Epoch {epoch + 1}/{epochs}: No valid batches processed")

    return self.studentModel
