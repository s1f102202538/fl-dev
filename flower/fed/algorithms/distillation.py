from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from ..models.base_model import BaseModel


class Distillation:
  def __init__(self, studentModel: BaseModel, public_data: DataLoader, soft_targets: List[torch.Tensor]) -> None:
    self.studentModel = studentModel
    self.public_data = public_data  # 公開データセット
    self.soft_targets = soft_targets  # ソフトターゲット

    # ソフトターゲットがリストかどうかを判定
    self.is_batch_list = isinstance(soft_targets, list) and len(soft_targets) > 0 and isinstance(soft_targets[0], torch.Tensor)
    # バッチ数の検証
    if self.is_batch_list:
      self._validate_batch_counts()

  def _validate_batch_counts(self) -> None:
    """バッチ数の検証を行う"""
    expected_batches = len(self.public_data)
    actual_batches = len(self.soft_targets)
    if expected_batches != actual_batches:
      print(f"Warning: Batch count mismatch. Public data: {expected_batches}, Soft targets: {actual_batches}")
      # より少ない方に合わせる
      min_batches = min(expected_batches, actual_batches)
      print(f"Using first {min_batches} batches for distillation")

  def _check_early_stopping(self, current_loss: float, best_loss: float, patience_counter: int, patience: int) -> tuple[bool, float, int]:
    """Early stoppingのチェックを行う

    Args:
        current_loss: 現在のエポックの損失
        best_loss: これまでの最良損失
        patience_counter: 改善が見られないエポック数のカウンター
        patience: 早期終了までの許容エポック数

    Returns:
        tuple: (early_stopフラグ, 更新された最良損失, 更新されたカウンター)
    """
    # 損失が改善した場合
    if current_loss < best_loss:
      best_loss = current_loss
      patience_counter = 0
      return False, best_loss, patience_counter

    # 損失が改善しなかった場合
    patience_counter += 1
    if patience_counter >= patience:
      print(f"Early stopping triggered: No improvement for {patience} epochs")
      return True, best_loss, patience_counter

    return False, best_loss, patience_counter

  def train_knowledge_distillation(self, epochs: int, learning_rate: float, T: float, alpha: float, beta: float, device: torch.device) -> BaseModel:
    """知識蒸留による訓練を実行

    Args:
        epochs: 訓練エポック数
        learning_rate: 学習率
        T: 蒸留温度
        alpha: KL蒸留損失の重み
        beta: CE損失の重み
        device: 計算デバイス

    Returns:
        訓練済みの生徒モデル
    """
    # 損失関数と最適化アルゴリズム
    ce_loss = nn.CrossEntropyLoss()
    optimizer = Adam(self.studentModel.parameters(), lr=learning_rate)

    # スケジューラーを常に使用
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    self.studentModel.train()  # 生徒モデルを学習モードに設定

    # Early stopping用の変数（固定patience=5）
    best_loss = float("inf")
    patience_counter = 0
    early_stopping_patience = 5  # 固定値: 5エポック改善なしで停止

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

          inputs = batch_data["image"].to(device)
          labels = batch_data["label"].to(device)

          soft_batch = self.soft_targets[batch_idx]

          optimizer.zero_grad()

          student_logits = self.studentModel.predict(inputs)

          # 温度スケーリングされたソフトマックス
          teacher_probs = F.softmax(soft_batch / T, dim=1)  # 教師の確率分布
          student_log_probs = F.log_softmax(student_logits / T, dim=1)  # 生徒のlog確率分布

          # 数値安定性のためのクリッピング
          eps = 1e-8
          teacher_probs = torch.clamp(teacher_probs, min=eps, max=1.0 - eps)

          # KL損失: KL(teacher || student) - 教師から生徒への知識転移
          distillation_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T**2)

          # 生徒モデルの通常のCE損失
          student_loss = ce_loss(student_logits, labels)

          loss = alpha * distillation_loss + beta * student_loss

          # 損失を逆伝搬
          loss.backward()

          # 勾配クリッピング
          clip_grad_norm_(self.studentModel.parameters(), max_norm=1.0)

          optimizer.step()
          running_loss += loss.item()
          batch_count += 1

        if batch_count > 0:
          epoch_loss = running_loss / batch_count
          scheduler.step(epoch_loss)
          print(f"FedKD Distillation Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}, Processed batches: {batch_count}")

          # Early stoppingのチェック
          should_stop, best_loss, patience_counter = self._check_early_stopping(epoch_loss, best_loss, patience_counter, early_stopping_patience)
          if should_stop:
            print(f"Early stopping at epoch {epoch + 1}/{epochs}")
            break
        else:
          print(f"FedKD Distillation Epoch {epoch + 1}/{epochs}: No valid batches processed")

    return self.studentModel
