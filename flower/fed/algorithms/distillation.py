from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.utils.data import DataLoader

from ..models.base_model import BaseModel


class Distillation:
  def __init__(self, studentModel: BaseModel, public_data: DataLoader, soft_targets: List[torch.Tensor]) -> None:
    self.studentModel = studentModel
    self.public_data = public_data  # 公開データセット
    self.soft_targets = soft_targets  # ソフトターゲット

  def _validate_batch_compatibility(self) -> None:
    """バッチサイズの互換性を事前チェック"""
    print(f"[Distillation] Starting batch compatibility check with {len(self.soft_targets)} soft target batches")

    for i, batch_data in enumerate(self.public_data):
      if i < len(self.soft_targets):
        data_batch_size = batch_data["image"].size(0)
        soft_batch_size = self.soft_targets[i].size(0)
        if data_batch_size != soft_batch_size:
          print(f"[Distillation] Batch {i}: Data={data_batch_size}, Soft={soft_batch_size} (will adjust)")
      if i >= 2:  # 最初の数バッチのみチェック
        break

    print("[Distillation] Batch compatibility check completed")

  def _adjust_batch_tensors(
    self, inputs: torch.Tensor, labels: torch.Tensor, soft_batch: torch.Tensor, batch_idx: int
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """バッチテンソルのサイズを調整

    Args:
        inputs: 入力画像テンソル
        labels: ラベルテンソル
        soft_batch: ソフトターゲットテンソル
        batch_idx: バッチインデックス

    Returns:
        調整されたテンソルのタプル、またはスキップすべき場合はNone
    """
    current_batch_size = inputs.size(0)
    soft_batch_size = soft_batch.size(0)

    if current_batch_size != soft_batch_size:
      # サイズが異なる場合の調整
      min_batch_size = min(current_batch_size, soft_batch_size)
      if min_batch_size < 1:
        print(f"[Distillation] Warning: Empty batch {batch_idx}, skipping")
        return None

      inputs = inputs[:min_batch_size]
      labels = labels[:min_batch_size]
      soft_batch = soft_batch[:min_batch_size]
      print(f"[Distillation] Batch {batch_idx}: Adjusted from {current_batch_size} to {min_batch_size}")

    return inputs, labels, soft_batch

  def _validate_logits_compatibility(self, student_logits: torch.Tensor, soft_batch: torch.Tensor, batch_idx: int) -> bool:
    """ロジットの形状互換性をチェック

    Args:
        student_logits: 学生モデルのロジット
        soft_batch: ソフトターゲット
        batch_idx: バッチインデックス

    Returns:
        互換性があればTrue、なければFalse
    """
    if student_logits.size(0) != soft_batch.size(0):
      print(f"[Distillation] Error: Logits size mismatch in batch {batch_idx} - student: {student_logits.size(0)}, soft: {soft_batch.size(0)}")
      return False

    if student_logits.size(1) != soft_batch.size(1):
      print(f"[Distillation] Error: Feature size mismatch in batch {batch_idx} - student: {student_logits.size(1)}, soft: {soft_batch.size(1)}")
      return False

    return True

  def train_knowledge_distillation(self, epochs: int, learning_rate: float, T: float, alpha: float, beta: float, device: torch.device) -> BaseModel:
    # 損失関数と最適化アルゴリズム
    ce_loss = nn.CrossEntropyLoss()
    optimizer = SGD(self.studentModel.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    self.studentModel.train()  # 生徒モデルを学習モードに設定

    # ソフトターゲットをデバイスに移動
    self.soft_targets = [batch.to(device) for batch in self.soft_targets]

    # バッチサイズ互換性の事前チェック
    self._validate_batch_compatibility()

    for epoch in range(epochs):
      running_loss = 0.0
      batch_count = 0

      # データローダーとソフトターゲットバッチを同時にイテレート
      for batch_idx, batch_data in enumerate(self.public_data):
        # データの形式を柔軟に処理
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        # ソフトターゲットバッチのインデックスチェック
        if batch_idx >= len(self.soft_targets):
          print(f"[Distillation] Warning: No soft target for batch {batch_idx}, skipping")
          continue

        soft_batch = self.soft_targets[batch_idx]

        # バッチテンソルのサイズ調整
        adjusted_tensors = self._adjust_batch_tensors(inputs, labels, soft_batch, batch_idx)
        if adjusted_tensors is None:
          continue

        inputs, labels, soft_batch = adjusted_tensors

        optimizer.zero_grad()

        try:
          student_logits = self.studentModel.predict(inputs)

          # ロジットの互換性チェック
          if not self._validate_logits_compatibility(student_logits, soft_batch, batch_idx):
            continue

          # 温度スケーリングされたソフトマックス
          teacher_probs = F.softmax(soft_batch / T, dim=1)  # 教師の確率分布
          student_log_probs = F.log_softmax(student_logits / T, dim=1)  # 生徒のlog確率分布

          # KL損失: KL(teacher || student) - 教師から生徒への知識転移
          distillation_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T**2)

        except RuntimeError as e:
          print(f"[Distillation] Runtime error in batch {batch_idx}: {e}")
          continue

        # 生徒モデルの通常のCE損失
        student_loss = ce_loss(student_logits, labels)

        loss = alpha * distillation_loss + beta * student_loss

        # 損失を逆伝搬
        loss.backward()

        # 勾配クリッピング
        clip_grad_norm_(self.studentModel.parameters(), max_norm=1.0)

        optimizer.step()

        # 損失の統計を更新
        running_loss += loss.item()
        batch_count += 1

      # エポックごとの損失ログ出力
      avg_loss = running_loss / batch_count if batch_count > 0 else 0.0

      print(f"[Distillation] Epoch {epoch + 1}/{epochs}: Total Loss: {avg_loss:.6f}, ")

    return self.studentModel
