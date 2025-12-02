from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from ..models.base_model import BaseModel


class Distillation:
  def __init__(
    self,
    studentModel: BaseModel,
    public_data: DataLoader,
    soft_targets: List[torch.Tensor],
    *,
    shuffle_public_data: bool = False,
  ) -> None:
    """
    Args:
        studentModel: 生徒モデル（BaseModel）
        public_data: 公開データの DataLoader（**shuffle=False を推奨**）
        soft_targets: サーバから渡されたソフトターゲット（通常は logits のリスト）
        shuffle_public_data: DataLoader が shuffle=True の場合、対応が崩れるので False を推奨
    """
    self.studentModel = studentModel
    self.public_data = public_data
    self.soft_targets = soft_targets

    self.is_batch_list = isinstance(soft_targets, list) and len(soft_targets) > 0 and isinstance(soft_targets[0], torch.Tensor)
    if self.is_batch_list:
      self._validate_batch_counts()

    if shuffle_public_data:
      print("Warning: public_data was created with shuffle=True. This often breaks correspondence with soft_targets.")

  def _validate_batch_counts(self) -> None:
    expected_batches = len(self.public_data)
    actual_batches = len(self.soft_targets)
    if expected_batches != actual_batches:
      print(f"Warning: Batch count mismatch. Public data batches: {expected_batches}, Soft targets batches: {actual_batches}")
      min_batches = min(expected_batches, actual_batches)
      print(f"Using first {min_batches} batches for distillation")

  def _check_early_stopping(self, current_loss: float, best_loss: float, patience_counter: int, patience: int) -> Tuple[bool, float, int]:
    if current_loss < best_loss:
      best_loss = current_loss
      patience_counter = 0
      return False, best_loss, patience_counter

    patience_counter += 1
    if patience_counter >= patience:
      print(f"Early stopping triggered: No improvement for {patience} epochs")
      return True, best_loss, patience_counter

    return False, best_loss, patience_counter

  def train_knowledge_distillation(
    self,
    epochs: int,
    learning_rate: float,
    T: float,
    alpha: float,
    beta: float,
    device: torch.device,
    *,
    early_stopping_patience: int = 5,
    grad_clip: float = 1.0,
    debug_print_first_batch: bool = True,
  ) -> BaseModel:
    """
    実際の蒸留学習ループ（改良版）

    Args:
        epochs: epoch 数
        learning_rate: 学習率
        T: 温度
        alpha: KL 蒸留損失の重み
        beta: CE 損失の重み
        device: cpu / cuda device
        early_stopping_patience: 早期終了の patience
        grad_clip: 勾配クリッピングの max_norm
        debug_print_first_batch: 先頭バッチ時にデバッグ統計を出力するか
    """

    ce_loss = nn.CrossEntropyLoss()
    optimizer = Adam(self.studentModel.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    self.studentModel.to(device)
    self.studentModel.train()

    best_loss = float("inf")
    patience_counter = 0

    if not self.is_batch_list:
      raise ValueError("soft_targets must be a non-empty list of torch.Tensor (one tensor per public-data batch).")

    # 事前に soft_targets を device に移動（遅延転送ではなく一括転送）
    # soft_targets の各要素はバッチ単位のテンソル (batch_size, num_classes) を想定
    self.soft_targets = [st.to(device) for st in self.soft_targets]

    min_batches = min(len(self.public_data), len(self.soft_targets))

    for epoch in range(epochs):
      running_loss = 0.0
      batch_count = 0

      for batch_idx, batch_data in enumerate(self.public_data):
        if batch_idx >= min_batches:
          break

        # --- prepare inputs/labels ---
        inputs = batch_data.get("image", None)
        labels = batch_data.get("label", None)
        if inputs is None or labels is None:
          raise KeyError("public_data batches must contain keys 'image' and 'label'")

        inputs = inputs.to(device)
        labels = labels.to(device)

        # teacher batch (from server)
        soft_batch = self.soft_targets[batch_idx]  # already moved to device above

        # --- ensure shapes match ---
        # teacher may be logits or probabilities; detect it
        if soft_batch.dim() != 2:
          raise ValueError(f"soft_targets[{batch_idx}] must be 2D (batch_size, num_classes), got shape {soft_batch.shape}")

        # optionally align batch sizes: if different, use min samples
        if soft_batch.size(0) != inputs.size(0):
          # 可能なら警告してから min 部分のみ使用
          min_bs = min(soft_batch.size(0), inputs.size(0))
          print(f"Warning: batch size mismatch at batch {batch_idx}: inputs {inputs.size(0)}, soft_targets {soft_batch.size(0)}. Using first {min_bs} samples.")
          inputs = inputs[:min_bs]
          labels = labels[:min_bs]
          soft_batch = soft_batch[:min_bs]

        optimizer.zero_grad()

        student_logits = self.studentModel.predict(inputs)

        if not isinstance(student_logits, torch.Tensor):
          raise TypeError("studentModel.predict / forward must return a torch.Tensor of logits")

        with torch.no_grad():
          teacher_logits = soft_batch.detach()
          teacher_probs = F.softmax(teacher_logits / T, dim=1)

          # --- make sure teacher_probs is detached and stable ---
          # numerical clamp optional (avoid exact 0)
          teacher_probs = torch.clamp(teacher_probs, min=1e-8, max=1.0 - 1e-8)

          # teacher の log 確率も計算（KL divergence用）
          teacher_log_probs = torch.log(teacher_probs)

        # --- student log probs with temperature ---
        student_log_probs = F.log_softmax(student_logits / T, dim=1)

        # --- distillation (KL) loss: KL(teacher || student) ---
        # log_target=True により KL(P_teacher || P_student) を計算
        distillation_loss = F.kl_div(student_log_probs, teacher_log_probs, reduction="batchmean", log_target=True) * (T * T)

        # --- normal CE loss on raw logits ---
        student_ce_loss = ce_loss(student_logits, labels)

        loss = alpha * distillation_loss + beta * student_ce_loss

        # backward / step
        loss.backward()
        clip_grad_norm_(self.studentModel.parameters(), max_norm=grad_clip)
        optimizer.step()

        running_loss += loss.item()
        batch_count += 1

        # debug print for first batch only (helps verifying distributions quickly)
        if debug_print_first_batch and epoch == 0 and batch_idx == 0:
          try:
            print("=== Distillation debug (epoch 1, batch 1) ===")
            print(f"teacher_probs sum (first 5): {teacher_probs.sum(dim=1)[:5].cpu().numpy()}")
            print(
              f"teacher_probs stats: mean={teacher_probs.mean().item():.6f}, std={teacher_probs.std().item():.6f}, min={teacher_probs.min().item():.6f}, max={teacher_probs.max().item():.6f}"
            )
            print(
              f"student_logits stats: mean={student_logits.mean().item():.6f}, std={student_logits.std().item():.6f}, min={student_logits.min().item():.6f}, max={student_logits.max().item():.6f}"
            )
            print("=============================================")
          except Exception:
            # 保険: データが GPU 上、numpy に変換できない等のケースで落ちないようにする
            pass
          # only print once
          debug_print_first_batch = False

      # end batches

      if batch_count > 0:
        epoch_loss = running_loss / batch_count
        scheduler.step(epoch_loss)
        print(f"FedKD Distillation Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}, Processed batches: {batch_count}")

        should_stop, best_loss, patience_counter = self._check_early_stopping(epoch_loss, best_loss, patience_counter, early_stopping_patience)
        if should_stop:
          print(f"Early stopping at epoch {epoch + 1}/{epochs}")
          break
      else:
        print(f"FedKD Distillation Epoch {epoch + 1}/{epochs}: No valid batches processed")

    return self.studentModel
