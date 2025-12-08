from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from ..models.base_model import BaseModel


class LoCaBasedDistillation:
  def __init__(
    self,
    studentModel: BaseModel,
    public_data: DataLoader,
    soft_targets: List[torch.Tensor],
  ) -> None:
    self.studentModel = studentModel
    self.public_data = public_data
    self.soft_targets = soft_targets

    self.is_batch_list = isinstance(soft_targets, list) and len(soft_targets) > 0 and isinstance(soft_targets[0], torch.Tensor)
    if self.is_batch_list:
      self._validate_batch_counts()

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

  @staticmethod
  def _loca_calibrate_teacher_logits(
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    tau: float = 0.9,
  ) -> torch.Tensor:
    if teacher_logits.dim() != 2:
      raise ValueError(f"teacher_logits must be 2D, got {teacher_logits.shape}")

    B, C = teacher_logits.shape
    device = teacher_logits.device

    # ステップ 1: ロジットから確率分布を計算（s の計算のため）
    with torch.no_grad():
      probs = F.softmax(teacher_logits, dim=1)

      # ステップ 2: 非正解クラスの最大確率を求める
      non_gt_mask = torch.ones(B, C, dtype=torch.bool, device=device)
      non_gt_mask[torch.arange(B, device=device), labels] = False
      p_biggest = probs.masked_fill(~non_gt_mask, -1).max(dim=1).values  # shape: (B,)

      # ステップ 3: 正解クラスの確率を取得
      p_gt = probs[torch.arange(B, device=device), labels]  # shape: (B,)

      # ステップ 4: LoCa のスケーリング係数 s を計算
      # s_max = 1.0 / (1 - p_y + p_biggest)
      # s = τ * s_max
      # 注意: s_max が 1 を超える可能性があるため、s を 1.0 以下にクリップ
      # （s > 1.0 だと非正解クラスを増幅してしまい、抑制の意図と逆になる）
      s_max = 1.0 / (1 - p_gt + p_biggest + 1e-8)  # 数値安定性
      s = torch.clamp(tau * s_max, max=1.0)  # shape: (B,), s <= 1.0 を保証

    # ステップ 5: ロジット空間で非正解クラスをスケーリング
    logits_loca = teacher_logits.clone()
    s_expanded = s.unsqueeze(1).expand(B, C)  # shape: (B, C)

    # 非正解クラスのロジットのみを s 倍（ロジット空間での縮小）
    logits_loca = torch.where(non_gt_mask, logits_loca * s_expanded, logits_loca)

    return logits_loca

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
    loca_tau: float = 0.92,
  ) -> BaseModel:
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
        teacher_logits = self.soft_targets[batch_idx]  # already moved to device above

        # --- ensure shapes match ---
        if teacher_logits.dim() != 2:
          raise ValueError(f"soft_targets[{batch_idx}] must be 2D (batch_size, num_classes), got shape {teacher_logits.shape}")

        # optionally align batch sizes: if different, use min samples
        if teacher_logits.size(0) != inputs.size(0):
          min_bs = min(teacher_logits.size(0), inputs.size(0))
          print(
            f"Warning: batch size mismatch at batch {batch_idx}: inputs {inputs.size(0)}, soft_targets {teacher_logits.size(0)}. Using first {min_bs} samples."
          )
          inputs = inputs[:min_bs]
          labels = labels[:min_bs]
          teacher_logits = teacher_logits[:min_bs]

        optimizer.zero_grad()

        student_logits = self.studentModel.predict(inputs)
        if not isinstance(student_logits, torch.Tensor):
          raise TypeError("studentModel.predict / forward must return a torch.Tensor of logits")

        # --- LoCa 校正 + 温度スケーリング ---
        # teacher_logits: クライアントから送信された補正なしの集約ロジット
        # labels: サーバの公開データから取得したラベル
        # → ロジット空間で LoCa 補正 → 温度スケーリング → softmax
        with torch.no_grad():
          # 1. LoCa 補正（ロジット空間で非正解クラスをスケーリング）
          teacher_logits_loca = self._loca_calibrate_teacher_logits(teacher_logits.detach(), labels, loca_tau)

          # 2. 温度スケーリング + softmax で確率分布に変換
          teacher_probs = F.softmax(teacher_logits_loca / T, dim=1)

          # numerical clamp
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
            pass
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
