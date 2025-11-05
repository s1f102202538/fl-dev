from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from ..models.base_model import BaseModel


class Distillation:
  def __init__(self, studentModel: BaseModel, public_data: DataLoader, soft_targets: List[torch.Tensor]) -> None:
    self.studentModel = studentModel
    self.public_data = public_data  # 公開データセット
    self.soft_targets = soft_targets  # ソフトターゲット

  def train_knowledge_distillation(self, epochs: int, learning_rate: float, T: float, alpha: float, beta: float, device: torch.device) -> BaseModel:
    # 損失関数と最適化アルゴリズム
    ce_loss = nn.CrossEntropyLoss()
    optimizer = Adam(self.studentModel.parameters(), lr=learning_rate)

    self.studentModel.train()  # 生徒モデルを学習モードに設定

    # ソフトターゲットをデバイスに移動
    self.soft_targets = [batch.to(device) for batch in self.soft_targets]

    for epoch in range(epochs):
      running_loss = 0.0
      batch_count = 0

      # データローダーとソフトターゲットバッチを同時にイテレート
      for batch_idx, batch_data in enumerate(self.public_data):
        # データの形式を柔軟に処理
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        soft_batch = self.soft_targets[batch_idx]

        optimizer.zero_grad()

        student_logits = self.studentModel.predict(inputs)

        # 温度スケーリングされたソフトマックス
        teacher_probs = F.softmax(soft_batch / T, dim=1)  # 教師の確率分布
        student_log_probs = F.log_softmax(student_logits / T, dim=1)  # 生徒のlog確率分布

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

        # 損失の統計を更新
        running_loss += loss.item()
        batch_count += 1

      # エポックごとの損失ログ出力
      avg_loss = running_loss / batch_count if batch_count > 0 else 0.0

      print(f"[Distillation] Epoch {epoch + 1}/{epochs}: Total Loss: {avg_loss:.6f}, ")

    return self.studentModel
