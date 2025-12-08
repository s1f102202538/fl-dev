import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from ..models.base_model import BaseModel


class CNNTask:
  @staticmethod
  def train(net: BaseModel, train_loader: DataLoader, epochs: int, lr: float, device: torch.device) -> float:
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
      for batch in train_loader:
        images = batch["image"]
        labels = batch["label"]
        optimizer.zero_grad()

        outputs = net.predict(images.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / (len(train_loader) * epochs)
    return avg_train_loss

  @staticmethod
  def test(net: BaseModel, test_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total_loss, total_samples = 0, 0.0, 0

    with torch.no_grad():
      for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        outputs = net.predict(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = (correct / total_samples) * 100.0
    return avg_loss, accuracy

  @staticmethod
  def inference(net: BaseModel, data_loader: DataLoader, device: torch.device) -> list[torch.Tensor]:
    """Generate logits without any correction.

    Args:
        net: Model to generate logits from
        data_loader: DataLoader containing data
        device: Device to run inference on
    """
    net.to(device)
    net.eval()
    logits = []
    with torch.no_grad():
      for batch in data_loader:
        images = batch["image"].to(device)
        outputs = net.predict(images)
        logits.append(outputs.cpu())

    return logits

  @staticmethod
  def inference_with_label_correction(net: BaseModel, data_loader: DataLoader, device: torch.device, correction_strength: float = 1.0) -> list[torch.Tensor]:
    """Generate logits with label-based correction using top1-top2 margin.

    Args:
        net: Model to generate logits from
        data_loader: DataLoader containing labeled IID public data
        device: Device to run inference on
        correction_strength: Strength multiplier for correction (default: 1.0)

    Returns:
        List of corrected logit tensors
    """
    net.to(device)
    net.eval()
    logits = []
    with torch.no_grad():
      for batch in data_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        outputs = net.predict(images)

        # Calculate top1-top2 margin correction
        # top1 = maximum logit, top2 = second maximum logit (excluding correct class)
        top2 = outputs.clone()
        top2.scatter_(1, labels.unsqueeze(1), float("-inf"))  # Exclude correct class
        top2_vals, _ = top2.max(dim=1)

        # Margin = top1 - top2
        margin = outputs.max(dim=1).values - top2_vals

        # Add correction to correct class based on margin
        outputs[torch.arange(len(labels)), labels] += correction_strength * margin

        logits.append(outputs.cpu())

    return logits

  @staticmethod
  def inference_with_loca(net: BaseModel, data_loader: DataLoader, device: torch.device, tau: float = 0.9) -> list[torch.Tensor]:
    """
    Generate logits with LoCa (Logit Calibration) correction.

    論文準拠の実装（ロジット空間で補正）:
      1. ロジットから確率分布を計算して縮小係数 s を求める
      2. 非正解クラスのロジットを s でスケーリング（ロジット空間）
      3. 補正後のロジットを返す

    LoCa 論文の定義:
      s_max = 1 / (1 - p_y + p_biggest)
      s = τ * s_max
      ここで τ は "shrink factor" (0 < τ ≤ 1)
      τ が小さいほど非正解クラスを強く抑制

    Args:
        net: Model to generate logits from
        data_loader: DataLoader containing labeled data
        device: Device to run inference on
        tau: LoCa の縮小係数 τ（0 < τ ≤ 1）
             τ=1.0 で最小限の補正、τ→0 で強い抑制

    Returns:
        List of LoCa-calibrated logit tensors
    """
    net.to(device)
    net.eval()
    logits = []

    with torch.no_grad():
      for batch in data_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # 1. 通常の推論
        outputs = net.predict(images)
        B, C = outputs.shape

        # 2. ロジットから確率分布を計算（s の計算のため）
        probs = torch.nn.functional.softmax(outputs, dim=1)

        # 3. 非正解クラスの最大確率を求める
        non_gt_mask = torch.ones(B, C, dtype=torch.bool, device=device)
        non_gt_mask[torch.arange(B, device=device), labels] = False
        p_biggest = probs.masked_fill(~non_gt_mask, -1).max(dim=1).values  # shape: (B,)

        # 4. 正解クラスの確率を取得
        p_gt = probs[torch.arange(B, device=device), labels]  # shape: (B,)

        # 5. LoCa のスケーリング係数 s を計算
        # s_max = 1.0 / (1 - p_y + p_biggest)
        # s = τ * s_max
        # 注意: s を 1.0 以下にクリップ（s > 1.0 だと非正解クラスを増幅してしまう）
        s_max = 1.0 / (1 - p_gt + p_biggest + 1e-8)  # 数値安定性
        s = torch.clamp(tau * s_max, max=1.0)  # shape: (B,), s <= 1.0 を保証

        # 6. ロジット空間で非正解クラスをスケーリング
        outputs_calibrated = outputs.clone()
        s_expanded = s.unsqueeze(1).expand(B, C)  # shape: (B, C)

        # 非正解クラスのロジットのみを s 倍（ロジット空間での縮小）
        outputs_calibrated = torch.where(non_gt_mask, outputs_calibrated * s_expanded, outputs_calibrated)

        logits.append(outputs_calibrated.cpu())

    return logits

  def inference_with_loca_extended(
    net: BaseModel,
    data_loader: DataLoader,
    device: torch.device,
    tau: float = 0.99,
    alpha: float = 1.05,
  ) -> list[torch.Tensor]:
    net.to(device)
    net.eval()
    logits = []

    with torch.no_grad():
      for batch in data_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = net.predict(images)  # shape (B, C)
        B, C = outputs.shape

        # Softmax → s の計算のため
        probs = torch.nn.functional.softmax(outputs, dim=1)
        non_gt_mask = torch.ones(B, C, dtype=torch.bool, device=device)
        non_gt_mask[torch.arange(B, device=device), labels] = False

        # LoCa 用 s 計算
        p_biggest = probs.masked_fill(~non_gt_mask, -1).max(dim=1).values
        p_gt = probs[torch.arange(B, device=device), labels]
        s_max = 1.0 / (1 - p_gt + p_biggest + 1e-8)
        s = torch.clamp(tau * s_max, max=1.0)

        outputs_calibrated = outputs.clone()
        s_expanded = s.unsqueeze(1).expand(B, C)

        # LoCa の補正
        outputs_calibrated = torch.where(non_gt_mask, outputs_calibrated * s_expanded, outputs_calibrated)

        # 正解クラスのロジットの補正
        if alpha > 0:
          # LoCa補正後の最大非正解ロジット
          max_non_gt = outputs_calibrated.masked_fill(~non_gt_mask, -1e10).max(dim=1).values

          # 正解クラスのロジット
          z_y = outputs_calibrated[torch.arange(B), labels]

          # マージン = 正解と最大非正解の差
          margin = z_y - max_non_gt

          # マージンが負または小さい場合のみ補正
          # margin が負の場合は絶対値で補正量を決定
          correction = torch.where(margin < 0, alpha * torch.abs(margin), torch.zeros_like(margin))

          outputs_calibrated[torch.arange(B), labels] = z_y + correction

        logits.append(outputs_calibrated.cpu())

    return logits
