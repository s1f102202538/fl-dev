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
  def inference_with_loca(net: BaseModel, data_loader: DataLoader, device: torch.device, alpha: float = 1.5) -> list[torch.Tensor]:
    """
    Generate logits with LoCa calibration.

    ```
    Args:
        net: Model to generate logits from
        data_loader: DataLoader containing labeled data
        device: Device to run inference on
        alpha: Scaling factor for target logit (LoCa)

    Returns:
        List of calibrated logit tensors
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

        # 2. LoCa 校正
        # target logit を alpha 倍し、非 target の logits を再調整
        target_logits = outputs[torch.arange(len(labels)), labels]
        target_logits_calibrated = target_logits * alpha
        outputs_calibrated = outputs.clone()
        outputs_calibrated[torch.arange(len(labels)), labels] = target_logits_calibrated

        # optional: 非 target logits の再スケーリング
        non_target_mask = torch.ones_like(outputs, dtype=torch.bool)
        non_target_mask[torch.arange(len(labels)), labels] = False
        outputs_calibrated[non_target_mask] = outputs_calibrated[non_target_mask] / alpha**0.5

        logits.append(outputs_calibrated.cpu())

    return logits
