import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader


class CNNTask:
  @staticmethod
  def train(net: nn.Module, train_loader: DataLoader, epochs: int, lr: float, device: torch.device) -> float:
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
        loss = criterion(net(images.to(device)), labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    return avg_train_loss

  @staticmethod
  def test(net: nn.Module, test_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    total_samples = 0
    with torch.no_grad():
      for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        outputs = net(images)
        loss += criterion(outputs, labels).item()
        correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = (correct / total_samples) * 100.0
    loss = loss / len(test_loader)
    return loss, accuracy

  @staticmethod
  def inference(net: nn.Module, data_loader: DataLoader, device: torch.device) -> list[torch.Tensor]:
    net.to(device)
    net.eval()
    logits = []
    with torch.no_grad():
      for batch in data_loader:
        images = batch["image"].to(device)
        outputs = net(images)
        logits.append(outputs.cpu())

    return logits

  @staticmethod
  def moon_test(moon_model: nn.Module, test_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """MoonModel用のテストメソッド

    Args:
        moon_model: MoonModel（複数出力: h, proj, y）
        test_loader: テストデータローダー
        device: 計算デバイス

    Returns:
        (loss, accuracy): 損失と精度
    """
    moon_model.to(device)
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    total_samples = 0

    with torch.no_grad():
      for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # MoonModelの出力: (h, proj, y)
        _, _, outputs = moon_model(images)

        loss += criterion(outputs, labels).item()
        correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = (correct / total_samples) * 100.0
    loss = loss / len(test_loader)
    return loss, accuracy

  @staticmethod
  def moon_inference(moon_model: nn.Module, data_loader: DataLoader, device: torch.device) -> list[torch.Tensor]:
    """MoonModel用の推論メソッド

    Args:
        moon_model: MoonModel（複数出力: h, proj, y）
        data_loader: データローダー
        device: 計算デバイス

    Returns:
        分類ロジットのリスト
    """
    moon_model.to(device)
    moon_model.eval()
    logits = []

    with torch.no_grad():
      for batch in data_loader:
        images = batch["image"].to(device)

        # MoonModelの出力: (h, proj, y)
        _, _, outputs = moon_model(images)

        logits.append(outputs.cpu())

    return logits
