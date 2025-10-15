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
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    total_samples = 0
    with torch.no_grad():
      for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        outputs = net.predict(images)
        loss += criterion(outputs, labels).item()
        correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = (correct / total_samples) * 100.0
    loss = loss / len(test_loader)
    return loss, accuracy

  @staticmethod
  def inference(net: BaseModel, data_loader: DataLoader, device: torch.device) -> list[torch.Tensor]:
    net.to(device)
    net.eval()
    logits = []
    with torch.no_grad():
      for batch in data_loader:
        images = batch["image"].to(device)
        outputs = net.predict(images)
        logits.append(outputs.cpu())

    return logits
