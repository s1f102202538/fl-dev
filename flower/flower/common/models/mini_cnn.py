import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MiniCNN(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.conv1 = nn.Conv2d(1, 16, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(16, 32, 5)
    self.fc1 = nn.Linear(32 * 4 * 4, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x: Tensor) -> Tensor:
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 32 * 4 * 4)
    x = F.relu(self.fc1(x))
    return self.fc2(x)
