import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MiniCNN(nn.Module):
  """軽量なMiniCNNモデル - オリジナルのシンプルなアーキテクチャ"""

  def __init__(self, dropout_rate: float = 0.2) -> None:
    super().__init__()

    # シンプルな3層構造
    self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.dropout = nn.Dropout2d(dropout_rate)

    # 全結合層
    self.fc1 = nn.Linear(32 * 7 * 7, 128)
    self.fc2 = nn.Linear(128, 10)
    self.dropout_fc = nn.Dropout(dropout_rate)

  def forward(self, x: Tensor) -> Tensor:
    # 畳み込み層
    x = F.relu(self.conv1(x))
    x = self.pool(x)  # 28x28 -> 14x14

    x = F.relu(self.conv2(x))
    x = self.pool(x)  # 14x14 -> 7x7
    x = self.dropout(x)

    # 全結合層
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = self.dropout_fc(x)
    x = self.fc2(x)

    return x
