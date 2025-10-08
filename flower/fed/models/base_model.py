from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn
from torch import Tensor


class BaseModel(ABC, nn.Module):
  def __init__(self) -> None:
    super().__init__()

  @abstractmethod
  def forward(self, x: Tensor) -> Any:
    pass

  def predict(self, x: Tensor) -> Tensor:
    return self.forward(x)
