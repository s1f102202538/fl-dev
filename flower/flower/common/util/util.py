from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn


def get_weights(net: nn.Module) -> List[np.ndarray]:
  return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters: List[np.ndarray]) -> None:
  params_dict = zip(net.state_dict().keys(), parameters)
  state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
  net.load_state_dict(state_dict, strict=True)
