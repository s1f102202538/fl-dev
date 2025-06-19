import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from flwr.common.typing import NDArrays, UserConfig


def get_weights(net: nn.Module) -> NDArrays:
  return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters: NDArrays) -> None:
  params_dict = zip(net.state_dict().keys(), parameters)
  state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
  net.load_state_dict(state_dict, strict=True)


def create_run_dir(config: UserConfig) -> Tuple[Path, str]:
  """Create a directory where to save results from this run."""
  # Create output directory given current timestamp
  current_time = datetime.now()
  run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
  # Save path is based on the current directory
  save_path = Path.cwd() / f"outputs/{run_dir}"
  save_path.mkdir(parents=True, exist_ok=False)

  # Save run config as json
  with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
    json.dump(config, fp)

  return save_path, run_dir
