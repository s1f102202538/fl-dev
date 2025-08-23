import base64
import io
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from flwr.common import Scalar
from flwr.common.typing import NDArrays, UserConfig
from torch import Tensor, load, save


def get_weights(net: nn.Module) -> NDArrays:
  return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters: NDArrays) -> None:
  params_dict = zip(net.state_dict().keys(), parameters)
  state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
  net.load_state_dict(state_dict, strict=True)


def weighted_average(metrics: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, Scalar]:
  # Multiply accuracy of each client by number of examples used
  accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
  examples = [num_examples for num_examples, _ in metrics]

  # Aggregate and return custom metric (weighted average)
  return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


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


def batch_list_to_base64(batch_list: List[Tensor]) -> str:
  """Convert a list of tensors (batches) to base64 string."""
  buffer = io.BytesIO()
  save(batch_list, buffer)
  return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_batch_list(b64str: str) -> List[Tensor]:
  """Convert base64 string to a list of tensors (batches)."""
  buffer = io.BytesIO(base64.b64decode(b64str))
  return load(buffer)
