import base64
import copy
import io
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from flwr.common import ArrayRecord, RecordDict, Scalar
from flwr.common.typing import NDArrays, UserConfig
from torch import Tensor, load, save

from ..models.base_model import BaseModel


def get_weights(net: BaseModel) -> NDArrays:
  return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: BaseModel, parameters: NDArrays) -> None:
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
  buffer = io.BytesIO()
  save(batch_list, buffer)
  return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_batch_list(b64str: str) -> List[Tensor]:
  buffer = io.BytesIO(base64.b64decode(b64str))
  return load(buffer)


# モデル状態管理用関数
def save_model_to_state(model: BaseModel, client_state: RecordDict, model_name: str) -> None:
  if model is not None:
    arr_record = ArrayRecord(model.state_dict())  # type: ignore
    # RecordDictに追加（既に存在する場合は置換）
    client_state[model_name] = arr_record


def load_model_from_state(
  client_state: RecordDict,
  reference_model: BaseModel,
  model_name: str,
) -> BaseModel | None:
  if model_name not in client_state.array_records:
    return None

  state_dict = client_state[model_name].to_torch_state_dict()  # type: ignore
  # 新しいモデルを作成して読み込み
  new_model = copy.deepcopy(reference_model)
  new_model.load_state_dict(state_dict, strict=True)  # type: ignore
  return new_model
