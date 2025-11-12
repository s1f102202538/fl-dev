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


def filter_and_calibrate_logits(
  logit_batches: List[Tensor], temperature: float = 1.5, enable_quality_filter: bool = False, confidence_threshold: float = 0.2
) -> List[Tensor]:
  """ロジットの基本的な較正処理（フィルタリング除外設定を削除）

  Args:
      logit_batches: 較正対象のロジットバッチリスト
      temperature: 温度スケーリングのパラメータ（サーバーから受信）
      enable_quality_filter: 品質フィルタリングの有効/無効（デフォルト無効）
      confidence_threshold: 信頼度の閾値（未使用）

  Returns:
      較正されたロジットバッチリスト
  """
  calibrated_logits = []

  for batch_idx, batch in enumerate(logit_batches):
    # NaN/Inf値の検出と修正（基本的な数値安定性のみ）
    if torch.isnan(batch).any() or torch.isinf(batch).any():
      print(f"[Client] 警告: バッチ{batch_idx}でNaN/Infを検出、ゼロで置換")
      batch = torch.zeros_like(batch)

    # 数値安定性のための基本クリッピング
    calibrated_batch = torch.clamp(batch, min=-20, max=20)

    # 温度スケーリング（サーバーから受信した温度を使用）
    if temperature != 1.0:
      calibrated_batch = calibrated_batch / temperature

    # 最終的な極値クリッピング
    calibrated_batch = torch.clamp(calibrated_batch, min=-10, max=10)

    calibrated_logits.append(calibrated_batch)

  print(f"[Client] ロジット較正完了: {len(calibrated_logits)} バッチを処理 (フィルタリング除外なし)")

  return calibrated_logits


# モデル状態管理用関数
def save_model_to_state(model: BaseModel, client_state: RecordDict, model_name: str) -> None:
  """モデルの重みをclient stateに保存

  Args:
      model: 保存するPyTorchモデル
      client_state: Flowerのクライアント状態
      model_name: 状態保存時の識別名
  """
  if model is not None:
    arr_record = ArrayRecord(model.state_dict())  # type: ignore
    # RecordDictに追加（既に存在する場合は置換）
    client_state[model_name] = arr_record


def load_model_from_state(
  client_state: RecordDict,
  reference_model: BaseModel,
  model_name: str,
) -> BaseModel | None:
  """client stateからモデルの重みを読み込み

  Args:
      client_state: Flowerのクライアント状態
      reference_model: モデル構造の参照用ベースモデル
      model_name: 状態保存時の識別名

  Returns:
      復元されたモデル（復元失敗時はNone）
  """
  if model_name not in client_state.array_records:
    return None

  state_dict = client_state[model_name].to_torch_state_dict()  # type: ignore
  # 新しいモデルを作成して読み込み
  new_model = copy.deepcopy(reference_model)
  new_model.load_state_dict(state_dict, strict=True)  # type: ignore
  return new_model
