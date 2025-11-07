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
  logit_batches: List[Tensor], temperature: float = 1.5, enable_quality_filter: bool = True, confidence_threshold: float = 0.2
) -> List[Tensor]:
  """Non-IID環境対応の強化されたロジットフィルタリングと較正処理

  Args:
      logit_batches: フィルタリング対象のロジットバッチリスト
      temperature: 温度スケーリングのパラメータ（サーバーから受信）
      enable_quality_filter: 品質フィルタリングの有効/無効
      confidence_threshold: 信頼度の最低閾値

  Returns:
      フィルタリング・較正されたロジットバッチリスト
  """
  filtered_logits = []
  filtered_count = 0

  for batch_idx, batch in enumerate(logit_batches):
    # NaN/Inf値の検出と修正
    if torch.isnan(batch).any() or torch.isinf(batch).any():
      print(f"警告: バッチ{batch_idx}でNaN/Infを検出、ゼロで置換")
      batch = torch.zeros_like(batch)

    # 数値安定性のための基本クリッピング
    calibrated_batch = torch.clamp(batch, min=-20, max=20)

    # 品質フィルタリング（有効な場合のみ）
    if enable_quality_filter:
      quality_passed, reason = _assess_batch_quality(calibrated_batch, confidence_threshold)
      if not quality_passed:
        print(f"[Client] バッチ{batch_idx}を品質フィルタで除外: {reason}")
        filtered_count += 1
        continue

    # 温度スケーリング（サーバーから受信した温度を使用）
    if temperature != 1.0:
      calibrated_batch = calibrated_batch / temperature

    # 最終的な極値クリッピング
    calibrated_batch = torch.clamp(calibrated_batch, min=-10, max=10)

    filtered_logits.append(calibrated_batch)

  if filtered_count > 0:
    retention_rate = len(filtered_logits) / len(logit_batches) * 100
    print(f"[Client] 品質フィルタリング完了: {filtered_count}/{len(logit_batches)} バッチを除外 (保持率: {retention_rate:.1f}%)")

  return filtered_logits


def _assess_batch_quality(logits: Tensor, confidence_threshold: float) -> Tuple[bool, str]:
  """ロジットバッチの品質を評価

  Args:
      logits: 評価対象のロジットテンソル
      confidence_threshold: 信頼度の最低閾値

  Returns:
      (quality_passed, reason) のタプル
  """
  with torch.no_grad():
    # ソフトマックス確率を計算
    probs = torch.softmax(logits, dim=1)

    # 1. 異常値チェック
    if torch.any(torch.abs(logits) > 50):
      return False, "extreme_values"

    # 2. 分散チェック（過度に均一でないか）
    variance = logits.var(dim=1).mean().item()
    if variance < 0.01:  # 分散が小さすぎる
      return False, f"low_variance({variance:.4f})"

    # 3. 信頼度チェック
    max_probs = probs.max(dim=1)[0]
    avg_confidence = max_probs.mean().item()
    if avg_confidence < confidence_threshold:
      return False, f"low_confidence({avg_confidence:.3f})"

    # 4. エントロピーチェック（情報量）
    eps = 1e-8
    probs_safe = torch.clamp(probs, min=eps, max=1.0 - eps)
    entropy = -torch.sum(probs_safe * torch.log(probs_safe), dim=1).mean().item()
    if entropy < 0.1:  # エントロピーが低すぎる（過信状態）
      return False, f"low_entropy({entropy:.3f})"

    return True, "high_quality"


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
