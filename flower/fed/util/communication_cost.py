"""Communication cost calculation utilities for federated learning."""

import pickle
import sys
from typing import Dict

from flwr.common.typing import Parameters


def calculate_communication_cost(parameters: Parameters) -> Dict[str, float]:
  """通信コストを計算する（バイト単位）

  Args:
    parameters: Flowerのパラメータオブジェクト

  Returns:
    通信コストの情報を含む辞書
    - size_bytes: バイト単位のサイズ
    - size_mb: MB単位のサイズ（小数点4桁まで）
  """
  try:
    # Parametersオブジェクトをシリアライズしてサイズを測定
    serialized = pickle.dumps(parameters)
    size_bytes = len(serialized)
    size_mb = size_bytes / (1024 * 1024)  # MB単位

    return {"size_bytes": float(size_bytes), "size_mb": round(size_mb, 4)}
  except Exception:
    # シリアライズに失敗した場合、sys.getsizeofでおおよそのサイズを測定
    size_bytes = sys.getsizeof(parameters)
    size_mb = size_bytes / (1024 * 1024)

    return {"size_bytes": float(size_bytes), "size_mb": round(size_mb, 4)}


def calculate_data_size_mb(data: str) -> float:
  """文字列データのサイズをMB単位で計算する

  Args:
    data: サイズを測定する文字列データ

  Returns:
    MB単位のサイズ（小数点4桁まで）
  """
  size_bytes = len(data.encode("utf-8"))
  size_mb = size_bytes / (1024 * 1024)
  return round(size_mb, 4)
