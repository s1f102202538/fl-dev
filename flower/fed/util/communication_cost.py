import pickle
import sys
from typing import Dict

from flwr.common.typing import Parameters


def calculate_communication_cost(parameters: Parameters) -> Dict[str, float]:
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
  size_bytes = len(data.encode("utf-8"))
  size_mb = size_bytes / (1024 * 1024)
  return round(size_mb, 4)
