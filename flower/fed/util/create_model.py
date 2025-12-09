from ..models.base_model import BaseModel
from ..models.mini_cnn import MiniCNN, MiniCNNMNIST
from ..models.moon_model import ModelFedCon, ModelFedCon_noheader
from ..models.simple_cnn import SimpleCNN, SimpleCNNMNIST

# モデルマッピング
_MODEL_MAP = {
  "mini-cnn": MiniCNN,
  "mini-cnn-mnist": MiniCNNMNIST,
  "simple-cnn": SimpleCNN,
  "simple-cnn-mnist": SimpleCNNMNIST,
}


def create_model(model_name: str, is_moon: bool = False, out_dim: int = 256, n_classes: int = 10, use_projection_head: bool = True) -> BaseModel:
  if is_moon:
    if use_projection_head:
      return ModelFedCon(base_model=model_name, out_dim=out_dim, n_classes=n_classes)
    else:
      return ModelFedCon_noheader(base_model=model_name, n_classes=n_classes)

  # 通常のモデル作成
  if model_name not in _MODEL_MAP:
    raise ValueError(f"Unknown model name: {model_name}. Available: {list(_MODEL_MAP.keys())}")

  return _MODEL_MAP[model_name](n_classes)
