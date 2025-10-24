from ..models.base_model import BaseModel
from ..models.mini_cnn import MiniCNN, MiniCNNMNIST
from ..models.simple_cnn import SimpleCNN, SimpleCNNMNIST
from ..models.moon_model import ModelFedCon, ModelFedCon_noheader


def create_model(
  model_name: str, is_moon: bool = False, out_dim: int = 256, n_classes: int = 10, use_projection_head: bool = True, unified_model: bool = False
) -> BaseModel:
  """統一されたモデル作成関数

  Args:
    model_name: モデル名 ("mini-cnn", "mini-cnn-mnist", "simple-cnn", "simple-cnn-mnist")
    is_moon: MOONモデルを使用するかどうか
    out_dim: 投影ヘッドの出力次元
    n_classes: クラス数
    use_projection_head: 投影ヘッドを使用するかどうか（MOONのみ）
    unified_model: すべての手法で統一されたモデルを使用するかどうか
  """
  if is_moon or unified_model:
    return _create_moon_model(model_name, out_dim, n_classes, use_projection_head)
  else:
    return _create_normal_model(model_name)


def _create_normal_model(model_name: str) -> BaseModel:
  if model_name == "mini-cnn":
    return MiniCNN()
  elif model_name == "mini-cnn-mnist":
    return MiniCNNMNIST()
  elif model_name == "simple-cnn":
    return SimpleCNN()
  elif model_name == "simple-cnn-mnist":
    return SimpleCNNMNIST()
  else:
    raise ValueError(f"Unknown model name: {model_name}.")


def _create_moon_model(model_name: str, out_dim: int = 256, n_classes: int = 10, use_projection_head: bool = True) -> BaseModel:
  """統一されたベースのMOONモデルを作成"""

  if use_projection_head:
    return ModelFedCon(base_model=model_name, out_dim=out_dim, n_classes=n_classes)
  else:
    return ModelFedCon_noheader(base_model=model_name, n_classes=n_classes)


# 統一されたモデルを作成する関数
def create_unified_model(model_name: str, n_classes: int = 10) -> BaseModel:
  """すべての手法で統一されたモデル構造を作成（projection headなし）"""
  return ModelFedCon_noheader(base_model=model_name, n_classes=n_classes)


def create_unified_moon_model(model_name: str, out_dim: int = 256, n_classes: int = 10) -> BaseModel:
  """すべての手法で統一されたモデル構造を作成（projection headあり、MOON用）"""
  return ModelFedCon(base_model=model_name, out_dim=out_dim, n_classes=n_classes)


# 後方互換性のための関数
def create_fedcon_model(base_model: str, out_dim: int = 256, n_classes: int = 10) -> BaseModel:
  """元のMOON論文準拠のModelFedConを作成"""
  return ModelFedCon(base_model=base_model, out_dim=out_dim, n_classes=n_classes)


def create_fedcon_noheader_model(base_model: str, n_classes: int = 10) -> BaseModel:
  """元のMOON論文準拠のModelFedCon_noheaderを作成"""
  return ModelFedCon_noheader(base_model=base_model, n_classes=n_classes)
