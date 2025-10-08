from ..models.base_model import BaseModel
from ..models.mini_cnn import MiniCNN, MiniCNNMNIST
from ..models.moon_model import MoonModel


def create_model(model_name: str, is_moon: bool = False) -> BaseModel:
  if is_moon:
    return _create_moon_model(model_name)
  else:
    return _create_normal_model(model_name)


def _create_normal_model(model_name: str) -> BaseModel:
  if model_name == "mini-cnn":
    return MiniCNN()
  elif model_name == "mini-cnn-minist":
    return MiniCNNMNIST()
  else:
    raise ValueError(f"Unknown model name: {model_name}.")


def _create_moon_model(model_name: str) -> BaseModel:
  return MoonModel(model_name)
