"""モデルファクトリ - データセットに応じて適切なモデルを作成"""

from typing import Union

from flower.common.models.mini_cnn import MiniCNN, MiniCNNFeatures, MiniCNNFeaturesMNIST, MiniCNNMNIST
from flower.common.models.moon_model import MoonModel


def create_model_for_dataset(dataset_name: str, model_type: str = "mini_cnn") -> Union[MiniCNN, MiniCNNMNIST, MoonModel]:
  """データセット名に基づいて適切なモデルを作成

  Args:
      dataset_name: データセット名（例: "uoft-cs/cifar10", "zalando-datasets/fashion_mnist"）
      model_type: モデルタイプ（"mini_cnn" or "moon"）

  Returns:
      適切なモデルインスタンス
  """
  is_mnist_like = "fashion_mnist" in dataset_name.lower() or "mnist" in dataset_name.lower()

  if model_type == "moon":
    # MoonModelの場合、内部でMiniCNNFeaturesを使用するので対応を考慮
    return MoonModel(out_dim=256, n_classes=10)
  elif model_type == "mini_cnn":
    if is_mnist_like:
      return MiniCNNMNIST()  # 1チャンネル、28x28用
    else:
      return MiniCNN()  # 3チャンネル、32x32用（CIFAR-10など）
  else:
    raise ValueError(f"Unknown model type: {model_type}")


def create_features_for_dataset(dataset_name: str) -> Union[MiniCNNFeatures, MiniCNNFeaturesMNIST]:
  """データセット名に基づいて適切な特徴抽出器を作成

  Args:
      dataset_name: データセット名

  Returns:
      適切な特徴抽出器インスタンス
  """
  is_mnist_like = "fashion_mnist" in dataset_name.lower() or "mnist" in dataset_name.lower()

  if is_mnist_like:
    return MiniCNNFeaturesMNIST()  # 1チャンネル、28x28用
  else:
    return MiniCNNFeatures()  # 3チャンネル、32x32用（CIFAR-10など）


def get_dataset_info(dataset_name: str) -> dict:
  """データセット情報を取得

  Args:
      dataset_name: データセット名

  Returns:
      データセット情報辞書
  """
  if "fashion_mnist" in dataset_name.lower() or "mnist" in dataset_name.lower():
    return {"channels": 1, "image_size": 28, "num_classes": 10, "name": "MNIST-like"}
  elif "cifar10" in dataset_name.lower():
    return {"channels": 3, "image_size": 32, "num_classes": 10, "name": "CIFAR-10"}
  elif "cifar100" in dataset_name.lower():
    return {"channels": 3, "image_size": 32, "num_classes": 100, "name": "CIFAR-100"}
  else:
    # デフォルト（CIFAR-10相当）
    return {"channels": 3, "image_size": 32, "num_classes": 10, "name": "Unknown (default to CIFAR-10-like)"}
