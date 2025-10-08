from typing import Any, Dict

from PIL import Image
from torchvision.transforms import (
  Compose,
  Normalize,
  RandomCrop,
  RandomHorizontalFlip,
  ToTensor,
)

from .data_loader_config import DataLoaderConfig


class DataTransformManager:
  def __init__(self, config: DataLoaderConfig):
    self.config = config
    self._setup_transforms()

  def _setup_transforms(self):
    """Setup data transforms based on dataset."""
    if "fashion_mnist" in self.config.dataset_name.lower():
      normalization = ((0.1307,), (0.3081,))
      crop_size = 28
      channels = 1
    elif "cifar" in self.config.dataset_name.lower():
      # CIFAR-10/100 specific normalization
      normalization = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
      crop_size = 32
      channels = 3
    else:
      # Default CIFAR-like normalization for other datasets
      normalization = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
      crop_size = 32
      channels = 3

    self.eval_transforms = Compose([ToTensor(), Normalize(*normalization)])
    self.train_transforms = Compose(
      [
        RandomCrop(crop_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(*normalization),
      ]
    )

  def apply_train_transforms(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Apply transforms to the training partition."""
    # CIFAR-10データセットでは'img'キーを使用
    image_key = "img" if "img" in batch else "image"

    transformed_images = []
    for img in batch[image_key]:
      if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
      transformed_images.append(self.train_transforms(img))

    # 元のキーを削除して新しいキーに設定
    if image_key != "image":
      del batch[image_key]
    batch["image"] = transformed_images
    return batch

  def apply_eval_transforms(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Apply transforms to the evaluation partition."""
    # CIFAR-10データセットでは'img'キーを使用
    image_key = "img" if "img" in batch else "image"

    transformed_images = []
    for img in batch[image_key]:
      if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
      transformed_images.append(self.eval_transforms(img))

    # 元のキーを削除して新しいキーに設定
    if image_key != "image":
      del batch[image_key]
    batch["image"] = transformed_images
    return batch
