from typing import Any, Dict, Tuple

from flwr.common.typing import UserConfigValue
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import (
  Compose,
  Normalize,
  RandomCrop,
  RandomHorizontalFlip,
  ToTensor,
)

fds = None  # Cache FederatedDataset

FM_NORMALIZATION = ((0.1307,), (0.3081,))
EVAL_TRANSFORMS = Compose([ToTensor(), Normalize(*FM_NORMALIZATION)])
TRAIN_TRANSFORMS = Compose(
  [
    RandomCrop(28, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(*FM_NORMALIZATION),
  ]
)


def load_data(partition_id: UserConfigValue, num_partitions: UserConfigValue) -> Tuple[DataLoader, DataLoader]:
  """Load partition FashionMNIST data."""
  # Only initialize `FederatedDataset` once
  global fds
  if fds is None:
    partitioner = DirichletPartitioner(
      num_partitions=num_partitions,
      partition_by="label",
      alpha=1.0,
      seed=42,
    )
    fds = FederatedDataset(
      dataset="zalando-datasets/fashion_mnist",
      partitioners={"train": partitioner},
    )
  partition = fds.load_partition(partition_id)
  # Divide data on each node: 80% train, 20% test
  partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

  train_partition = partition_train_test["train"].with_transform(apply_train_transforms)
  test_partition = partition_train_test["test"].with_transform(apply_eval_transforms)
  train_loader = DataLoader(train_partition, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_partition, batch_size=32)
  return train_loader, test_loader


def apply_train_transforms(batch: Dict[str, Any]) -> Dict[str, Any]:
  """Apply transforms to the partition from FederatedDataset."""
  batch["image"] = [TRAIN_TRANSFORMS(img) for img in batch["image"]]
  return batch


def apply_eval_transforms(batch: Dict[str, Any]) -> Dict[str, Any]:
  """Apply transforms to the partition from FederatedDataset."""
  batch["image"] = [EVAL_TRANSFORMS(img) for img in batch["image"]]
  return batch
