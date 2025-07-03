from typing import Any, Dict, Tuple

from datasets import load_dataset
from flwr.common.typing import UserConfigValue
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
  Compose,
  Normalize,
  RandomCrop,
  RandomHorizontalFlip,
  ToTensor,
)


class PublicDataset(Dataset):
  """PyTorch Dataset wrapper for public dataset with transforms."""

  def __init__(self, hf_dataset, transform=None):
    self.hf_dataset = hf_dataset
    self.transform = transform

  def __len__(self):
    return len(self.hf_dataset)

  def __getitem__(self, idx):
    item = self.hf_dataset[idx]
    image = item["image"]
    label = item["label"]

    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
      image = Image.fromarray(image)

    if self.transform:
      image = self.transform(image)

    return {"image": image, "label": label}


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
      num_partitions=int(num_partitions),
      partition_by="label",
      alpha=1.0,
      seed=42,
    )
    fds = FederatedDataset(
      dataset="zalando-datasets/fashion_mnist",
      partitioners={"train": partitioner},
    )
  partition = fds.load_partition(int(partition_id))
  # Divide data on each node: 80% train, 20% test
  partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

  train_partition = partition_train_test["train"].with_transform(apply_train_transforms)
  test_partition = partition_train_test["test"].with_transform(apply_eval_transforms)

  train_loader = DataLoader(train_partition, batch_size=32, shuffle=True)  # type: ignore
  test_loader = DataLoader(test_partition, batch_size=32)  # type: ignore
  return train_loader, test_loader


def load_public_data(batch_size: int = 32) -> DataLoader:
  """Load public FashionMNIST test data that is common to all clients."""
  # Load the test split of FashionMNIST dataset
  public_dataset = load_dataset("zalando-datasets/fashion_mnist", split="test")

  # Create a PyTorch Dataset wrapper with transforms
  public_dataset_wrapped = PublicDataset(public_dataset, transform=EVAL_TRANSFORMS)

  # Create DataLoader for public data
  public_loader = DataLoader(public_dataset_wrapped, batch_size=batch_size, shuffle=False)

  return public_loader


def apply_train_transforms(batch: Dict[str, Any]) -> Dict[str, Any]:
  """Apply transforms to the partition from FederatedDataset."""
  batch["image"] = [TRAIN_TRANSFORMS(img) for img in batch["image"]]
  return batch


def apply_eval_transforms(batch: Dict[str, Any]) -> Dict[str, Any]:
  """Apply transforms to the partition from FederatedDataset."""
  batch["image"] = [EVAL_TRANSFORMS(img) for img in batch["image"]]
  return batch
