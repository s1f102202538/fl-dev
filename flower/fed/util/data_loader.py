from typing import Tuple

from flwr.common.typing import UserConfigValue
from torch.utils.data import DataLoader
from torchvision.transforms import (
  Compose,
  Normalize,
  RandomCrop,
  RandomHorizontalFlip,
  ToTensor,
)

from ..data.data_loader_config import DataLoaderConfig
from ..data.data_loader_manager import DataLoaderManager

# Global variables for backward compatibility
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


def load_data(config: DataLoaderConfig, partition_id: UserConfigValue, num_partitions: UserConfigValue) -> Tuple[DataLoader, DataLoader]:
  default_data_manager = DataLoaderManager(config)
  return default_data_manager.load_data(partition_id=partition_id, num_partitions=num_partitions)


def load_public_data(config: DataLoaderConfig, batch_size: int = 32, max_samples: int = 1000) -> DataLoader:
  default_data_manager = DataLoaderManager(config)
  return default_data_manager.load_public_data(batch_size=batch_size, max_samples=max_samples)
