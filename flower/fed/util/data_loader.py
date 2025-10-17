from typing import Tuple

from flwr.common.typing import UserConfigValue
from torch.utils.data import DataLoader

from ..data.data_loader_config import DataLoaderConfig
from ..data.data_loader_manager import DataLoaderManager


def load_data(config: DataLoaderConfig, partition_id: UserConfigValue, num_partitions: UserConfigValue) -> Tuple[DataLoader, DataLoader]:
  default_data_manager = DataLoaderManager(config)
  return default_data_manager.load_data(partition_id=partition_id, num_partitions=num_partitions)


def load_public_data(config: DataLoaderConfig) -> DataLoader:
  default_data_manager = DataLoaderManager(config)
  return default_data_manager.load_public_data()
