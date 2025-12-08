from typing import Tuple

from torch.utils.data import DataLoader

from ..data.data_loader_config import DataLoaderConfig
from ..data.data_loader_manager import DataLoaderManager


def load_train_data(config: DataLoaderConfig) -> DataLoader:
  data_manager = DataLoaderManager()
  return data_manager.load_train_data(config)


def load_test_data(config: DataLoaderConfig) -> DataLoader:
  data_manager = DataLoaderManager()
  return data_manager.load_test_data(config)


def load_public_data(config: DataLoaderConfig) -> DataLoader:
  data_manager = DataLoaderManager()
  return data_manager.load_public_data(config)


def load_data(config: DataLoaderConfig) -> Tuple[DataLoader, DataLoader]:
  data_manager = DataLoaderManager()
  train_loader = data_manager.load_train_data(config)
  test_loader = data_manager.load_test_data(config)
  return train_loader, test_loader
