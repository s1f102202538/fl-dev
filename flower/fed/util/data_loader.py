from typing import Tuple

from torch.utils.data import DataLoader

from ..data.data_loader_config import DataLoaderConfig
from ..data.data_loader_manager import DataLoaderManager


def load_train_data(config: DataLoaderConfig) -> DataLoader:
  """Load training data for federated learning.

  Args:
    config: DataLoaderConfig instance containing partition settings

  Returns:
    DataLoader for training data
  """
  data_manager = DataLoaderManager()
  return data_manager.load_train_data(config)


def load_test_data(config: DataLoaderConfig) -> DataLoader:
  """Load test data for evaluation.

  Args:
    config: DataLoaderConfig instance

  Returns:
    DataLoader for evaluation test data
  """
  data_manager = DataLoaderManager()
  return data_manager.load_test_data(config)


def load_public_data(config: DataLoaderConfig) -> DataLoader:
  """Load public data for knowledge distillation.

  Args:
    config: DataLoaderConfig instance

  Returns:
    DataLoader for public data
  """
  data_manager = DataLoaderManager()
  return data_manager.load_public_data(config)


def load_data(config: DataLoaderConfig) -> Tuple[DataLoader, DataLoader]:
  """Load both training and test data (backward compatibility).

  Args:
    config: DataLoaderConfig instance

  Returns:
    Tuple of (train_loader, test_loader)
  """
  data_manager = DataLoaderManager()
  train_loader = data_manager.load_train_data(config)
  test_loader = data_manager.load_test_data(config)
  return train_loader, test_loader
