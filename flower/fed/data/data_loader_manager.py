from datasets import load_dataset
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader

from ..util.create_partitioner import create_partitioner
from .data_loader_config import DataLoaderConfig
from .data_transform_manager import DataTransformManager
from .public_data import PublicDataset


class DataLoaderManager:
  @staticmethod
  def _validate_config(config: DataLoaderConfig) -> int:
    """Validate configuration and return total test samples.

    Args:
      config: DataLoaderConfig instance

    Returns:
      Total number of test samples available

    Raises:
      ValueError: If configuration is invalid
    """
    # Load dataset to check available samples
    full_test_dataset = load_dataset(config.dataset_name, split="test")
    total_test_samples = len(full_test_dataset)  # type: ignore

    # Validate configuration
    required_samples = config.eval_test_samples + config.public_max_samples
    if required_samples > total_test_samples:
      raise ValueError(
        f"Insufficient test data: need {required_samples} samples "
        f"(eval: {config.eval_test_samples} + public: {config.public_max_samples}) "
        f"but only {total_test_samples} available"
      )

    return total_test_samples

  def load_train_data(self, config: DataLoaderConfig) -> DataLoader:
    """Load training data for federated learning.

    Args:
      config: DataLoaderConfig instance containing partition settings

    Returns:
      DataLoader for training data
    """
    # Create transform manager
    transform_manager = DataTransformManager(config)

    # Create Non-IID training data partitioner
    train_partitioner = create_partitioner(config)

    # Initialize FederatedDataset for training data only
    fds = FederatedDataset(
      dataset=config.dataset_name,
      partitioners={"train": train_partitioner},
    )

    # Load train partition (Non-IID) - specific to each client
    train_partition = fds.load_partition(config.partition_id, "train")
    train_partition = train_partition.with_transform(transform_manager.apply_train_transforms)

    train_loader = DataLoader(train_partition, batch_size=config.batch_size, shuffle=config.shuffle_train)  # type: ignore

    return train_loader

  def load_test_data(self, config: DataLoaderConfig) -> DataLoader:
    """Load test data for evaluation.

    Args:
      config: DataLoaderConfig instance

    Returns:
      DataLoader for evaluation test data
    """
    # Validate configuration
    self._validate_config(config)

    # Create transform manager
    transform_manager = DataTransformManager(config)

    # Load evaluation test data (excluding public data)
    eval_test_dataset = load_dataset(config.dataset_name, split=f"test[:{config.eval_test_samples}]")
    eval_test_wrapped = PublicDataset(eval_test_dataset, transform=transform_manager.eval_transforms)

    test_loader = DataLoader(eval_test_wrapped, batch_size=config.batch_size, shuffle=config.shuffle_test)

    return test_loader

  def load_public_data(self, config: DataLoaderConfig) -> DataLoader:
    """Load public data for knowledge distillation.

    Args:
      config: DataLoaderConfig instance

    Returns:
      DataLoader for public data
    """
    # Validate configuration
    self._validate_config(config)

    # Create transform manager
    transform_manager = DataTransformManager(config)

    # Load public data from the LAST part of test split (separated from evaluation data)
    public_dataset = load_dataset(config.dataset_name, split=f"test[-{config.public_max_samples}:]")
    public_dataset_wrapped = PublicDataset(public_dataset, transform=transform_manager.eval_transforms)

    public_loader = DataLoader(public_dataset_wrapped, batch_size=config.batch_size, shuffle=False)

    return public_loader
