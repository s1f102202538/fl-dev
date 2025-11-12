from typing import Optional, Tuple

from datasets import load_dataset
from flwr.common.typing import UserConfigValue
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader

from ..util.create_partitioner import create_partitioner
from .data_loader_config import DataLoaderConfig
from .data_transform_manager import DataTransformManager
from .public_data import PublicDataset


class DataLoaderManager:
  def __init__(self, config: DataLoaderConfig):
    """Initialize the federated data loader manager.

    Args:
      config: DataLoaderConfig instance. If None, uses default configuration.
    """
    self.config = config
    self.transform_manager = DataTransformManager(self.config)
    self.fds: Optional[FederatedDataset] = None

  def _initialize_federated_dataset(self, num_partitions: int):
    """Initialize FederatedDataset if not already done."""
    if self.fds is None:
      # Create partitioners
      train_partitioner = create_partitioner(self.config, num_partitions)
      # Note: Test data will be handled separately as common data for all clients

      self.fds = FederatedDataset(dataset=self.config.dataset_name, partitioners={"train": train_partitioner})

  def load_data(
    self,
    partition_id: UserConfigValue,
    num_partitions: UserConfigValue,
  ) -> Tuple[DataLoader, DataLoader]:
    """Load partition data for federated learning.

    Args:
      partition_id: ID of the partition to load
      num_partitions: Total number of partitions

    Returns:
      Tuple of (train_loader, test_loader) where test_loader is common for all clients
    """
    num_partitions_int = int(num_partitions)
    partition_id_int = int(partition_id)

    # Initialize federated dataset
    self._initialize_federated_dataset(num_partitions_int)

    # Load partition data
    assert self.fds is not None  # Help type checker understand fds is initialized

    # Load train partition (client-specific)
    train_partition = self.fds.load_partition(partition_id_int, "train")

    # Load common test data for all clients (from the beginning, excluding public data at the end)
    full_test_dataset = load_dataset(self.config.dataset_name, split="test")
    total_test_samples = len(full_test_dataset)
    max_public_samples = min(self.config.public_max_samples, total_test_samples)
    available_test_samples = max(0, total_test_samples - max_public_samples)

    if available_test_samples > 0:
      # Use the first 'available_test_samples' for common evaluation
      test_dataset = full_test_dataset.select(range(available_test_samples))
    else:
      # If no samples available, create empty dataset
      test_dataset = full_test_dataset.select([])

    # Apply transforms
    train_partition = train_partition.with_transform(self.transform_manager.apply_train_transforms)
    test_partition = test_dataset.with_transform(self.transform_manager.apply_eval_transforms)

    train_loader = DataLoader(train_partition, batch_size=self.config.batch_size, shuffle=self.config.shuffle_train)  # type: ignore
    test_loader = DataLoader(test_partition, batch_size=self.config.batch_size, shuffle=self.config.shuffle_test)  # type: ignore

    return train_loader, test_loader

  def load_public_data(self) -> DataLoader:
    """Load public data that is common to all clients.

    Uses the last max_samples from the test dataset to ensure no overlap
    with federated test data.

    Returns:
      DataLoader for public data
    """
    batch_size = self.config.batch_size
    max_samples = self.config.public_max_samples

    # Load the full test dataset and take the last max_samples
    full_test_dataset = load_dataset(self.config.dataset_name, split="test")
    total_samples = len(full_test_dataset)

    if max_samples > 0 and total_samples > 0:
      # Take the last max_samples samples to avoid overlap with federated test data
      start_idx = max(0, total_samples - max_samples)
      public_dataset = full_test_dataset.select(range(start_idx, total_samples))
    else:
      # If max_samples is 0 or dataset is empty, create empty dataset
      public_dataset = full_test_dataset.select([])

    # Create a PyTorch Dataset wrapper with transforms
    public_dataset_wrapped = PublicDataset(public_dataset, transform=self.transform_manager.eval_transforms)

    # Create DataLoader for public data
    public_loader = DataLoader(public_dataset_wrapped, batch_size=batch_size, shuffle=False)

    dataset_size = len(public_dataset_wrapped)
    expected_batches = dataset_size // batch_size + (1 if dataset_size % batch_size > 0 else 0)
    print(f"[DEBUG] Public data: {dataset_size} samples, batch_size={batch_size}, expected_batches={expected_batches}")
    print(f"[DEBUG] Public data taken from test dataset indices {start_idx if max_samples > 0 else 0} to {total_samples}")

    return public_loader
