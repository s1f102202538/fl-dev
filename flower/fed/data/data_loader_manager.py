from typing import Optional, Tuple

from datasets import load_dataset
from flwr.common.typing import UserConfigValue
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
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
      train_partitioner = create_partitioner(self.config, num_partitions)
      test_partitioner = IidPartitioner(num_partitions=num_partitions)  # Always IID for test data

      self.fds = FederatedDataset(
        dataset=self.config.dataset_name,
        partitioners={"train": train_partitioner, "test": test_partitioner},
      )

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
      Tuple of (train_loader, test_loader)
    """
    num_partitions_int = int(num_partitions)
    partition_id_int = int(partition_id)

    # Initialize federated dataset
    self._initialize_federated_dataset(num_partitions_int)

    # Load partition data
    assert self.fds is not None  # Help type checker understand fds is initialized

    # Load train partition (follows specified partitioner) and test partition (always IID)
    train_partition = self.fds.load_partition(partition_id_int, "train")
    test_partition = self.fds.load_partition(partition_id_int, "test")

    # Apply transforms
    train_partition = train_partition.with_transform(self.transform_manager.apply_train_transforms)
    test_partition = test_partition.with_transform(self.transform_manager.apply_eval_transforms)

    train_loader = DataLoader(train_partition, batch_size=self.config.batch_size, shuffle=self.config.shuffle_train)  # type: ignore
    test_loader = DataLoader(test_partition, batch_size=self.config.batch_size, shuffle=self.config.shuffle_test)  # type: ignore

    return train_loader, test_loader

  def load_public_data(self) -> DataLoader:
    """Load public data that is common to all clients.

    Args:
      batch_size: Batch size for DataLoader
      max_samples: Maximum number of samples to load

    Returns:
      DataLoader for public data
    """
    batch_size = self.config.batch_size
    max_samples = self.config.public_max_samples

    public_dataset = load_dataset(self.config.dataset_name, split=f"test[-{max_samples}:]")

    # Create a PyTorch Dataset wrapper with transforms
    public_dataset_wrapped = PublicDataset(public_dataset, transform=self.transform_manager.eval_transforms)

    # Create DataLoader for public data
    public_loader = DataLoader(public_dataset_wrapped, batch_size=batch_size, shuffle=False)

    dataset_size = len(public_dataset_wrapped)
    expected_batches = dataset_size // batch_size + (1 if dataset_size % batch_size > 0 else 0)
    print(f"[DEBUG] Public data: {dataset_size} samples, batch_size={batch_size}, expected_batches={expected_batches}")

    return public_loader
