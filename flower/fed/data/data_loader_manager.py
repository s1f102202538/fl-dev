import os
from pathlib import Path

from datasets import load_dataset
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader

from ..util.create_partitioner import create_partitioner
from .data_loader_config import DataLoaderConfig
from .data_transform_manager import DataTransformManager
from .public_data import PublicDataset

# Ensure Hugging Face Datasets cache directory exists
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "datasets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR)


class DataLoaderManager:
  # Class variables for dataset caching (memory shared)
  _dataset_cache = {}
  _preloading_done = set()  # Preloading completion flags

  @classmethod
  def _preload_dataset_once(cls, dataset_name: str):
    """Preload dataset once to avoid API rate limiting."""
    if dataset_name in cls._preloading_done:
      return

    print(f"Starting initial preload for dataset '{dataset_name}'...")

    try:
      # Preload basic splits
      splits_to_preload = ["train", "test"]

      for split in splits_to_preload:
        cache_key = f"{dataset_name}:{split}"
        if cache_key not in cls._dataset_cache:
          print(f"  Downloading {split} data...")
          # Explicitly specify cache directory
          cls._dataset_cache[cache_key] = load_dataset(dataset_name, split=split, cache_dir=str(CACHE_DIR))
          print(f"  Loaded {split}: {len(cls._dataset_cache[cache_key])} samples")
      cls._preloading_done.add(dataset_name)
      print(f"Dataset '{dataset_name}' preloading completed!")

    except Exception as e:
      print(f"Error during preloading (continuing execution): {e}")
      # Set flag even on error to prevent duplicate attempts
      cls._preloading_done.add(dataset_name)

  @classmethod
  def _get_cached_dataset(cls, dataset_name: str, split: str):
    """Get cached dataset with automatic initial preloading."""
    # Preload entire dataset on first access only
    cls._preload_dataset_once(dataset_name)

    cache_key = f"{dataset_name}:{split}"

    if cache_key not in cls._dataset_cache:
      print(f"Downloading additional dataset: {cache_key}")
      cls._dataset_cache[cache_key] = load_dataset(dataset_name, split=split, cache_dir=str(CACHE_DIR))
    else:
      print(f"Using cached dataset: {cache_key}")

    return cls._dataset_cache[cache_key]

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
    # Load dataset to check available samples (using cache)
    full_test_dataset = DataLoaderManager._get_cached_dataset(config.dataset_name, "test")
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
    # Preload dataset (first time only, to avoid API rate limiting)
    self._preload_dataset_once(config.dataset_name)

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

    # Load evaluation test data (excluding public data, using cache)
    eval_test_dataset = DataLoaderManager._get_cached_dataset(config.dataset_name, f"test[:{config.eval_test_samples}]")

    eval_test_wrapped = PublicDataset(eval_test_dataset, transform=transform_manager.eval_transforms)

    test_loader = DataLoader(eval_test_wrapped, batch_size=config.batch_size, shuffle=config.shuffle_test, drop_last=True)

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

    # Load public data from the LAST part of test split (separated from evaluation data, using cache)
    public_dataset = DataLoaderManager._get_cached_dataset(config.dataset_name, f"test[-{config.public_max_samples}:]")

    public_dataset_wrapped = PublicDataset(public_dataset, transform=transform_manager.eval_transforms)

    public_loader = DataLoader(public_dataset_wrapped, batch_size=config.batch_size, shuffle=False, drop_last=True)

    return public_loader
