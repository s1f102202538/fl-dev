from dataclasses import dataclass


@dataclass
class DataLoaderConfig:
  """Configuration class for federated data loading.

  This class contains all configurable parameters for federated data loading,
  including dataset selection, partitioner settings, data splits, and visualization options.

  Note: Test data is always distributed using IID partitioning for fair evaluation,
  while training data follows the specified partitioner configuration.

  Example:
    ```python
    # Default configuration (Non-IID training data, IID test data)
    config = DataLoaderConfig()

    # Custom configuration with very heterogeneous training data
    config = DataLoaderConfig(
        dataset_name="zalando-datasets/fashion_mnist",
        partitioner_type="dirichlet",
        alpha=0.1,  # Very heterogeneous training data
        seed=42,
        batch_size=64,
        eval_test_samples=3000,  # Custom evaluation dataset size
        public_max_samples=2000,  # Custom public dataset size
        enable_visualization=True,
        plot_type="heatmap"
    )

    # IID training and test data
    config = DataLoaderConfig(
        dataset_name="zalando-datasets/fashion_mnist",
        partitioner_type="iid",  # IID training data
        seed=123,
        batch_size=64,
        enable_visualization=True,
        plot_type="heatmap"
    )
    ```
  """

  # Dataset configuration
  dataset_name: str

  # Federated learning configuration (optional for test-only usage)
  partition_id: int = 0
  num_partitions: int = 1

  # Partitioner configuration
  partitioner_type: str = "dirichlet"  # "dirichlet" or "iid"
  alpha: float = 0.2  # For DirichletPartitioner
  partition_by: str = "label"
  seed: int = 42

  # Data split configuration
  test_size: float = 0.2
  split_seed: int = 42

  # DataLoader configuration
  batch_size: int = 32
  shuffle_train: bool = True
  shuffle_test: bool = False

  plot_type: str = "bar"  # "bar" or "heatmap"
  size_unit: str = "absolute"  # "absolute" or "percent"

  # Training dataset configuration
  train_max_samples: int = 50000

  # Public dataset configuration
  public_max_samples: int = 8000

  # Evaluation dataset configuration (use more data for stable accuracy measurement)
  eval_test_samples: int = 2000  # Use 5000 samples for more reliable accuracy evaluation
