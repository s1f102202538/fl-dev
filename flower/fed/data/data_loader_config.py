from dataclasses import dataclass


@dataclass
class DataLoaderConfig:
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
