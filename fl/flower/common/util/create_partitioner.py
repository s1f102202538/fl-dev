from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner, Partitioner

from flower.common._class.data_loader_config import DataLoaderConfig


def create_partitioner(config: DataLoaderConfig, num_partitions: int) -> Partitioner:
  """Create partitioner based on configuration."""
  if config.partitioner_type.lower() == "dirichlet":
    return DirichletPartitioner(
      num_partitions=num_partitions,
      partition_by=config.partition_by,
      alpha=config.alpha,
      seed=config.seed,
    )
  elif config.partitioner_type.lower() == "iid":
    return IidPartitioner(num_partitions=num_partitions)
  else:
    raise ValueError(f"Unsupported partitioner type: {config.partitioner_type}")
