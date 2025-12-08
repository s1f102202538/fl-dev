from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner, Partitioner

from ..data.data_loader_config import DataLoaderConfig


def create_partitioner(config: DataLoaderConfig) -> Partitioner:
  if config.partitioner_type.lower() == "dirichlet":
    return DirichletPartitioner(
      num_partitions=config.num_partitions,
      partition_by=config.partition_by,
      alpha=config.alpha,
      seed=config.seed,
    )
  elif config.partitioner_type.lower() == "iid":
    return IidPartitioner(num_partitions=config.num_partitions)
  else:
    raise ValueError(f"Unsupported partitioner type: {config.partitioner_type}")
