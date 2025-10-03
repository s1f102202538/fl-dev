from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from datasets import load_dataset
from flwr.common.typing import UserConfigValue
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner, Partitioner
from flwr_datasets.visualization import plot_label_distributions
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
  Compose,
  Normalize,
  RandomCrop,
  RandomHorizontalFlip,
  ToTensor,
)


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
  dataset_name: str = "uoft-cs/cifar10"

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

  # Visualization configuration
  enable_visualization: bool = False
  plot_type: str = "bar"  # "bar" or "heatmap"
  size_unit: str = "absolute"  # "absolute" or "percent"

  # Public dataset configuration
  public_max_samples: int = 1000


class FederatedDataLoaderManager:
  """Manages federated data loading with configurable parameters.

  This class provides a flexible and configurable interface for federated data loading,
  supporting different datasets, partitioners, and visualization options.

  Training data follows the specified partitioner configuration (e.g., Dirichlet for Non-IID),
  while test data is always distributed using IID partitioning for fair evaluation.

  Examples:
    ```python
    # Basic usage with default configuration (Non-IID train, IID test)
    manager = FederatedDataLoaderManager()
    train_loader, test_loader = manager.load_data(partition_id=0, num_partitions=2)

    # Non-IID training data with automatic IID test data
    config = DataLoaderConfig(
        dataset_name="zalando-datasets/fashion_mnist",
        partitioner_type="dirichlet",
        alpha=0.1,  # Very heterogeneous training data
        seed=42,
        enable_visualization=True
    )
    manager = FederatedDataLoaderManager(config)
    train_loader, test_loader = manager.load_data(
        partition_id=0,
        num_partitions=10,
        save_plot_path="data_distribution.png"
    )

    # Load public data for evaluation
    public_loader = manager.load_public_data(batch_size=128, max_samples=1000)

    # Standalone visualization
    df = manager.visualize_data_distribution(
        num_partitions=10,
        save_path="distribution_heatmap.png",
        plot_type="heatmap",
        size_unit="percent"
    )
    ```
  """

  def __init__(self, config: Optional[DataLoaderConfig] = None):
    """Initialize the federated data loader manager.

    Args:
      config: DataLoaderConfig instance. If None, uses default configuration.
    """
    self.config = config or DataLoaderConfig()
    self.fds: Optional[FederatedDataset] = None
    self._setup_transforms()

  def _setup_transforms(self):
    """Setup data transforms based on dataset."""
    if "fashion_mnist" in self.config.dataset_name.lower():
      normalization = ((0.1307,), (0.3081,))
      crop_size = 28
      channels = 1
    elif "cifar" in self.config.dataset_name.lower():
      # CIFAR-10/100 specific normalization
      normalization = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
      crop_size = 32
      channels = 3
    else:
      # Default CIFAR-like normalization for other datasets
      normalization = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
      crop_size = 32
      channels = 3

    self.eval_transforms = Compose([ToTensor(), Normalize(*normalization)])
    self.train_transforms = Compose(
      [
        RandomCrop(crop_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(*normalization),
      ]
    )

  def _create_partitioner(self, num_partitions: int) -> Partitioner:
    """Create partitioner based on configuration."""
    if self.config.partitioner_type.lower() == "dirichlet":
      return DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by=self.config.partition_by,
        alpha=self.config.alpha,
        seed=self.config.seed,
      )
    elif self.config.partitioner_type.lower() == "iid":
      return IidPartitioner(num_partitions=num_partitions)
    else:
      raise ValueError(f"Unsupported partitioner type: {self.config.partitioner_type}")

  def _initialize_federated_dataset(self, num_partitions: int):
    """Initialize FederatedDataset if not already done."""
    if self.fds is None:
      train_partitioner = self._create_partitioner(num_partitions)
      test_partitioner = IidPartitioner(num_partitions=num_partitions)  # Always IID for test data

      self.fds = FederatedDataset(
        dataset=self.config.dataset_name,
        partitioners={"train": train_partitioner, "test": test_partitioner},
      )

  def load_data(
    self, partition_id: UserConfigValue, num_partitions: UserConfigValue, visualize: Optional[bool] = None, save_plot_path: Optional[str] = None
  ) -> Tuple[DataLoader, DataLoader]:
    """Load partition data for federated learning.

    Args:
      partition_id: ID of the partition to load
      num_partitions: Total number of partitions
      visualize: Whether to show data distribution visualization
      save_plot_path: Path to save the visualization plot

    Returns:
      Tuple of (train_loader, test_loader)
    """
    num_partitions_int = int(num_partitions)
    partition_id_int = int(partition_id)

    # Initialize federated dataset
    self._initialize_federated_dataset(num_partitions_int)

    # Show visualization if requested (only when first initializing)
    if visualize is None:
      visualize = self.config.enable_visualization

    if visualize and partition_id_int == 0:  # Only visualize once for first partition
      print(f"Visualizing data distribution for {num_partitions_int} partitions...")
      self.visualize_data_distribution(
        num_partitions=num_partitions_int, save_path=save_plot_path, plot_type=self.config.plot_type, size_unit=self.config.size_unit
      )

    # Load partition data
    assert self.fds is not None  # Help type checker understand fds is initialized

    # Load train partition (follows specified partitioner) and test partition (always IID)
    train_partition = self.fds.load_partition(partition_id_int, "train")
    test_partition = self.fds.load_partition(partition_id_int, "test")

    # Apply transforms
    train_partition = train_partition.with_transform(self._apply_train_transforms)
    test_partition = test_partition.with_transform(self._apply_eval_transforms)

    train_loader = DataLoader(train_partition, batch_size=self.config.batch_size, shuffle=self.config.shuffle_train)  # type: ignore
    test_loader = DataLoader(test_partition, batch_size=self.config.batch_size, shuffle=self.config.shuffle_test)  # type: ignore

    return train_loader, test_loader

  def load_public_data(self, batch_size: Optional[int] = None, max_samples: Optional[int] = None) -> DataLoader:
    """Load public data that is common to all clients.

    Args:
      batch_size: Batch size for DataLoader
      max_samples: Maximum number of samples to load

    Returns:
      DataLoader for public data
    """
    batch_size = batch_size or self.config.batch_size
    max_samples = max_samples or self.config.public_max_samples

    # Load the test split of dataset
    public_dataset = load_dataset(self.config.dataset_name, split=f"test[:{max_samples}]")

    # Create a PyTorch Dataset wrapper with transforms
    public_dataset_wrapped = PublicDataset(public_dataset, transform=self.eval_transforms)

    # Create DataLoader for public data
    public_loader = DataLoader(public_dataset_wrapped, batch_size=batch_size, shuffle=False)

    dataset_size = len(public_dataset_wrapped)
    expected_batches = dataset_size // batch_size + (1 if dataset_size % batch_size > 0 else 0)
    print(f"[DEBUG] Public data: {dataset_size} samples, batch_size={batch_size}, expected_batches={expected_batches}")

    return public_loader

  def visualize_data_distribution(
    self, num_partitions: int, save_path: Optional[str] = None, plot_type: Optional[str] = None, size_unit: Optional[str] = None, quiet: bool = False
  ):
    """Visualize the data distribution across federated partitions.

    Args:
      num_partitions: Number of partitions used in the federated dataset
      save_path: Optional path to save the plot
      plot_type: Type of plot ("bar" or "heatmap")
      size_unit: Size unit ("absolute" or "percent")
      quiet: If True, suppress detailed console output

    Returns:
      DataFrame containing the data distribution matrix
    """
    plot_type = plot_type or self.config.plot_type
    size_unit = size_unit or self.config.size_unit

    # Create partitioner for visualization
    partitioner = self._create_partitioner(num_partitions)

    # Create FederatedDataset with the partitioner
    fds_vis = FederatedDataset(
      dataset=self.config.dataset_name,
      partitioners={"train": partitioner},
    )

    # Get the partitioner from the federated dataset
    partitioner_with_data = fds_vis.partitioners["train"]

    # For heatmap, create both absolute and percentage versions
    if plot_type == "heatmap":
      return self._create_dual_heatmaps(partitioner_with_data, num_partitions, save_path, quiet)
    else:
      # Create single visualization for bar plots
      dataset_name = self.config.dataset_name.split("/")[-1].upper()
      alpha_info = f" (α={self.config.alpha})" if self.config.partitioner_type.lower() == "dirichlet" else ""
      title = f"{dataset_name} Data Distribution{alpha_info} ({size_unit.capitalize()} counts, {plot_type} plot)"
      plot_kwargs = {"annot": True} if plot_type == "heatmap" else {}

      fig, ax, df = plot_label_distributions(
        partitioner=partitioner_with_data,
        label_name="label",
        plot_type=plot_type,
        size_unit=size_unit,
        partition_id_axis="x",
        legend=True,
        verbose_labels=True,
        title=title,
        cmap="tab10" if plot_type == "bar" else "YlOrRd",
        plot_kwargs=plot_kwargs,
      )

      if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Data distribution plot saved to: {save_path}")
      else:
        import matplotlib.pyplot as plt

        plt.show()

      return self._print_analysis_and_return_df(df, size_unit, num_partitions)

  def _create_dual_heatmaps(self, partitioner_with_data, num_partitions: int, save_path: Optional[str] = None, quiet: bool = False):
    """Create individual absolute and percentage heatmaps for data distribution."""
    import matplotlib.pyplot as plt

    dataset_name = self.config.dataset_name.split("/")[-1].upper()
    alpha_info = f" (α={self.config.alpha})" if self.config.partitioner_type.lower() == "dirichlet" else ""

    # Create absolute counts heatmap
    fig1, ax1_temp, df_abs = plot_label_distributions(
      partitioner=partitioner_with_data,
      label_name="label",
      plot_type="heatmap",
      size_unit="absolute",
      partition_id_axis="x",
      legend=True,
      verbose_labels=True,
      title=f"{dataset_name} Data Distribution{alpha_info} (Absolute counts)",
      cmap="YlOrRd",
      plot_kwargs={"annot": True, "fmt": "d"},
    )
    plt.close(fig1)  # Close the temporary figure

    # Create percentage heatmap
    fig2, ax2_temp, df_pct = plot_label_distributions(
      partitioner=partitioner_with_data,
      label_name="label",
      plot_type="heatmap",
      size_unit="percent",
      partition_id_axis="x",
      legend=True,
      verbose_labels=True,
      title=f"{dataset_name} Data Distribution{alpha_info} (Percentage)",
      cmap="YlOrRd",
      plot_kwargs={"annot": True, "fmt": ".1f"},
    )
    plt.close(fig2)  # Close the temporary figure

    # Manually create individual heatmaps
    import seaborn as sns

    # Transpose DataFrames to have Partition ID on x-axis and Class Labels on y-axis
    df_abs_T = df_abs.T
    df_pct_T = df_pct.T

    if save_path:
      # Generate paths for individual heatmaps
      if save_path.endswith(".png"):
        abs_path = save_path.replace(".png", "_absolute.png")
        pct_path = save_path.replace(".png", "_percentage.png")
      else:
        abs_path = f"{save_path}_absolute.png"
        pct_path = f"{save_path}_percentage.png"

      # Create and save absolute counts heatmap
      fig_abs, ax_abs = plt.subplots(figsize=(12, 8))
      sns.heatmap(df_abs_T, annot=True, fmt="d", cmap="YlOrRd", ax=ax_abs, cbar_kws={"label": "Sample Count"})
      ax_abs.set_title(f"{dataset_name} Data Distribution{alpha_info} (Absolute Counts)", fontsize=14, fontweight="bold")
      ax_abs.set_xlabel("Partition ID")
      ax_abs.set_ylabel("Class Labels")
      fig_abs.savefig(abs_path, dpi=300, bbox_inches="tight")
      plt.close(fig_abs)

      # Create and save percentage heatmap
      fig_pct, ax_pct = plt.subplots(figsize=(12, 8))
      sns.heatmap(df_pct_T, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax_pct, cbar_kws={"label": "Percentage (%)"})
      ax_pct.set_title(f"{dataset_name} Data Distribution{alpha_info} (Percentage)", fontsize=14, fontweight="bold")
      ax_pct.set_xlabel("Partition ID")
      ax_pct.set_ylabel("Class Labels")
      fig_pct.savefig(pct_path, dpi=300, bbox_inches="tight")
      plt.close(fig_pct)

      print(f"Individual absolute heatmap saved to: {abs_path}")
      print(f"Individual percentage heatmap saved to: {pct_path}")
    else:
      # Display the heatmaps if no save path is provided
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

      # Plot absolute counts heatmap
      sns.heatmap(df_abs_T, annot=True, fmt="d", cmap="YlOrRd", ax=ax1, cbar_kws={"label": "Sample Count"})
      ax1.set_title(f"{dataset_name} Data Distribution{alpha_info} (Absolute Counts)", fontsize=14, fontweight="bold")
      ax1.set_xlabel("Partition ID")
      ax1.set_ylabel("Class Labels")

      # Plot percentage heatmap
      sns.heatmap(df_pct_T, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax2, cbar_kws={"label": "Percentage (%)"})
      ax2.set_title(f"{dataset_name} Data Distribution{alpha_info} (Percentage)", fontsize=14, fontweight="bold")
      ax2.set_xlabel("Partition ID")
      ax2.set_ylabel("Class Labels")

      plt.tight_layout()
      plt.show()

    # Print analysis for both absolute and percentage data (only if not quiet)
    if not quiet:
      print("\n=== Absolute Counts Analysis ===")
      self._print_analysis_and_return_df(df_abs, "absolute", len(df_abs.index), print_df=True)

      print("\n=== Percentage Analysis ===")
      self._print_analysis_and_return_df(df_pct, "percent", len(df_pct.index), print_df=True)

    return df_abs  # Return absolute counts as primary data

  def _print_analysis_and_return_df(self, df, size_unit: str, num_partitions: int, print_df: bool = True):
    """Print analysis and return DataFrame."""
    if print_df:
      # Print the distribution DataFrame for detailed analysis
      print(f"\nData distribution matrix ({size_unit}):")
      print("Rows: Partition IDs, Columns: Class labels")
      print(df)

    # Calculate and display heterogeneity metrics
    print(f"\nData heterogeneity analysis ({size_unit}):")
    for partition_id in range(num_partitions):
      if partition_id in df.index:
        partition_data = df.iloc[partition_id]
        total_samples = int(partition_data.sum()) if size_unit == "absolute" else 100
        max_class_samples = int(partition_data.max()) if size_unit == "absolute" else partition_data.max()
        dominant_class = partition_data.idxmax()
        dominance_ratio = max_class_samples / total_samples if total_samples > 0 else 0.0
        num_classes_with_data = int((partition_data > 0).sum())

        if size_unit == "absolute":
          print(f"Partition {partition_id}:")
          print(f"  - Total samples: {total_samples}")
          print(f"  - Dominant class: {dominant_class} ({max_class_samples} samples, {dominance_ratio:.1%})")
          print(f"  - Classes with data: {num_classes_with_data}/{len(df.columns)}")
        else:
          print(f"Partition {partition_id}:")
          print(f"  - Dominant class: {dominant_class} ({max_class_samples:.1f}%)")
          print(f"  - Classes with data: {num_classes_with_data}/{len(df.columns)}")

    return df

  def _apply_train_transforms(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Apply transforms to the training partition."""
    # CIFAR-10データセットでは'img'キーを使用
    image_key = "img" if "img" in batch else "image"

    # PIL.Imageをテンソルに変換
    from PIL import Image

    transformed_images = []
    for img in batch[image_key]:
      if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
      transformed_images.append(self.train_transforms(img))

    # 元のキーを削除して新しいキーに設定
    if image_key != "image":
      del batch[image_key]
    batch["image"] = transformed_images
    return batch

  def _apply_eval_transforms(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Apply transforms to the evaluation partition."""
    # CIFAR-10データセットでは'img'キーを使用
    image_key = "img" if "img" in batch else "image"

    # PIL.Imageをテンソルに変換
    from PIL import Image

    transformed_images = []
    for img in batch[image_key]:
      if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
      transformed_images.append(self.eval_transforms(img))

    # 元のキーを削除して新しいキーに設定
    if image_key != "image":
      del batch[image_key]
    batch["image"] = transformed_images
    return batch


class PublicDataset(Dataset):
  """PyTorch Dataset wrapper for public dataset with transforms."""

  def __init__(self, hf_dataset, transform=None):
    self.hf_dataset = hf_dataset
    self.transform = transform

  def __len__(self):
    return len(self.hf_dataset)

  def __getitem__(self, idx):
    item = self.hf_dataset[idx]
    # CIFAR-10データセットでは'img'キーを使用
    image_key = "img" if "img" in item else "image"
    image = item[image_key]
    label = item["label"]

    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
      image = Image.fromarray(image)

    if self.transform:
      image = self.transform(image)

    return {"image": image, "label": label}


# ============================================================
# Backward Compatibility: Legacy Functions
# ============================================================

# Global variables for backward compatibility
fds = None  # Cache FederatedDataset
FM_NORMALIZATION = ((0.1307,), (0.3081,))
EVAL_TRANSFORMS = Compose([ToTensor(), Normalize(*FM_NORMALIZATION)])
TRAIN_TRANSFORMS = Compose(
  [
    RandomCrop(28, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(*FM_NORMALIZATION),
  ]
)

# Default data loader manager instance for legacy functions
_default_data_manager = FederatedDataLoaderManager()


def load_data(
  partition_id: UserConfigValue, num_partitions: UserConfigValue, visualize: bool = False, save_plot_path: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
  """Legacy function for backward compatibility.

  Use FederatedDataLoaderManager for new implementations.
  """
  return _default_data_manager.load_data(partition_id=partition_id, num_partitions=num_partitions, visualize=visualize, save_plot_path=save_plot_path)


def load_public_data(batch_size: int = 32, max_samples: int = 1000) -> DataLoader:
  """Legacy function for backward compatibility.

  Use FederatedDataLoaderManager for new implementations.
  """
  return _default_data_manager.load_public_data(batch_size=batch_size, max_samples=max_samples)


def visualize_data_distribution(num_partitions: int, save_path: Optional[str] = None, plot_type: str = "bar", size_unit: str = "absolute"):
  """Legacy function for backward compatibility.

  Use FederatedDataLoaderManager for new implementations.
  """
  return _default_data_manager.visualize_data_distribution(num_partitions=num_partitions, save_path=save_path, plot_type=plot_type, size_unit=size_unit)


def apply_train_transforms(batch: Dict[str, Any]) -> Dict[str, Any]:
  """Legacy function for backward compatibility."""
  batch["image"] = [TRAIN_TRANSFORMS(img) for img in batch["image"]]
  return batch


def apply_eval_transforms(batch: Dict[str, Any]) -> Dict[str, Any]:
  """Legacy function for backward compatibility."""
  batch["image"] = [EVAL_TRANSFORMS(img) for img in batch["image"]]
  return batch
