from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.visualization import plot_label_distributions

from flower.common._class.data_loader_config import DataLoaderConfig
from flower.common.util.create_partitioner import create_partitioner


def visualize_data_distribution(
  config: DataLoaderConfig,
  num_partitions: int,
  save_path: str,
):
  plot_type = config.plot_type

  partitioner = create_partitioner(config, num_partitions)

  fds = FederatedDataset(
    dataset=config.dataset_name,
    partitioners={"train": partitioner},
  )

  # Get the partitioner from the federated dataset
  partitioner_with_data = fds.partitioners["train"]

  # For heatmap, create both absolute and percentage versions
  if plot_type == "heatmap":
    return create_dual_heatmaps(config, partitioner_with_data, save_path)
  else:
    # Create single visualization for bar plots
    return create_bar(config, partitioner_with_data, save_path)


def create_dual_heatmaps(config: DataLoaderConfig, partitioner_with_data: Partitioner, save_path: Optional[str] = None):
  """Create individual absolute and percentage heatmaps for data distribution."""

  dataset_name = config.dataset_name.split("/")[-1].upper()
  alpha_info = f" (α={config.alpha})" if config.partitioner_type.lower() == "dirichlet" else ""

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

  return df_abs  # Return absolute counts as primary data


def create_bar(config: DataLoaderConfig, partitioner_with_data: Partitioner, save_path: Optional[str] = None):
  plot_type = config.plot_type
  size_unit = config.size_unit
  dataset_name = config.dataset_name.split("/")[-1].upper()
  alpha_info = f" (α={config.alpha})" if config.partitioner_type.lower() == "dirichlet" else ""
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
