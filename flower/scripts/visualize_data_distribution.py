#!/usr/bin/env python3
"""
Dataset Distribution Visualization Script for Federated Learning

This script provides a command-line interface for visualizing data distribution
across federated partitions using various partitioners and datasets.

=== PARTITIONER TYPES EXPLAINED ===

1. IID (Independent and Identically Distributed):
   - Each client has the same distribution across all classes
   - All clients have roughly equal amounts of each class
   - Example: Each client has ~10% of each class (0-9)
   - Realistic: NO - Real federated environments are rarely this balanced
   - Use case: Baseline comparisons, initial algorithm testing

2. Dirichlet Partitioner:
   - Each client has different class distributions based on Dirichlet distribution
   - Controlled by alpha (α) parameter for heterogeneity level
   - Example: Client 1 has 80% class 0, Client 2 has 70% class 5, etc.
   - Realistic: YES - Models real-world federated scenarios

   Alpha Parameter Effects:
   - α = ∞     : IID-like (uniform distribution)
   - α = 10.0  : Low heterogeneity (mild imbalance)
   - α = 1.0   : Moderate heterogeneity (noticeable imbalance)
   - α = 0.5   : High heterogeneity (significant imbalance)
   - α = 0.1   : Very high heterogeneity (extreme imbalance)
   - α = 0.01  : Extreme heterogeneity (almost single-class clients)

=== SAMPLE USAGE EXAMPLES ===

# Basic IID visualization (uniform distribution)
python visualize_data_distribution.py --partitioner iid --partitions 4 --output results/

# Moderate heterogeneity (good for general FL research)
python visualize_data_distribution.py --partitioner dirichlet --alpha 1.0 --partitions 5 --output results/

# High heterogeneity (realistic federated scenario)
python visualize_data_distribution.py --partitioner dirichlet --alpha 0.1 --partitions 8 --output results/

# Compare multiple random seeds
python visualize_data_distribution.py --seeds 42 123 456 --alpha 0.5 --partitions 4 --output comparison/

# Generate bar plots instead of both (bar + heatmap)
python visualize_data_distribution.py --plot-type bar --partitioner dirichlet --alpha 0.3 --partitions 6

# Generate only heatmaps
python visualize_data_distribution.py --plot-type heatmap --partitioner dirichlet --alpha 0.3 --partitions 6

# Generate both bar and heatmap (default behavior)
python visualize_data_distribution.py --partitioner dirichlet --alpha 0.3 --partitions 6

# Show detailed analysis in console
python visualize_data_distribution.py --alpha 0.1 --partitions 4 --show-analysis

# Different datasets (all available from Hugging Face Hub)
python visualize_data_distribution.py --dataset "uoft-cs/cifar10" --alpha 0.5 --partitions 10
python visualize_data_distribution.py --dataset "ylecun/mnist" --alpha 0.3 --partitions 8
python visualize_data_distribution.py --dataset "uoft-cs/cifar100" --alpha 0.1 --partitions 12

=== SUPPORTED DATASETS ===

All datasets from Hugging Face Hub are supported! Common examples:
- zalando-datasets/fashion_mnist (default)
- ylecun/mnist
- uoft-cs/cifar10
- uoft-cs/cifar100
- zh-plus/tiny-imagenet
- flwrlabs/femnist
- ufldl-stanford/svhn
- And thousands more at https://huggingface.co/datasets

To test dataset compatibility:
python fl/flower/scripts/test_datasets.py --dataset "your-dataset-name"
python fl/flower/scripts/test_datasets.py --list-common

=== FEDERATED LEARNING IMPLICATIONS ===

IID Environment:
+ Fast convergence (all clients learn similarly)
+ Good generalization performance
+ Easy to achieve consensus
- Unrealistic assumption for real federated settings
- Overestimates algorithm performance

Non-IID Environment (Dirichlet with low α):
+ Realistic simulation of federated environments
+ Better evaluation of algorithm robustness
+ Identifies client drift and convergence issues
- Slower convergence requiring advanced FL algorithms
- May need techniques like FedProx, FedNova, SCAFFOLD
- Communication overhead increases

Recommended α values for research:
- α = 1.0   : Balanced research setting
- α = 0.5   : Challenging but manageable
- α = 0.1   : High difficulty, realistic scenario
- α = 0.01  : Extreme scenario for stress testing

Usage:
    python visualize_data_distribution.py --help
    python visualize_data_distribution.py --partitions 4 --alpha 0.5 --output results/
    python visualize_data_distribution.py --partitioner iid --partitions 10 --plot-type bar
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directories to Python path for imports FIRST
current_dir = Path(__file__).parent
flower_dir = current_dir.parent
fl_dir = flower_dir.parent
sys.path.insert(0, str(fl_dir))

from flower.fed.data.data_loader_config import DataLoaderConfig  # noqa: E402
from flower.fed.util.visualize_data import visualize_data_distribution  # noqa: E402


def parse_arguments() -> argparse.Namespace:
  """Parse command line arguments."""
  parser = argparse.ArgumentParser(
    description="Visualize data distribution across federated partitions", formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  # Dataset configuration
  parser.add_argument("--dataset", type=str, default="zalando-datasets/fashion_mnist", help="Dataset to use for visualization")

  # Partitioner configuration
  parser.add_argument("--partitioner", type=str, choices=["dirichlet", "iid"], default="dirichlet", help="Partitioner type to use")

  parser.add_argument("--partitions", type=int, default=4, help="Number of partitions to create")

  # Alpha parameter controls heterogeneity in Dirichlet partitioner:
  # α = 10.0+: Nearly IID (minimal heterogeneity)
  # α = 1.0:   Moderate heterogeneity (research standard)
  # α = 0.5:   High heterogeneity (challenging scenario)
  # α = 0.1:   Very high heterogeneity (realistic FL)
  # α = 0.01:  Extreme heterogeneity (stress test)
  parser.add_argument("--alpha", type=float, default=1.0, help="Alpha parameter for Dirichlet partitioner (lower = more heterogeneous)")

  # Seed configuration
  parser.add_argument("--seed", type=int, default=42, help="Random seed for single visualization")

  parser.add_argument("--seeds", type=int, nargs="+", help="Multiple seeds for comparison (overrides --seed)")

  # Visualization configuration
  parser.add_argument(
    "--plot-type", type=str, choices=["bar", "heatmap", "both"], default="both", help="Type of plot to generate (both generates bar and heatmap)"
  )

  parser.add_argument("--size-unit", type=str, choices=["absolute", "percent"], default="absolute", help="Unit for displaying data sizes")

  parser.add_argument("--figsize", type=str, default="12,8", help="Figure size as 'width,height' (e.g., '12,8')")

  # Output configuration
  parser.add_argument("--output", type=str, help="Output directory for saving visualizations")

  parser.add_argument("--prefix", type=str, default="data_distribution", help="Prefix for output filenames")

  return parser.parse_args()


def generate_filename(
  base_name: str, partitioner_type: str, num_partitions: int, alpha: Optional[float] = None, plot_type: str = "heatmap", output_dir: Optional[str] = None
) -> str:
  """Generate filename for the visualization output."""
  # Build filename components
  filename_parts = [base_name, partitioner_type, f"partitions{num_partitions}"]

  if alpha is not None and partitioner_type == "dirichlet":
    filename_parts.append(f"alpha{alpha}")

  filename_parts.append(plot_type)

  # Create filename
  filename = "_".join(filename_parts) + ".png"

  # Add output directory if specified
  if output_dir:
    return str(Path(output_dir) / filename)
  else:
    return filename


def main():
  """Main function to run the visualization script.

  This function demonstrates different federated data partitioning scenarios:

  EXAMPLE SCENARIOS:

  1. IID Scenario (Unrealistic but baseline):
     - All clients have balanced class distributions
     - Quick convergence but not realistic
     - Command: --partitioner iid --partitions 4

  2. Moderate Non-IID (Research standard):
     - Some heterogeneity but manageable
     - Good balance between realism and tractability
     - Command: --partitioner dirichlet --alpha 1.0 --partitions 4

  3. High Non-IID (Realistic scenario):
     - Significant data heterogeneity across clients
     - Models real federated environments
     - Command: --partitioner dirichlet --alpha 0.1 --partitions 4

  4. Extreme Non-IID (Stress testing):
     - Each client has very few classes
     - Tests algorithm robustness limits
     - Command: --partitioner dirichlet --alpha 0.01 --partitions 8

  OUTPUT FILES:
  By default (both), the following files are generated:
  Bar plot:
  - *_bar.png: Bar chart showing data distribution per partition
  Heatmap (generates 2 files):
  - *_heatmap_absolute.png: Raw data counts per client/class
  - *_heatmap_percentage.png: Percentage distribution per client
  """
  args = parse_arguments()

  # Parse seeds
  seeds_to_process = args.seeds if args.seeds else [args.seed]

  # Set up output directory
  output_dir = None
  if args.output:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir)

  print("Starting data distribution visualization...")
  print(f"Processing {len(seeds_to_process)} seed(s): {seeds_to_process}")

  for i, seed in enumerate(seeds_to_process, 1):
    print(f"\n{'=' * 60}")
    print(f"Processing seed {seed} ({i}/{len(seeds_to_process)})")
    print(f"{'=' * 60}")

    # Create DataLoader configuration
    # This demonstrates how different partitioner settings affect data distribution:
    #
    # For IID: All clients get roughly equal samples from each class
    # Result: partition_0: [10%, 10%, 10%, ...], partition_1: [10%, 10%, 10%, ...]
    #
    # For Dirichlet α=1.0: Moderate heterogeneity
    # Result: partition_0: [15%, 8%, 12%, ...], partition_1: [5%, 18%, 9%, ...]
    #
    # For Dirichlet α=0.1: High heterogeneity (realistic FL scenario)
    # Result: partition_0: [45%, 2%, 0%, ...], partition_1: [1%, 67%, 3%, ...]
    config = DataLoaderConfig(
      dataset_name=args.dataset,
      partitioner_type=args.partitioner,
      alpha=args.alpha if args.partitioner == "dirichlet" else 1.0,
      seed=seed,
      plot_type=args.plot_type,
      size_unit=args.size_unit,
    )

    print("Configuration:")
    print(f"  Dataset: {config.dataset_name}")
    print(f"  Partitioner: {config.partitioner_type}")
    print(f"  Num partitions: {args.partitions}")
    if config.alpha and args.partitioner == "dirichlet":
      print(f"  Alpha: {config.alpha}")
    print(f"  Seed: {config.seed}")
    print(f"  Size unit: {config.size_unit}")
    if output_dir:
      print(f"  Output dir: {output_dir}")
    print(f"  Plot type: {config.plot_type}")
    print()

    # Generate visualization
    print(f"\nGenerating visualization for seed {seed}...")

    # Determine which plot types to generate
    plot_types_to_generate = []
    if args.plot_type == "both":
      plot_types_to_generate = ["bar", "heatmap"]
    else:
      plot_types_to_generate = [args.plot_type]

    generated_files = []

    for plot_type in plot_types_to_generate:
      # Update config for current plot type
      config.plot_type = plot_type

      # Generate filename for current plot type
      current_filename = None
      if output_dir:
        seed_suffix = f"_seed{seed}" if len(seeds_to_process) > 1 else ""

        # For heatmap, we don't use the base filename since individual files are generated
        if plot_type == "heatmap":
          # Use a base name for heatmap but actual files will be _absolute.png and _percentage.png
          current_filename = generate_filename(
            base_name=args.prefix + seed_suffix,
            partitioner_type=args.partitioner,
            num_partitions=args.partitions,
            alpha=args.alpha if args.partitioner == "dirichlet" else None,
            plot_type=plot_type,
            output_dir=output_dir,
          )
        else:
          current_filename = generate_filename(
            base_name=args.prefix + seed_suffix,
            partitioner_type=args.partitioner,
            num_partitions=args.partitions,
            alpha=args.alpha if args.partitioner == "dirichlet" else None,
            plot_type=plot_type,
            output_dir=output_dir,
          )

      try:
        print(f"  Generating {plot_type} visualization...")
        if current_filename:
          visualize_data_distribution(config, args.partitions, current_filename)

        if current_filename:
          if plot_type == "heatmap":
            # For heatmap, add the actual generated files
            base_path = current_filename.replace(".png", "")
            abs_path = f"{base_path}_absolute.png"
            pct_path = f"{base_path}_percentage.png"
            generated_files.extend([abs_path, pct_path])
            print(f"    Saved heatmap files: {abs_path}, {pct_path}")
          else:
            # For other plot types, add the single file
            generated_files.append(current_filename)
            print(f"    Saved {plot_type}: {current_filename}")

      except Exception as e:
        print(f"    ✗ Error generating {plot_type} visualization: {str(e)}")
        continue

  # Summary
  print(f"\n{'=' * 60}")
  print("All visualizations completed!")
  if output_dir:
    print(f"Output directory: {output_dir}")
  print(f"{'=' * 60}")


if __name__ == "__main__":
  main()
