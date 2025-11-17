#!/usr/bin/env python3

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
  parser.add_argument("--dataset", type=str, default="uoft-cs/cifar10", help="Dataset to use for visualization")

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
          visualize_data_distribution(config, current_filename)

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
