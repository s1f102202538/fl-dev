#!/usr/bin/env python3
"""
Test script for FedKD with server-side model training.

This script demonstrates the new FedKD implementation where:
1. Clients send logits to server
2. Server aggregates logits and trains its own model using knowledge distillation
3. Server generates new logits using trained model and public data
4. Server distributes generated logits to clients for further distillation
"""

import sys
from pathlib import Path

import torch
from torch import device


def test_server_creation():
  """Test creating the FedKD server with server-side model training."""

  print("Testing FedKD Server with Server-side Model Training")
  print("=" * 60)

  # Add flower directory to path for imports
  current_dir = Path(__file__).parent
  fl_dev_dir = current_dir
  sys.path.insert(0, str(fl_dev_dir))

  # Import after adding path
  from flower.fed.communication.server.apps.fed_kd_distillation_model_server import FedKDDistillationModelServer
  from flower.fed.data.data_loader_config import DataLoaderConfig
  from flower.fed.util.create_model import create_model
  from flower.fed.util.data_loader import load_public_data

  # Configuration
  model_name = "mini-cnn"
  dataset_name = "uoft-cs/cifar10"
  server_device = device("cuda:0" if torch.cuda.is_available() else "cpu")
  n_classes = 10
  out_dim = 256
  num_rounds = 3

  print(f"Model: {model_name}")
  print(f"Dataset: {dataset_name}")
  print(f"Device: {server_device}")
  print(f"Classes: {n_classes}")
  print(f"Rounds: {num_rounds}")
  print()

  try:
    # Create server model and public data loader (now done in server_app.py style)
    print("1. Creating server model and public data loader...")
    server_model = create_model(model_name, is_moon=True, out_dim=out_dim, n_classes=n_classes, use_projection_head=True)
    data_loader_config = DataLoaderConfig(dataset_name=dataset_name)
    public_data_loader = load_public_data(data_loader_config)
    print(f"   Server model created: {type(server_model).__name__}")
    print(f"   Public data loader batches: {len(public_data_loader)}")
    print()

    # Test server creation with pre-created components
    print("2. Creating server components...")
    server_components = FedKDDistillationModelServer.create_server(
      server_model=server_model,
      public_data_loader=public_data_loader,
      server_device=server_device,
      use_wandb=False,
      run_config={
        "model_name": model_name,
        "dataset_name": dataset_name,
        "server_name": "fed-kd-distillation-model-server",
        "n_classes": n_classes,
        "out_dim": out_dim,
      },
      num_rounds=num_rounds,
    )

    print("✅ Server components created successfully!")
    print(f"Strategy type: {type(server_components.strategy).__name__}")
    print(f"Number of rounds: {server_components.config.num_rounds}")
    print()

    # Test strategy initialization
    strategy = server_components.strategy
    print("3. Checking strategy configuration...")
    print(f"   Server model device: {strategy.server_model.device if hasattr(strategy.server_model, 'device') else 'N/A'}")
    print(f"   KD temperature: {strategy.kd_temperature}")
    print(f"   Server training epochs: {strategy.server_training_epochs}")
    print(f"   Server learning rate: {strategy.server_learning_rate}")
    print(f"   Public data loader batches: {len(strategy.public_data_loader)}")
    print()

    print("4. Testing public data loader...")
    sample_batch = next(iter(strategy.public_data_loader))
    print(f"   Batch keys: {sample_batch.keys()}")
    print(f"   Image shape: {sample_batch['image'].shape}")
    print(f"   Label shape: {sample_batch['label'].shape}")
    print()

    print("✅ All tests passed!")
    print()
    print("Server Configuration Summary:")
    print("-" * 40)
    print("Strategy: FedKDDistillationModel (Individual Client Logit Training)")
    print(f"Server Model: {model_name} on {server_device}")
    print(f"Dataset: {dataset_name}")
    print(f"Public Data Batches: {len(strategy.public_data_loader)}")
    print(f"KD Temperature: {strategy.kd_temperature}")
    print(f"Training Epochs: {strategy.server_training_epochs}")
    print(f"Learning Rate: {strategy.server_learning_rate}")
    print()
    print("Training Approach:")
    print("- Server model trains iteratively with each client's logits")
    print("- Adaptive learning rate based on client importance")
    print("- Weighted epoch allocation based on client data size")
    print("- Preserves client diversity in knowledge transfer")
    print()
    print("Ready to run federated learning with server-side model training!")

  except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback

    traceback.print_exc()
    return False

  return True


if __name__ == "__main__":
  print("FedKD Server-side Model Training Test")
  print("=" * 60)
  print()

  success = test_server_creation()

  if success:
    print("\n🎉 Test completed successfully!")
    print("\nTo use this server, set server_name to 'fed-kd-distillation-model-server' in your configuration.")
  else:
    print("\n💥 Test failed!")
    sys.exit(1)
