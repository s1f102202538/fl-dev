"""FedKD with Logit Sharing: Flower / PyTorch app"""

from typing import Dict, Tuple

import torch
from fed.algorithms.distillation import Distillation
from fed.models.base_model import BaseModel
from fed.task.cnn_task import CNNTask
from fed.util.model_util import (
  base64_to_batch_list,
  batch_list_to_base64,
  filter_and_calibrate_logits,
  load_model_from_state,
  save_model_to_state,
)
from flwr.client import NumPyClient
from flwr.common import RecordDict
from flwr.common.typing import NDArrays, UserConfigValue
from torch.utils.data import DataLoader


class FedKdClient(NumPyClient):
  """FedKD client with knowledge distillation and logit sharing capabilities."""

  def __init__(
    self,
    net: BaseModel,
    client_state: RecordDict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    public_test_data: DataLoader,
    local_epochs: UserConfigValue,
  ) -> None:
    super().__init__()
    self.net = net
    self.client_state = client_state
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.local_epochs: int = int(local_epochs)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.net.to(self.device)
    self.public_test_data = public_test_data

    # Model state storage keys
    self.local_model_name = "local-model"
    self.global_model_name = "global-model"

  def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
    """FedKD client training with knowledge distillation and logit sharing."""
    temperature = float(config.get("temperature", 3.0))

    # Perform knowledge distillation if server logits are available
    if "avg_logits" in config and config["avg_logits"] is not None:
      self._perform_knowledge_distillation(config["avg_logits"], temperature)
    else:
      print("[INFO] No server logits available, skipping distillation")

    # Perform local training
    train_loss = self._perform_local_training()

    # Save trained model and generate logits for sharing
    save_model_to_state(self.net, self.client_state, self.local_model_name)
    filtered_logits = self._generate_and_filter_logits()

    print(f"Client training loss: {train_loss:.4f}")

    return (
      [],  # Empty list for logit-only sharing (no parameter aggregation)
      len(self.train_loader.dataset),  # type: ignore
      {
        "train_loss": train_loss,
        "logits": batch_list_to_base64(filtered_logits),
      },
    )

  def _perform_knowledge_distillation(self, avg_logits: str, temperature: float) -> None:
    """Perform knowledge distillation using server-aggregated logits."""

    # Use saved global model for distillation if available
    global_model_for_distillation = load_model_from_state(self.client_state, self.net, self.global_model_name)
    if global_model_for_distillation is not None:
      distillation_model = global_model_for_distillation
      print("[DEBUG] Using saved global model for knowledge distillation")
    else:
      distillation_model = self.net
      print("[DEBUG] No saved global model found, using current model for distillation")

    # Convert base64 logits to tensor batches
    logits = base64_to_batch_list(avg_logits)

    # Perform knowledge distillation
    distillation = Distillation(
      studentModel=distillation_model,
      public_data=self.public_test_data,
      soft_targets=logits,
    )

    # Train model with knowledge distillation
    self.net = distillation.train_knowledge_distillation(
      epochs=3,
      learning_rate=0.01,
      T=temperature,  # Server-provided temperature
      alpha=0.7,  # KL distillation loss weight
      beta=0.3,  # CE loss weight
      device=self.device,
    )

    # Save distilled model as global model
    save_model_to_state(self.net, self.client_state, self.global_model_name)
    print(f"[DEBUG] Knowledge distillation completed (temperature: {temperature:.3f})")

  def _perform_local_training(self) -> float:
    """Perform local training on the current model."""
    train_loss = CNNTask.train(
      net=self.net,
      train_loader=self.train_loader,
      epochs=self.local_epochs,
      lr=0.01,
      device=self.device,
    )
    print(f"[DEBUG] Local training completed with loss: {train_loss:.4f}")
    return train_loss

  def _generate_and_filter_logits(self) -> list:
    """Generate and filter logits for sharing with server."""

    # Generate raw logits using trained model
    raw_logits = CNNTask.inference(self.net, self.public_test_data, device=self.device)
    print(f"[DEBUG] Raw logits generated: {len(raw_logits)} batches")

    # Filter and calibrate logits
    filtered_logits = filter_and_calibrate_logits(raw_logits)
    print(f"[DEBUG] Filtered logits: {len(filtered_logits)} batches")

    return filtered_logits

  def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
    """Evaluate model performance on validation data."""
    # Load the trained model
    loaded_model = load_model_from_state(self.client_state, self.net, self.local_model_name)
    if loaded_model is not None:
      self.net = loaded_model
      print("[DEBUG] Model loaded successfully for evaluation")
    else:
      print("[WARNING] No saved model state found, using initial model")

    loss, accuracy = CNNTask.test(self.net, self.val_loader, device=self.device)

    return (
      loss,
      len(self.val_loader.dataset),  # type: ignore
      {"accuracy": accuracy},
    )
