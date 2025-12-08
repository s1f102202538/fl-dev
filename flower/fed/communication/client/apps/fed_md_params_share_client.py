"""FedKD with Parameter Sharing: Flower / PyTorch app - Clients return logits"""

from typing import Dict, Tuple

import torch
from fed.models.base_model import BaseModel
from fed.task.cnn_task import CNNTask
from fed.util.model_util import batch_list_to_base64, set_weights
from flwr.client import NumPyClient
from flwr.common import RecordDict
from flwr.common.typing import NDArrays, UserConfigValue
from torch.utils.data import DataLoader


class FedMdParamsShareClient(NumPyClient):
  """FedMD client that receives parameters and returns logits.

  This client:
  1. Receives model parameters from server
  2. Trains the model locally
  3. Generates logits from trained model
  4. Sends logits back to server
  """

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

  def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
    """FedKD client training: receive parameters, train, return logits."""

    # Apply server parameters to local model
    if parameters is not None and len(parameters) > 0:
      print("[INFO] Applying server parameters to local model")
      set_weights(self.net, parameters)
    else:
      print("[INFO] No server parameters provided, using current model state")

    # Perform local training
    train_loss = self._perform_local_training()
    print(f"Client training loss: {train_loss:.4f}")

    # Generate logits from trained model
    logits = self._generate_logits()
    logits_base64 = batch_list_to_base64(logits)
    print(f"[INFO] Generated {len(logits)} logit batches")

    # Return empty parameters (we're sending logits instead)
    return (
      [],
      len(self.train_loader.dataset),  # type: ignore
      {
        "train_loss": train_loss,
        "logits": logits_base64,  # Send logits to server
      },
    )

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

  def _generate_logits(self) -> list:
    """Generate logits from the trained model."""
    logits = CNNTask.inference(self.net, self.public_test_data, device=self.device)
    print(f"[DEBUG] Generated logits from {len(logits)} batches")
    return logits

  def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
    """Evaluate model performance using server-provided parameters."""
    # Apply server parameters for evaluation
    if parameters is not None and len(parameters) > 0:
      print("[DEBUG] Applying server model parameters for evaluation")
      set_weights(self.net, parameters)

    loss, accuracy = CNNTask.test(self.net, self.val_loader, self.device)
    return loss, len(self.val_loader.dataset), {"accuracy": accuracy}  # type: ignore
