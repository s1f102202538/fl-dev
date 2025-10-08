"""pytorch-example: A Flower / PyTorch app."""

import numpy as np
import torch
from fed.models.base_model import BaseModel
from fed.task.cnn_task import CNNTask
from fed.util.model_util import get_weights, set_weights
from flwr.client import NumPyClient
from flwr.common import ArrayRecord, RecordDict
from flwr.common.typing import NDArrays, UserConfigValue
from torch.utils.data import DataLoader


# Define Flower Client and client_fn
class FedAvgClient(NumPyClient):
  """A simple client that showcases how to use the state.

  It implements a basic version of `personalization` by which
  the classification layer of the CNN is stored locally and used
  and updated during `fit()` and used during `evaluate()`.
  """

  def __init__(self, net: BaseModel, client_state: RecordDict, train_loader: DataLoader, val_loader: DataLoader, local_epochs: UserConfigValue) -> None:
    self.net = net
    self.client_state = client_state
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.local_epochs: int = int(local_epochs)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.net.to(self.device)
    self.local_layer_name = "classification-head"

  def fit(self, parameters: NDArrays, config: dict) -> tuple[NDArrays, int, dict]:
    """Train model locally.

    The client stores in its context the parameters of the last layer in the model
    (i.e. the classification head). The classifier is saved at the end of the
    training and used the next time this client participates.
    """

    # Apply weights from global models (the whole model is replaced)
    set_weights(self.net, parameters)

    # Override weights in classification layer with those this client
    # had at the end of the last fit() round it participated in
    self._load_layer_weights_from_state()

    train_loss = CNNTask.train(
      self.net,
      self.train_loader,
      self.local_epochs,
      lr=float(config["lr"]),
      device=self.device,
    )
    # Save classification head to context's state to use in a future fit() call
    self._save_layer_weights_to_state()

    # Return locally-trained model and metrics
    return (
      get_weights(self.net),
      len(self.train_loader.dataset),  # type: ignore
      {"train_loss": train_loss},
    )

  def _save_layer_weights_to_state(self) -> None:
    """Save last layer weights to state."""
    arr_record = ArrayRecord(self.net.fc2.state_dict())  # type: ignore

    # Add to RecordDict (replace if already exists)
    self.client_state[self.local_layer_name] = arr_record

  def _load_layer_weights_from_state(self) -> None:
    """Load last layer weights to state."""
    if self.local_layer_name not in self.client_state.array_records:
      return

    state_dict = self.client_state[self.local_layer_name].to_torch_state_dict()  # type: ignore

    # apply previously saved classification head by this client
    self.net.fc2.load_state_dict(state_dict, strict=True)  # type: ignore

  def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple[float, int, dict]:
    """Evaluate the global model on the local validation set.

    Note the classification head is replaced with the weights this client had the
    last time it trained the model.
    """
    set_weights(self.net, parameters)
    # Override weights in classification layer with those this client
    # had at the end of the last fit() round it participated in
    self._load_layer_weights_from_state()
    loss, accuracy = CNNTask.test(self.net, self.val_loader, self.device)
    return loss, len(self.val_loader.dataset), {"accuracy": accuracy}  # type: ignore
