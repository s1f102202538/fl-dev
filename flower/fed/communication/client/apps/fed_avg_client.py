import numpy as np
import torch
from fed.models.base_model import BaseModel
from fed.task.cnn_task import CNNTask
from fed.util.model_util import get_weights, set_weights
from flwr.client import NumPyClient
from flwr.common import RecordDict
from flwr.common.typing import NDArrays, UserConfigValue
from torch.utils.data import DataLoader


# Define Flower Client and client_fn
class FedAvgClient(NumPyClient):
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
    # Apply weights from global models (the whole model is replaced)
    set_weights(self.net, parameters)

    train_loss = CNNTask.train(
      self.net,
      self.train_loader,
      self.local_epochs,
      lr=0.01,
      device=self.device,
    )

    # Return locally-trained model and metrics
    return (
      get_weights(self.net),
      len(self.train_loader.dataset),  # type: ignore
      {"train_loss": train_loss},
    )

  def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple[float, int, dict]:
    set_weights(self.net, parameters)
    loss, accuracy = CNNTask.test(self.net, self.val_loader, self.device)
    return loss, len(self.val_loader.dataset), {"accuracy": accuracy}  # type: ignore
