"""pytorch-example: A Flower / PyTorch app."""

from ast import Tuple
from typing import Dict

import numpy as np
import torch
from flower.common.models.mini_cnn import MiniCNN
from flower.common.task.cnn_task import CNNTask
from flower.common.util.util import base64_to_tensor, get_weights, set_weights
from flwr.client import NumPyClient
from flwr.client.client import Client
from flwr.common import ArrayRecord, Context, RecordDict
from flwr.common.typing import NDArrays, UserConfigValue
from torch.utils.data import DataLoader

from fl.flower.common.dataLoader.data_loader import load_data
from fl.flower.common.task import distillation
from fl.flower.common.task.distillation import Distillation


class FedKDClient(NumPyClient):
  def __init__(
    self,
    net: MiniCNN,
    client_state: RecordDict,
    train_loader: DataLoader,
    val_loader: DataLoader,
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
    self.local_layer_name = "classification-head"

  def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
    # Config から学習率と共有ロジットの取得
    lr = float(config["lr"])
    # ロジットをbase64からテンソルに変換
    logits = base64_to_tensor(config["avg_logits"])

    # 共有ロジットを使用して知識蒸留を行う
    distillation = Distillation(
      studentModel=self.net,
      train_data=self.train_loader,
      soft_target_losses=logits,
    )
    distillation.train_knowledge_distillation()
