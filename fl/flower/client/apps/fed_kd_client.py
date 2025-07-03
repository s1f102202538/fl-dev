"""pytorch-example: A Flower / PyTorch app."""

from typing import Dict, Tuple

import torch
from flower.common.models.mini_cnn import MiniCNN
from flower.common.task.cnn_task import CNNTask
from flower.common.util.util import base64_to_tensor, tensor_to_base64
from flwr.client import NumPyClient
from flwr.client.client import Client
from flwr.common import ArrayRecord, Context, RecordDict
from flwr.common.typing import NDArrays, UserConfigValue
from torch.utils.data import DataLoader

from fl.flower.common.dataLoader.data_loader import load_data, load_public_data
from fl.flower.common.task.distillation import Distillation


class FedKDClient(NumPyClient):
  def __init__(
    self,
    net: MiniCNN,
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
    self.local_layer_name = "classification-head"
    self.public_test_data = public_test_data

  def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
    # Config から学習率と共有ロジットの取得
    lr = float(config["lr"])
    # ロジットをbase64からテンソルに変換
    logits = base64_to_tensor(config["avg_logits"])

    # 共有ロジットを使用して知識蒸留を行う
    distillation = Distillation(
      studentModel=self.net,
      train_data=self.public_test_data,
      soft_target_losses=logits,
    )
    # 知識蒸留の実行してモデルを更新
    self.net = distillation.train_knowledge_distillation(
      5,
      learning_rate=0.01,
      T=2.0,
      soft_target_loss_weight=0.5,
      ce_loss_weight=0.5,
      device=self.device,
    )

    # 分類層のパラメータを復元
    self._load_layer_weights_from_state()

    train_loss = CNNTask.train(
      self.net,
      self.train_loader,
      self.local_epochs,
      lr=lr,
      device=self.device,
    )

    # 分類層のパラメータを state に保存
    self._save_layer_weights_to_state()

    # 学習済みのモデルで公開データの推論を行いロジットを取得
    logit = CNNTask.inference(self.net, self.public_test_data, device=self.device)

    return (
      [],  # モデルの集約は行わないため空リストを返す
      len(self.train_loader.dataset),  # type: ignore
      {
        "train_loss": train_loss,
        "logits": tensor_to_base64(logit.cpu().numpy().tolist()),  # type: ignore
      },
    )

  def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
    """Evaluate model locally."""
    # ロジットをbase64からテンソルに変換
    logits = base64_to_tensor(config["avg_logits"])

    # 知識蒸留を行う
    distillation = Distillation(
      studentModel=self.net,
      train_data=self.public_test_data,
      soft_target_losses=logits,
    )
    self.net = distillation.train_knowledge_distillation(
      5,
      learning_rate=0.01,
      T=2.0,
      soft_target_loss_weight=0.5,
      ce_loss_weight=0.5,
      device=self.device,
    )

    # Override weights in classification layer with those this client
    # had at the end of the last fit() round it participated in
    self._load_layer_weights_from_state()

    loss, accuracy = CNNTask.test(self.net, self.val_loader, device=self.device)

    return loss, len(self.val_loader.dataset), {"accuracy": accuracy}  # type: ignore

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

  @staticmethod
  def client_fn(context: Context) -> Client:
    # Load model and data
    net = MiniCNN()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_loader, val_loader = load_data(partition_id, num_partitions)
    public_test_data = load_public_data(batch_size=32)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    # We pass the state to persist information across
    # participation rounds. Note that each client always
    # receives the same Context instance (it's a 1:1 mapping)
    client_state = context.state
    return FedKDClient(net, client_state, train_loader, val_loader, public_test_data, local_epochs).to_client()
