"""pytorch-example: A Flower / PyTorch app."""

from typing import Dict, Tuple

import torch
from flower.common.dataLoader.data_loader import load_data, load_public_data
from flower.common.models.mini_cnn import MiniCNN
from flower.common.task.cnn_task import CNNTask
from flower.common.task.distillation import Distillation
from flower.common.util.util import base64_to_batch_list, batch_list_to_base64
from flwr.client import NumPyClient
from flwr.client.client import Client
from flwr.common import ArrayRecord, Context, RecordDict
from flwr.common.typing import NDArrays, UserConfigValue
from torch.utils.data import DataLoader


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
    self.local_model_name = "full-model"
    self.public_test_data = public_test_data

  def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
    # Config から学習率と共有ロジットの取得
    lr = float(config["lr"])

    # 初回ラウンドでは avg_logits がない場合があるのでチェック
    if "avg_logits" in config and config["avg_logits"] is not None:
      # ロジットをbase64からバッチリストに変換
      logits = base64_to_batch_list(config["avg_logits"])

      # サーバーから送信された温度パラメータを取得（デフォルトは3.0）
      temperature = float(config.get("temperature", 3.0))

      # 共有ロジットを使用して知識蒸留を行う
      distillation = Distillation(
        studentModel=self.net,
        public_data=self.public_test_data,
        soft_targets=logits,
      )
      # 知識蒸留の実行してモデルを更新（動的温度を使用）
      self.net = distillation.train_knowledge_distillation(
        epochs=2,  # 蒸留エポック数を増加
        learning_rate=0.005,  # 蒸留用学習率を増加
        T=temperature,  # サーバーから受信した温度を使用
        soft_target_loss_weight=0.5,  # 蒸留損失の重みを増加
        ce_loss_weight=0.5,
        device=self.device,
      )
      print(f"Knowledge distillation performed with server logits (temperature: {temperature:.3f})")

      # 蒸留後のモデル状態を保存
      self._save_model_weights_to_state()
    else:
      print("No server logits available, skipping knowledge distillation")

    train_loss = CNNTask.train(
      self.net,
      self.train_loader,
      self.local_epochs,
      lr=lr,
      device=self.device,
    )

    # モデル全体のパラメータを state に保存
    self._save_model_weights_to_state()

    # 学習済みのモデルで公開データの推論を行いロジットを取得
    logit_batches = CNNTask.inference(self.net, self.public_test_data, device=self.device)

    # NaN/Infのチェックと修正
    filtered_logits = []
    for batch in logit_batches:
      if torch.isnan(batch).any() or torch.isinf(batch).any():
        print("WARNING: Client detected NaN/Inf in logits, replacing with zeros")
        batch = torch.zeros_like(batch)
      filtered_logits.append(batch)

    print("Client send logits stats:", [b.mean().item() for b in filtered_logits])
    print(f"Client training loss: {train_loss:.4f}")

    return (
      [],  # モデルの集約は行わないため空リストを返す
      len(self.train_loader.dataset),  # type: ignore
      {
        "train_loss": train_loss,
        "logits": batch_list_to_base64(filtered_logits),  # type: ignore
      },
    )

  def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
    """Evaluate model locally."""
    # # 初回ラウンドでは avg_logits がない場合があるのでチェック
    # if "avg_logits" in config and config["avg_logits"] is not None:
    #   # ロジットをbase64からバッチリストに変換
    #   logits = base64_to_batch_list(config["avg_logits"])

    #   # 知識蒸留を行う
    #   distillation = Distillation(
    #     studentModel=self.net,
    #     public_data=self.public_test_data,
    #     soft_targets=logits,
    #   )
    #   # 知識蒸留の実行してモデルを更新
    #   self.net = distillation.train_knowledge_distillation(
    #     2,
    #     learning_rate=0.001,
    #     T=3.0,
    #     soft_target_loss_weight=0.3,
    #     ce_loss_weight=0.7,
    #     device=self.device,
    #   )
    #   print("Lightweight knowledge distillation performed with server logits in evaluate")
    # else:
    #   print("No server logits available for evaluation, skipping knowledge distillation")

    # # Override weights in model with those this client
    # # had at the end of the last fit() round it participated in
    # self._load_model_weights_from_state()

    # モデルをロードする
    self._load_model_weights_from_state()

    loss, accuracy = CNNTask.test(self.net, self.val_loader, device=self.device)

    return loss, len(self.val_loader.dataset), {"accuracy": accuracy}  # type: ignore

  def _save_model_weights_to_state(self) -> None:
    """Save entire model weights to state."""
    arr_record = ArrayRecord(self.net.state_dict())  # type: ignore

    # Add to RecordDict (replace if already exists)
    self.client_state[self.local_model_name] = arr_record

  def _load_model_weights_from_state(self) -> None:
    """Load entire model weights from state."""
    if self.local_model_name not in self.client_state.array_records:
      return

    state_dict = self.client_state[self.local_model_name].to_torch_state_dict()  # type: ignore

    # apply previously saved model weights by this client
    self.net.load_state_dict(state_dict, strict=True)  # type: ignore

  @staticmethod
  def client_fn(context: Context) -> Client:
    # Load model and data
    net = MiniCNN()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_loader, val_loader = load_data(partition_id, num_partitions)
    public_test_data = load_public_data(batch_size=32, max_samples=500)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    # We pass the state to persist information across
    # participation rounds. Note that each client always
    # receives the same Context instance (it's a 1:1 mapping)
    client_state = context.state
    return FedKDClient(net, client_state, train_loader, val_loader, public_test_data, local_epochs).to_client()
