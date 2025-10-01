"""pytorch-example: A Flower / PyTorch app."""

from typing import Dict, Tuple

import torch
from flwr.client import NumPyClient
from flwr.client.client import Client
from flwr.common import Context, RecordDict
from flwr.common.typing import NDArrays, UserConfigValue
from torch.utils.data import DataLoader

from flower.common.dataLoader.data_loader import load_data, load_public_data
from flower.common.models.mini_cnn import MiniCNN
from flower.common.task.cnn_task import CNNTask
from flower.common.task.distillation import Distillation
from flower.common.util.util import (
  base64_to_batch_list,
  batch_list_to_base64,
  filter_and_calibrate_logits,
  load_model_from_state,
  save_model_to_state,
)


class FedKdClient(NumPyClient):
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
    # 前のラウンドで保存されたモデルを self.net にロードする
    loaded_model = load_model_from_state(self.client_state, self.net, self.local_model_name)
    if loaded_model is not None:
      self.net = loaded_model
      print("[DEBUG] Previous round model loaded successfully")
    else:
      print("[DEBUG] No previous model found, using initial model")

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
      # 知識蒸留の実行してモデルを更新
      self.net = distillation.train_knowledge_distillation(
        epochs=1,  # 蒸留エポック数を1
        learning_rate=0.001,  # 蒸留用学習率
        T=temperature,  # サーバーから受信した温度
        soft_target_loss_weight=0.4,  # 蒸留損失の重み
        ce_loss_weight=0.6,  # CE損失の重み
        device=self.device,
      )
      print(f"Knowledge distillation performed with server logits (temperature: {temperature:.3f})")

      # 蒸留後のモデル状態を保存
      save_model_to_state(self.net, self.client_state, self.local_model_name)
    else:
      print("No server logits available, skipping knowledge distillation")

    train_loss = CNNTask.train(
      self.net,
      self.train_loader,
      self.local_epochs,
      lr=0.01,
      device=self.device,
    )

    # モデル全体のパラメータを state に保存
    save_model_to_state(self.net, self.client_state, self.local_model_name)

    # 学習済みのモデルで公開データの推論を行いロジットを取得
    raw_logits = CNNTask.inference(self.net, self.public_test_data, device=self.device)
    print(f"[DEBUG] Raw logits generated: {len(raw_logits)} batches")

    # ロジットのフィルタリングと較正処理
    filtered_logits = filter_and_calibrate_logits(raw_logits, temperature=1.5)
    print(f"[DEBUG] Filtered logits: {len(filtered_logits)} batches")

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
    loaded_model = load_model_from_state(self.client_state, self.net, self.local_model_name)
    if loaded_model is not None:
      self.net = loaded_model
      print("[DEBUG] Model loaded successfully for evaluation")
    else:
      print("警告: 保存されたモデル状態が見つからないため、初期状態のモデルを使用します")

    loss, accuracy = CNNTask.test(self.net, self.val_loader, device=self.device)

    return loss, len(self.val_loader.dataset), {"accuracy": accuracy}  # type: ignore

  @staticmethod
  def client_fn(context: Context) -> Client:
    # Load model and data
    net = MiniCNN()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_loader, val_loader = load_data(partition_id, num_partitions)
    public_test_data = load_public_data(batch_size=32, max_samples=1000)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    # We pass the state to persist information across
    # participation rounds. Note that each client always
    # receives the same Context instance (it's a 1:1 mapping)
    client_state = context.state
    return FedKdClient(net, client_state, train_loader, val_loader, public_test_data, local_epochs).to_client()
