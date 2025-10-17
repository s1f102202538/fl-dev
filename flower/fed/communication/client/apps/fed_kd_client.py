"""pytorch-example: A Flower / PyTorch app."""

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
    self.local_model_name = "local-model"
    self.global_model_name = "global-model"
    self.public_test_data = public_test_data

  def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
    train_loss = 0.0  # 初期化
    temperature = float(config.get("temperature", 3.0))

    if "avg_logits" in config and config["avg_logits"] is not None:
      # 蒸留には保存されたグローバルモデルを使用
      global_model_for_distillation = load_model_from_state(self.client_state, self.net, self.global_model_name)
      if global_model_for_distillation is not None:
        distillation_model = global_model_for_distillation
        print("[DEBUG] Using global model for knowledge distillation")
      else:
        distillation_model = self.net
        print("[DEBUG] No saved global model found, using current model for distillation")

      logits = base64_to_batch_list(config["avg_logits"])

      distillation = Distillation(
        studentModel=distillation_model,
        public_data=self.public_test_data,
        soft_targets=logits,
      )
      # 知識蒸留の実行してモデルを更新
      self.net = distillation.train_knowledge_distillation(
        epochs=3,
        learning_rate=0.001,  # 蒸留用学習率
        T=temperature,  # サーバーから受信した温度
        alpha=0.7,  # KL蒸留損失の重み
        beta=0.3,  # CE損失の重み
        device=self.device,
      )
      # 蒸留後のモデルをグローバルモデルとして保存
      save_model_to_state(self.net, self.client_state, self.global_model_name)

      print(f"Knowledge distillation completed (temperature: {temperature:.3f})")
    else:
      print("No server logits available, skipping distillation")

    train_loss = CNNTask.train(
      self.net,
      self.train_loader,
      self.local_epochs,
      lr=0.001,
      device=self.device,
    )
    print(f"Client training completed with loss: {train_loss:.4f}")

    # モデル全体のパラメータを state に保存
    save_model_to_state(self.net, self.client_state, self.local_model_name)

    # 学習済みのモデルで公開データの推論を行いロジットを取得
    raw_logits = CNNTask.inference(self.net, self.public_test_data, device=self.device)
    print(f"[DEBUG] Raw logits generated: {len(raw_logits)} batches")
    # ロジットのフィルタリングと較正処理
    filtered_logits = filter_and_calibrate_logits(raw_logits)
    print(f"[DEBUG] Filtered logits: {len(filtered_logits)} batches")

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

    # モデルをロードする
    loaded_model = load_model_from_state(self.client_state, self.net, self.local_model_name)
    if loaded_model is not None:
      self.net = loaded_model
      print("[DEBUG] Model loaded successfully for evaluation")
    else:
      print("[Warning]: No saved model state found, using out-of-the-box model")

    loss, accuracy = CNNTask.test(self.net, self.val_loader, device=self.device)

    return loss, len(self.val_loader.dataset), {"accuracy": accuracy}  # type: ignore
