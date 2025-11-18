"""FedMoon with Logit Sharing: Flower / PyTorch app"""

import copy
from typing import Dict, Tuple

import torch
from fed.algorithms.old_distillation import OldDistillation
from fed.algorithms.old_moon import OldMoonContrastiveLearning, OldMoonTrainer
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


class OldFedMoonClient(NumPyClient):
  """ロジット共有機能を持つFedMoonクライアント"""

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

    # モデル状態保存用の名前定義
    self.local_model_name = "local-model"
    self.global_model_name = "global-model"

    # Moon対比学習の初期化
    self.moon_learner = OldMoonContrastiveLearning(
      mu=1.0,  # 1.0
      temperature=0.5,
      device=self.device,
    )

    # Moonトレーナーの初期化
    self.moon_trainer = OldMoonTrainer(
      moon_learner=self.moon_learner,
      device=self.device,
    )

  def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
    """拡張FedMoon対比学習と適応ロジット共有によるローカルモデル訓練"""
    temperature = float(config.get("temperature", 3.0))

    previous_round_model = load_model_from_state(self.client_state, self.net, self.local_model_name)
    # 現在のローカルモデル状態を復元
    if previous_round_model is not None:
      print("[DEBUG] Previous round model loaded successfully")
    else:
      print("[DEBUG] No previous model found, using initial model")

    if previous_round_model is None:
      print("[INFO] First round: Performing normal training without Moon contrastive learning")
      train_loss = CNNTask.train(
        net=self.net,
        train_loader=self.train_loader,
        epochs=self.local_epochs,
        lr=0.001,
        device=self.device,
      )
    else:
      if "avg_logits" in config and config["avg_logits"] is not None:
        # 蒸留には保存されたグローバルモデルを使用
        global_model_for_distillation = load_model_from_state(self.client_state, self.net, self.global_model_name)
        if global_model_for_distillation is not None:
          distillation_base_model = global_model_for_distillation
          print("[DEBUG] Using saved global model for knowledge distillation")
        else:
          distillation_base_model = copy.deepcopy(self.net)
          print("[DEBUG] No saved global model found, using current model for distillation")

        logits = base64_to_batch_list(config["avg_logits"])

        # 蒸留により仮想グローバルモデルを直接作成
        distillation = OldDistillation(
          studentModel=distillation_base_model,  # グローバルモデルまたは現在のモデルを使用
          public_data=self.public_test_data,
          soft_targets=logits,
        )

        # FedKD論文に基づく知識蒸留パラメータで仮想グローバルモデルを作成
        virtual_global_model = distillation.train_knowledge_distillation(
          epochs=3,
          learning_rate=0.001,
          T=temperature,
          alpha=0.7,  # FedKD論文: KL蒸留損失の重み
          beta=0.3,  # FedKD論文: CE損失の重み
          device=self.device,
        )
        virtual_global_model.to(self.device)

        # グローバルモデル を moon 学習の起点にする
        self.net = virtual_global_model

        # 蒸留後のモデルをグローバルモデルとして保存
        save_model_to_state(virtual_global_model, self.client_state, self.global_model_name)
        print("[DEBUG] Distilled model saved as global model")

        self.moon_learner.update_models(previous_round_model, virtual_global_model)
        print("Updated Moon learner with previous round model and virtual global model")

      # グローバルモデル を moon 学習の起点にする
      train_loss = self.moon_trainer.train_with_enhanced_moon(
        model=self.net,
        train_loader=self.train_loader,
        lr=0.001,
        epochs=self.local_epochs,
      )

    # 学習完了後のローカルモデル状態を保存
    save_model_to_state(self.net, self.client_state, self.local_model_name)

    raw_logits = CNNTask.inference(self.net, self.public_test_data, device=self.device)
    print(f"[DEBUG] Raw logits generated: {len(raw_logits)} batches")
    # ロジットのフィルタリングと較正処理
    filtered_logits = filter_and_calibrate_logits(raw_logits)
    print(f"[DEBUG] Filtered logits: {len(filtered_logits)} batches")

    print(f"Client training loss: {train_loss:.4f}")

    return (
      [],  # ロジット共有のみでパラメータ集約は行わないため空リストを返す
      len(self.train_loader.dataset),  # type: ignore
      {
        "train_loss": train_loss,
        "logits": batch_list_to_base64(filtered_logits),
      },
    )

  def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
    """性能追跡による拡張モデル評価"""

    # モデルをロードする
    loaded_model = load_model_from_state(self.client_state, self.net, self.local_model_name)
    if loaded_model is not None:
      self.net = loaded_model
    else:
      print("[Warning] No saved model state found, using initial model")

    loss, accuracy = CNNTask.test(self.net, self.val_loader, device=self.device)

    return (
      loss,
      len(self.val_loader.dataset),  # type: ignore
      {"accuracy": accuracy},
    )
