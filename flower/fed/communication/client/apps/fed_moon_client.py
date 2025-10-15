"""FedMoon with Logit Sharing: Flower / PyTorch app"""

import copy
from typing import Dict, Tuple

import torch
from fed.algorithms.distillation import Distillation
from fed.algorithms.moon import MoonContrastiveLearning, MoonTrainer
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


class FedMoonClient(NumPyClient):
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
    self.local_model_name = "fed-moon-model"

    # Moon対比学習の初期化
    self.moon_learner = MoonContrastiveLearning(
      mu=1.0,
      temperature=0.5,
      device=self.device,
    )

    # Moonトレーナーの初期化
    self.moon_trainer = MoonTrainer(
      moon_learner=self.moon_learner,
      device=self.device,
    )

  def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
    """拡張FedMoon対比学習と適応ロジット共有によるローカルモデル訓練"""
    current_round = int(config.get("current_round", 0))
    temperature = float(config.get("temperature", 3.0))

    previous_round_model = load_model_from_state(self.client_state, self.net, self.local_model_name)
    # 現在のローカルモデル状態を復元
    if previous_round_model is not None:
      self.net = previous_round_model
      print("[DEBUG] Previous round model loaded successfully")
    else:
      print("[DEBUG] No previous model found, using initial model")

    if previous_round_model is None:
      print("[INFO] First round: Performing normal training without Moon contrastive learning")
      train_loss = CNNTask.train(
        net=self.net,
        train_loader=self.train_loader,
        epochs=self.local_epochs,
        lr=0.01,
        device=self.device,
      )
    else:
      if "avg_logits" in config and config["avg_logits"] is not None:
        logits = base64_to_batch_list(config["avg_logits"])

        # 蒸留により仮想グローバルモデルを直接作成
        distillation = Distillation(
          studentModel=copy.deepcopy(self.net),  # MoonModelを直接使用
          public_data=self.public_test_data,
          soft_targets=logits,
        )

        # MoonModelで知識蒸留を実行
        virtual_global_model = distillation.train_knowledge_distillation(
          epochs=3,
          learning_rate=0.001,
          T=temperature,
          alpha=0.9,  # KL蒸留損失の重み
          beta=0.1,  # CE損失の重み
          device=self.device,
        )
        virtual_global_model.to(self.device)

        self.moon_learner.update_models(previous_round_model, virtual_global_model)
        print("Updated Moon learner with previous round model and virtual global model")

      train_loss = self.moon_trainer.train_with_enhanced_moon(
        model=self.net,
        train_loader=self.train_loader,
        lr=0.01,
        epochs=self.local_epochs,
        current_round=current_round,
      )

    # MoonModel専用のinferenceメソッドを使用
    raw_logits = CNNTask.inference(self.net, self.public_test_data, device=self.device)
    logit_batches = filter_and_calibrate_logits(raw_logits, temperature=temperature)

    # 学習完了後のローカルモデル状態を保存
    save_model_to_state(self.net, self.client_state, self.local_model_name)

    print(f"FedMoon enhanced client completed (Round {current_round})")
    print("Client sending logit statistics:", [b.mean().item() for b in logit_batches])

    return (
      [],  # ロジット共有のみでパラメータ集約は行わないため空リストを返す
      len(self.train_loader.dataset),  # type: ignore
      {
        "train_loss": train_loss,
        "logits": batch_list_to_base64(logit_batches),
        "current_mu": self.moon_learner.mu,
      },
    )

  def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
    """性能追跡による拡張モデル評価"""
    # 注意: サーバーからロジットのみを受信するため、parametersは使用されません
    # モデル評価は現在のローカルモデル状態を使用します

    # モデルをロードする（MoonModel全体）
    loaded_model = load_model_from_state(self.client_state, self.net, self.local_model_name)
    if loaded_model is not None:
      self.net = loaded_model
    else:
      print("[Warning] No saved model state found, using initial model")

      # MoonModel専用のテストメソッドを使用
    loss, accuracy = CNNTask.test(self.net, self.val_loader, device=self.device)

    # 分析用の追加メトリクス
    current_round = int(config.get("current_round", 0))

    return (
      loss,
      len(self.val_loader.dataset),  # type: ignore
      {
        "accuracy": accuracy,
        "current_mu": self.moon_learner.mu,
        "round": current_round,
      },
    )
