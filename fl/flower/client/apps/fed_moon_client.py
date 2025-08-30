"""FedMoon with Logit Sharing: Flower / PyTorch アプリケーション"""

import copy
from typing import Dict, Tuple

import torch
from flower.common.dataLoader.data_loader import load_data, load_public_data
from flower.common.models.mini_cnn import MiniCNN
from flower.common.task.cnn_task import CNNTask
from flower.common.task.distillation import Distillation
from flower.common.task.moon import MoonContrastiveLearning, MoonTrainer
from flower.common.util.util import (
  base64_to_batch_list,
  batch_list_to_base64,
  filter_and_calibrate_logits,
  load_model_from_state,
  save_model_to_state,
)
from flwr.client import NumPyClient
from flwr.client.client import Client
from flwr.common import Context, RecordDict
from flwr.common.typing import NDArrays, UserConfigValue
from torch.utils.data import DataLoader


class FedMoonClient(NumPyClient):
  """ロジット共有機能を持つFedMoonクライアント"""

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
    self.public_test_data = public_test_data

    # モデル状態保存用の名前定義
    self.local_model_name = "fed-moon-model"

    # Moon対比学習の初期化
    self.moon_learner = MoonContrastiveLearning(
      mu=5.0,
      temperature=0.5,
      adaptive_mu=True,
      min_mu=1.0,
      max_mu=10.0,
      device=self.device,
    )

    # Moonトレーナーの初期化
    self.moon_trainer = MoonTrainer(
      moon_learner=self.moon_learner,
      device=self.device,
    )

  def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
    """拡張FedMoon対比学習と適応ロジット共有によるローカルモデル訓練"""
    # 設定から学習率を取得
    lr = float(config.get("lr", 0.1))
    current_round = int(config.get("current_round", 0))

    # ラウンドベースの適応パラメータ更新
    self.moon_learner.update_adaptive_parameters(current_round)

    # 共有ロジットによる拡張知識蒸留（仮想グローバルモデルの作成）
    distillation_performed = False
    virtual_global_model = None

    if "avg_logits" in config and config["avg_logits"] is not None:
      logits = base64_to_batch_list(config["avg_logits"])
      temperature = float(config.get("temperature", 3.0))

      # 知識蒸留を実行して仮想グローバルモデルを作成
      distillation = Distillation(
        studentModel=copy.deepcopy(self.net),
        public_data=self.public_test_data,
        soft_targets=logits,
      )

      # 蒸留により仮想グローバルモデルを作成
      virtual_global_model = distillation.train_knowledge_distillation(
        epochs=2,  # 蒸留のエポック数
        learning_rate=0.001,
        T=temperature,
        soft_target_loss_weight=0.4,
        ce_loss_weight=0.6,
        device=self.device,
      )
      distillation_performed = True

      # 仮想グローバルモデルでMoon学習器を更新
      # 前回のローカルモデル状態を復元
      previous_model = load_model_from_state(self.client_state, self.net, self.local_model_name)
      if previous_model is not None:
        # 現在のローカルモデルを基準にMoon学習器を更新
        self.moon_learner.update_models(previous_model, virtual_global_model)
      else:
        print("previous model が見つかりませんでした")
    else:
      # ロジットが提供されない場合、Moon学習のベースラインとして現在のモデルを使用
      if self.moon_learner.global_model is None:
        current_model_copy = copy.deepcopy(self.net)
        self.moon_learner.update_models(copy.deepcopy(self.net), current_model_copy)

    # 適応対比学習による拡張FedMoon訓練
    train_loss = self.moon_trainer.train_with_enhanced_moon(
      model=self.net,
      train_loader=self.train_loader,
      lr=lr,
      local_epochs=self.local_epochs,
      distillation_performed=distillation_performed,
    )

    # 共有用の高品質ロジット生成
    raw_logits = CNNTask.inference(self.net, self.public_test_data, device=self.device)
    logit_batches = filter_and_calibrate_logits(raw_logits, temperature=1.5)

    # 適応学習のための性能追跡
    self.moon_learner.track_performance(train_loss, distillation_performed, current_round)

    # 学習完了後のローカルモデル状態を保存
    save_model_to_state(self.net, self.client_state, self.local_model_name)

    print(f"FedMoon拡張クライアント完了 (ラウンド {current_round})")
    print("クライアント送信ロジット統計:", [b.mean().item() for b in logit_batches])

    return (
      [],  # ロジット共有のみでパラメータ集約は行わないため空リストを返す
      len(self.train_loader.dataset),  # type: ignore
      {
        "train_loss": train_loss,
        "logits": batch_list_to_base64(logit_batches),
        "distillation_performed": distillation_performed,
        "current_mu": self.moon_learner.mu,
      },
    )

  def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
    """性能追跡による拡張モデル評価"""
    # 注意: サーバーからロジットのみを受信するため、parametersは使用されません
    # モデル評価は現在のローカルモデル状態を使用します

    # モデルをロードする
    loaded_model = load_model_from_state(self.client_state, self.net, self.local_model_name)
    if loaded_model is not None:
      self.net = loaded_model
    else:
      print("警告: 保存されたモデル状態が見つからないため、初期状態のモデルを使用します")

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
        "performance_trend": len(self.moon_learner.performance_history),
      },
    )

  @staticmethod
  def client_fn(context: Context) -> Client:
    """FedMoonクライアントインスタンスを作成"""
    # モデルとデータの読み込み
    net = MiniCNN()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_loader, val_loader = load_data(partition_id, num_partitions)
    public_test_data = load_public_data(batch_size=32, max_samples=500)
    local_epochs = context.run_config["local-epochs"]

    # クライアント状態の作成
    client_state = context.state

    return FedMoonClient(
      net=net,
      client_state=client_state,
      train_loader=train_loader,
      val_loader=val_loader,
      public_test_data=public_test_data,
      local_epochs=local_epochs,
    ).to_client()
