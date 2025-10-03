"""FedMoon with Logit Sharing: Flower / PyTorch アプリケーション"""

import copy
from typing import Dict, Tuple

import torch
from flwr.client import NumPyClient
from flwr.client.client import Client
from flwr.common import Context, RecordDict
from flwr.common.typing import NDArrays, UserConfigValue
from torch.utils.data import DataLoader

from fl.flower.common.util.data_loader import load_data, load_public_data
from fl.flower.common.util.model_util import (
  base64_to_batch_list,
  batch_list_to_base64,
  filter_and_calibrate_logits,
  load_model_from_state,
  save_model_to_state,
)
from flower.common.models.mini_cnn import MiniCNN
from flower.common.models.moon_model import MoonModel
from flower.common.task.cnn_task import CNNTask
from flower.common.task.distillation import Distillation
from flower.common.task.moon import MoonContrastiveLearning, MoonTrainer


class FedMoonClient(NumPyClient):
  """ロジット共有機能を持つFedMoonクライアント

  注意: このクライアントは一貫してMoonModelを使用します。
  - self.net: MoonModel（MOON対比学習、訓練、評価、ロジット生成で一貫使用）
  """

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
    # MoonModelを一貫使用（蒸留、訓練、評価、ロジット生成すべて対応）
    self.net = MoonModel(out_dim=256, n_classes=10)
    self.client_state = client_state
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.local_epochs: int = int(local_epochs)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.net.to(self.device)
    self.public_test_data = public_test_data

    # モデル状態保存用の名前定義
    self.local_model_name = "fed-moon-model"
    # 前ラウンドモデル保存用
    self.previous_model_name = "fed-moon-previous-model"

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

    # 前ラウンドモデルを復元（対比学習用）
    previous_round_model = load_model_from_state(self.client_state, self.net, self.previous_model_name)

    # 現在のローカルモデル状態を復元
    loaded_model = load_model_from_state(self.client_state, self.net, self.local_model_name)
    if loaded_model is not None:
      self.net = loaded_model
      print(f"ローカルモデルを復元しました (ラウンド {current_round})")
    else:
      print(f"初回ラウンドまたはモデル状態なし (ラウンド {current_round})")

    virtual_global_model = None

    if "avg_logits" in config and config["avg_logits"] is not None:
      logits = base64_to_batch_list(config["avg_logits"])
      temperature = float(config.get("temperature", 3.0))

      # 蒸留により仮想グローバルモデルを直接作成（MoonModelを使用）
      virtual_global_model = MoonModel(out_dim=256, n_classes=10)
      distillation = Distillation(
        studentModel=virtual_global_model,  # MoonModelを直接使用
        public_data=self.public_test_data,
        soft_targets=logits,
      )

      # MoonModelで知識蒸留を実行
      virtual_global_model = distillation.train_knowledge_distillation(
        epochs=1,
        learning_rate=0.001,
        T=temperature,
        soft_target_loss_weight=0.4,
        ce_loss_weight=0.6,
        device=self.device,
      )
      virtual_global_model.to(self.device)

      # グローバルモデルの設定確認
      print("=== 仮想グローバルモデル設定確認 ===")
      print(f"virtual_global_model type: {type(virtual_global_model)}")
      print(f"virtual_global_model.features type: {type(virtual_global_model.features)}")
      print("蒸留によりMoonModelで直接仮想グローバルモデルを作成しました")
      print("=== 確認完了 ===")

      # Moon学習器を更新: previous_model -> 前ラウンドモデル, global_model -> 仮想グローバルモデル
      if previous_round_model is not None:
        # previous_round_modelは既にMoonModelなので、そのまま使用
        self.moon_learner.update_models(previous_round_model, virtual_global_model)
        print("前ラウンドモデルと仮想グローバルモデルでMoon学習器を更新しました")
      else:
        # 初回ラウンドの場合、現在のモデルをpreviousとして使用
        current_model_copy = copy.deepcopy(self.net)
        self.moon_learner.update_models(current_model_copy, virtual_global_model)
        print("初回ラウンド: 現在モデルをpreviousとしてMoon学習器を初期化")
    else:
      # サーバーロジットがない場合（第1ラウンドまたはロジットなし）
      if current_round == 1:
        print("第1ラウンド: 対比学習なしで通常訓練のみ")
        # 対比学習は無効（global_modelとprevious_modelがない）
      else:
        print(f"ラウンド {current_round}: サーバーロジットなし、対比学習なし")

    # MOON対比学習による訓練（MoonModelを使用）
    train_loss = self.moon_trainer.train_with_enhanced_moon(
      model=self.net,
      train_loader=self.train_loader,
      lr=0.01,  # 元論文の推奨値
      epochs=3,  # 設定ファイルの値を使用
      current_round=current_round,
    )

    # ロジット共有用の高品質ロジット生成（MoonModelで実行）
    # MoonModel専用のinferenceメソッドを使用（複数出力対応）
    raw_logits = CNNTask.moon_inference(self.net, self.public_test_data, device=self.device)
    logit_batches = filter_and_calibrate_logits(raw_logits, temperature=1.5)

    # 学習完了後のローカルモデル状態を保存（MoonModel全体）
    save_model_to_state(self.net, self.client_state, self.local_model_name)

    # 次ラウンド用：現在のMoonModel全体を前ラウンドモデルとして保存
    # 注意：学習前ではなく学習後のモデルを次ラウンドのprevious_modelとして使用
    # MoonModelには投影ヘッドも含まれているため、全体を保存する必要がある
    save_model_to_state(self.net, self.client_state, self.previous_model_name)

    print(f"FedMoon拡張クライアント完了 (ラウンド {current_round})")
    print("クライアント送信ロジット統計:", [b.mean().item() for b in logit_batches])

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
      print("警告: 保存されたモデル状態が見つからないため、初期状態のモデルを使用します")

      # MoonModel専用のテストメソッドを使用（複数出力対応）
    loss, accuracy = CNNTask.moon_test(self.net, self.val_loader, device=self.device)

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

  @staticmethod
  def client_fn(context: Context) -> Client:
    """FedMoonクライアントインスタンスを作成"""
    # モデルとデータの読み込み
    net = MiniCNN()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_loader, val_loader = load_data(partition_id, num_partitions)
    public_test_data = load_public_data(batch_size=32, max_samples=1000)
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
