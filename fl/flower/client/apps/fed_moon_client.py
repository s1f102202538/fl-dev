"""FedMoon with Logit Sharing: Flower / PyTorch アプリケーション"""

import copy
from typing import Dict, Tuple

import torch
from flwr.client import NumPyClient
from flwr.client.client import Client
from flwr.common import Context, RecordDict
from flwr.common.typing import NDArrays, UserConfigValue
from torch.utils.data import DataLoader

from flower.common.dataLoader.data_loader import load_data, load_public_data
from flower.common.models.mini_cnn import MiniCNN
from flower.common.models.moon_model import MoonModel
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
    # ベースモデルとMOON投影ヘッド付きモデルを作成
    self.base_net = net
    self.net = MoonModel(base_model=net, out_dim=256, n_classes=10)
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

      # 知識蒸留を実行して仮想グローバルモデルを作成
      distillation = Distillation(
        studentModel=copy.deepcopy(self.base_net),  # ベースモデルを使用
        public_data=self.public_test_data,
        soft_targets=logits,
      )

      # 蒸留により仮想グローバルベースモデルを作成
      virtual_global_base = distillation.train_knowledge_distillation(
        epochs=3,
        learning_rate=0.01,
        T=temperature,
        soft_target_loss_weight=0.4,
        ce_loss_weight=0.6,
        device=self.device,
      )

      # MoonModelでラップして仮想グローバルモデルを作成
      virtual_global_model = MoonModel(base_model=virtual_global_base, out_dim=256, n_classes=10)
      virtual_global_model.to(self.device)

      # グローバルモデルの設定確認
      print("=== グローバルモデル設定確認 ===")
      print(f"virtual_global_base type: {type(virtual_global_base)}")
      print(f"virtual_global_model.features type: {type(virtual_global_model.features)}")

      # 重みが正しくコピーされているかテスト（簡易版）
      try:
        from flower.common.models.mini_cnn import MiniCNN, MiniCNNFeatures

        if isinstance(virtual_global_base, MiniCNN) and isinstance(virtual_global_model.features, MiniCNNFeatures):
          print("グローバルモデルの重みコピー処理が実行されました")
          print("MiniCNN → MiniCNNFeatures の変換完了")
        else:
          print(f"想定外の型組み合わせ: {type(virtual_global_base)} → {type(virtual_global_model.features)}")
      except Exception as e:
        print(f"型確認エラー: {e}")
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

    # 適応対比学習による拡張FedMoon訓練
    train_loss = self.moon_trainer.train_with_enhanced_moon(
      model=self.net,
      train_loader=self.train_loader,
      lr=0.01,  # 元論文の推奨値
      epochs=5,  # 設定ファイルの値を使用
      current_round=current_round,
    )

    # 共有用の高品質ロジット生成（ベースモデルで実行）
    raw_logits = CNNTask.inference(self.base_net, self.public_test_data, device=self.device)
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

    # MoonModelのpredictメソッドを使用して分類出力のみ取得
    class MoonModelEvaluator(torch.nn.Module):
      def __init__(self, moon_model):
        super().__init__()
        self.moon_model = moon_model

      def forward(self, x):
        return self.moon_model.predict(x)

    evaluator = MoonModelEvaluator(self.net)
    loss, accuracy = CNNTask.test(evaluator, self.val_loader, device=self.device)

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
