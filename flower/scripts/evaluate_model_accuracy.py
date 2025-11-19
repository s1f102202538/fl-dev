#!/usr/bin/env python3
"""
IIDデータを使用してモデルの訓練と評価を行うスクリプト

使用方法:
    # デフォルト設定（CIFAR-10、MiniCNN、訓練あり）
    python scripts/evaluate_model_accuracy.py

    # エポック数を指定して訓練
    python scripts/evaluate_model_accuracy.py --epochs 10 --lr 0.001

    # 訓練データのサンプル数を制限
    python scripts/evaluate_model_accuracy.py --train-samples 5000 --epochs 5

    # MNIST用モデルの訓練と評価
    python scripts/evaluate_model_accuracy.py --model mini-cnn-mnist --dataset ylecun/mnist --epochs 10

    # SimpleCNNモデルの訓練と評価
    python scripts/evaluate_model_accuracy.py --model simple-cnn --dataset uoft-cs/cifar10 --epochs 15

    # MOONモデル（projection headあり）の訓練と評価
    python scripts/evaluate_model_accuracy.py --model mini-cnn --is-moon --use-projection-head --epochs 10

    # 訓練なし（初期化されたモデルの評価のみ）
    python scripts/evaluate_model_accuracy.py --no-train

    # 保存されたモデル（checkpoint）をロードして評価（訓練なし）
    python scripts/evaluate_model_accuracy.py --checkpoint path/to/model.pth --no-train
"""

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# flake8: noqa: E402
from fed.data.data_transform_manager import DataTransformManager
from fed.data.transformed_dataset import TransformedDataset
from fed.task.cnn_task import CNNTask
from fed.util.create_model import create_model


def parse_args():
  """コマンドライン引数をパース"""
  parser = argparse.ArgumentParser(
    description="IIDデータを使用してモデルの訓練と評価を行う",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
  )

  # モデル設定
  parser.add_argument(
    "--model",
    type=str,
    default="mini-cnn",
    choices=["mini-cnn", "mini-cnn-mnist", "simple-cnn", "simple-cnn-mnist"],
    help="使用するモデルの種類",
  )
  parser.add_argument(
    "--n-classes",
    type=int,
    default=10,
    help="分類クラス数",
  )
  parser.add_argument(
    "--is-moon",
    action="store_true",
    help="MOONモデルを使用（projection headの有無は--use-projection-headで制御）",
  )
  parser.add_argument(
    "--use-projection-head",
    action="store_true",
    default=False,
    help="MOONモデルでprojection headを使用（--is-moonと組み合わせて使用）",
  )
  parser.add_argument(
    "--no-use-projection-head",
    action="store_true",
    help="MOONモデルでprojection headを使用しない（明示的指定）",
  )
  parser.add_argument(
    "--out-dim",
    type=int,
    default=256,
    help="MOONモデルのprojection head出力次元",
  )

  # データセット設定
  parser.add_argument(
    "--dataset",
    type=str,
    default="uoft-cs/cifar10",
    help="使用するデータセット（Hugging Face Hub形式）",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    help="バッチサイズ",
  )
  parser.add_argument(
    "--train-samples",
    type=int,
    default=None,
    help="訓練サンプル数（Noneの場合は全訓練データを使用）",
  )
  parser.add_argument(
    "--test-samples",
    type=int,
    default=None,
    help="テストサンプル数（Noneの場合は全テストデータを使用）",
  )

  # 訓練設定
  parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    help="訓練エポック数",
  )
  parser.add_argument(
    "--lr",
    type=float,
    default=0.01,
    help="学習率",
  )
  parser.add_argument(
    "--no-train",
    action="store_true",
    help="訓練をスキップ（初期化されたモデルまたはcheckpointのモデルを評価のみ）",
  )

  # モデルロード設定
  parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="ロードするモデルのチェックポイントファイルパス",
  )

  # モデル保存設定
  parser.add_argument(
    "--save-model",
    type=str,
    default=None,
    help="訓練後にモデルを保存するパス",
  )

  # デバイス設定
  parser.add_argument(
    "--device",
    type=str,
    default="auto",
    choices=["auto", "cuda", "cpu"],
    help="使用するデバイス（autoは自動検出）",
  )

  # 出力設定
  parser.add_argument(
    "--verbose",
    action="store_true",
    help="詳細なログを表示",
  )

  return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
  """使用するデバイスを取得"""
  if device_arg == "auto":
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  return torch.device(device_arg)


def load_iid_train_data(dataset_name: str, batch_size: int, train_samples: int = None) -> DataLoader:
  """IID訓練データをロード

  Args:
      dataset_name: データセット名（Hugging Face Hub形式）
      batch_size: バッチサイズ
      train_samples: 使用する訓練サンプル数（Noneの場合は全データ）

  Returns:
      訓練データのDataLoader
  """
  print(f"📦 Loading train dataset: {dataset_name}")

  # 訓練データをロード
  if train_samples is not None:
    split_str = f"train[:{train_samples}]"
    print(f"   Using {train_samples} train samples")
  else:
    split_str = "train"
    print("   Using all available train samples")

  train_dataset = load_dataset(dataset_name, split=split_str)
  print(f"   Loaded {len(train_dataset)} samples")

  # データ変換の準備
  from fed.data.data_loader_config import DataLoaderConfig

  config = DataLoaderConfig(dataset_name=dataset_name)
  transform_manager = DataTransformManager(config)

  # PyTorch Datasetラッパーを作成
  train_dataset_wrapped = TransformedDataset(train_dataset, transform=transform_manager.train_transforms)

  # DataLoaderを作成（IIDなので shuffle=True）
  train_loader = DataLoader(
    train_dataset_wrapped,
    batch_size=batch_size,
    shuffle=True,  # 訓練データはシャッフル
    drop_last=True,  # 最後の不完全なバッチは削除
  )

  return train_loader


def load_iid_test_data(dataset_name: str, batch_size: int, test_samples: int = None) -> DataLoader:
  """IIDテストデータをロード

  Args:
      dataset_name: データセット名（Hugging Face Hub形式）
      batch_size: バッチサイズ
      test_samples: 使用するテストサンプル数（Noneの場合は全データ）

  Returns:
      テストデータのDataLoader
  """
  print(f"📦 Loading test dataset: {dataset_name}")

  # テストデータをロード
  if test_samples is not None:
    split_str = f"test[:{test_samples}]"
    print(f"   Using {test_samples} test samples")
  else:
    split_str = "test"
    print("   Using all available test samples")

  test_dataset = load_dataset(dataset_name, split=split_str)
  print(f"   Loaded {len(test_dataset)} samples")

  # データ変換の準備
  # データセット名から設定を推測
  from fed.data.data_loader_config import DataLoaderConfig

  config = DataLoaderConfig(dataset_name=dataset_name)
  transform_manager = DataTransformManager(config)

  # PyTorch Datasetラッパーを作成
  test_dataset_wrapped = TransformedDataset(test_dataset, transform=transform_manager.eval_transforms)

  # DataLoaderを作成（IIDなので shuffle=False）
  test_loader = DataLoader(
    test_dataset_wrapped,
    batch_size=batch_size,
    shuffle=False,  # IIDテストデータなのでシャッフル不要
    drop_last=False,  # 全データを評価するためdrop_lastはFalse
  )

  return test_loader


def evaluate_model(model, test_loader, device, verbose=False):
  """モデルを評価

  Args:
      model: 評価するモデル
      test_loader: テストデータローダー
      device: 使用するデバイス
      verbose: 詳細ログの表示

  Returns:
      (loss, accuracy)のタプル
  """
  print(f"\n🔍 Evaluating model on {device}")
  print(f"   Total batches: {len(test_loader)}")

  model.to(device)
  loss, accuracy = CNNTask.test(model, test_loader, device)

  return loss, accuracy


def train_model(model, train_loader, epochs, lr, device, verbose=False):
  """モデルを訓練

  Args:
      model: 訓練するモデル
      train_loader: 訓練データローダー
      epochs: エポック数
      lr: 学習率
      device: 使用するデバイス
      verbose: 詳細ログの表示

  Returns:
      最終エポックの平均訓練損失
  """
  print(f"\n🏋️  Training model on {device}")
  print(f"   Epochs: {epochs}")
  print(f"   Learning rate: {lr}")
  print(f"   Batches per epoch: {len(train_loader)}")

  model.to(device)
  final_loss = CNNTask.train(
    net=model,
    train_loader=train_loader,
    epochs=epochs,
    lr=lr,
    device=device,
  )

  print(f"   ✅ Training completed. Final loss: {final_loss:.6f}")

  return final_loss


def main():
  """メイン処理"""
  args = parse_args()

  print("=" * 80)
  print("🎯 Model Training and Evaluation with IID Data")
  print("=" * 80)

  # デバイス設定
  device = get_device(args.device)
  print("\n⚙️  Configuration:")
  print(f"   Model: {args.model}")
  print(f"   Dataset: {args.dataset}")
  print(f"   Device: {device}")
  print(f"   Batch size: {args.batch_size}")
  print(f"   Classes: {args.n_classes}")

  if not args.no_train:
    print("   Training: YES")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.lr}")
  else:
    print("   Training: NO (evaluation only)")

  # MOONモデルの設定
  use_projection_head = args.use_projection_head
  if args.no_use_projection_head:
    use_projection_head = False

  if args.is_moon:
    print(f"   MOON model: projection_head={'ON' if use_projection_head else 'OFF'}, out_dim={args.out_dim}")

  # モデルの作成またはロード
  print(f"\n🏗️  Creating model: {args.model}")
  model = create_model(
    model_name=args.model,
    is_moon=args.is_moon,
    out_dim=args.out_dim,
    n_classes=args.n_classes,
    use_projection_head=use_projection_head,
  )

  # チェックポイントのロード（指定されている場合）
  if args.checkpoint:
    print(f"📂 Loading checkpoint: {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
      print(f"❌ Error: Checkpoint file not found: {args.checkpoint}")
      sys.exit(1)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    print("   ✅ Checkpoint loaded successfully")
  else:
    print("   Using randomly initialized weights")

  # モデル情報の表示
  total_params = sum(p.numel() for p in model.parameters())
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"   Total parameters: {total_params:,}")
  print(f"   Trainable parameters: {trainable_params:,}")

  # 訓練の実行（--no-trainが指定されていない場合）
  if not args.no_train:
    train_loader = load_iid_train_data(
      dataset_name=args.dataset,
      batch_size=args.batch_size,
      train_samples=args.train_samples,
    )

    train_loss = train_model(
      model=model,
      train_loader=train_loader,
      epochs=args.epochs,
      lr=args.lr,
      device=device,
      verbose=args.verbose,
    )

    # モデルの保存（指定されている場合）
    if args.save_model:
      save_path = Path(args.save_model)
      save_path.parent.mkdir(parents=True, exist_ok=True)
      torch.save(model.state_dict(), save_path)
      print(f"\n💾 Model saved to: {save_path}")

  # テストデータのロード
  test_loader = load_iid_test_data(
    dataset_name=args.dataset,
    batch_size=args.batch_size,
    test_samples=args.test_samples,
  )

  # モデルの評価
  loss, accuracy = evaluate_model(
    model=model,
    test_loader=test_loader,
    device=device,
    verbose=args.verbose,
  )

  # 結果の表示
  print("\n" + "=" * 80)
  print("📊 Evaluation Results")
  print("=" * 80)
  print(f"   Test Loss: {loss:.6f}")
  print(f"   Test Accuracy: {accuracy:.2f}%")
  print("=" * 80)

  # 期待される精度範囲の表示（参考情報）
  print("\n📝 Reference Information:")
  if args.checkpoint is None and args.no_train:
    print("   ⚠️  Model is randomly initialized (not trained)")
    print(f"   Expected accuracy for random guessing: ~{100.0 / args.n_classes:.2f}%")
  elif not args.no_train:
    print("   ✅ Model was trained in this session")
    print(f"   Training epochs: {args.epochs}")
    print(f"   Final training loss: {train_loss:.6f}")
  else:
    print("   ✅ Model loaded from checkpoint")
    print("   Expected accuracy depends on training quality")

  if "cifar10" in args.dataset.lower():
    print("\n   CIFAR-10 Baseline Accuracies:")
    print("   - Random: ~10%")
    print("   - Simple CNN (well-trained): ~70-75%")
    print("   - ResNet (well-trained): ~90-95%")
  elif "mnist" in args.dataset.lower():
    print("\n   MNIST Baseline Accuracies:")
    print("   - Random: ~10%")
    print("   - Simple CNN (well-trained): ~98-99%")

  print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
  main()
