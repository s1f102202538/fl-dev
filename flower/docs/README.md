# Federated Learning Framework Documentation

## 概要

本プロジェクトは、Flower フレームワークをベースにした連合学習（Federated Learning）の実装です。クライアント間でモデルを分散学習し、サーバで集約・蒸留を行う仕組みを提供します。

## 実行環境

### 開発環境

- **OS**: Debian GNU/Linux 12 (bookworm)
- **コンテナ**: Dev Container
- **Python**: 3.12+

### 使用フレームワーク・ライブラリ | フレームワーク/ライブラリ | バージョン | 用途 | |------------------------|----------|------| | **Flower** | ≥1.18.0 | 連合学習フレームワーク（シミュレーション対応） | | **PyTorch** | 2.5.1 | 深層学習フレームワーク | | **TorchVision** | 0.20.1 | 画像処理・データセット | | **flwr-datasets** | ≥0.5.0 | Flower 用データセット管理 | | **Weights & Biases** | - | 実験管理・可視化（オプション） |

## ディレクトリ構造
flower/
├── pyproject.toml              # プロジェクト設定（依存関係、Flower アプリ設定）
├── docs/                       # ドキュメント
│   ├── README.md              # 本ドキュメント
│   ├── LOGIT_FILTERING_EVALUATION.md
│   └── SUPPORTED_DATASETS.md
├── fed/                        # 連合学習コア実装
│   ├── algorithms/            # 学習アルゴリズム
│   │   ├── csd_based_moon.py          # CSD ベース MOON
│   │   ├── distillation.py            # 知識蒸留
│   │   ├── loca_based_distillation.py # LoCa ベース蒸留
│   │   ├── logit_calibration_moon.py  # ロジット校正 MOON
│   │   └── moon.py                    # MOON コントラスト学習
│   ├── communication/         # クライアント・サーバ通信
│   │   ├── client/           # クライアント実装
│   │   │   ├── apps/        # クライアントアプリケーション
│   │   │   │   ├── fed_kd_client.py
│   │   │   │   ├── fed_kd_params_share_client.py
│   │   │   │   ├── fed_moon_client.py
│   │   │   │   ├── fed_moon_params_share_client.py
│   │   │   │   └── fed_moon_params_share_csd_client.py
│   │   │   └── client_app.py # クライアントルーティング
│   │   └── server/           # サーバ実装
│   │       ├── apps/        # サーバアプリケーション
│   │       │   ├── fed_avg_server.py
│   │       │   ├── fed_kd_params_share_server.py
│   │       │   └── fed_kd_params_share_csd_server.py
│   │       ├── strategy/    # 集約戦略
│   │       │   ├── fed_avg.py
│   │       │   ├── fed_kd_params_share.py
│   │       │   ├── fed_kd_params_share_csd.py
│   │       │   └── fed_moon_params_share.py
│   │       └── server_app.py # サーバルーティング
│   ├── data/                  # データ管理
│   │   ├── data_loader_config.py     # データローダー設定
│   │   └── data_loader_manager.py    # データローダー管理
│   ├── models/                # モデル定義
│   │   ├── base_model.py            # 基底モデル
│   │   ├── mini_cnn.py              # Mini CNN
│   │   ├── moon_model.py            # MOON モデル
│   │   └── resnet.py                # ResNet
│   ├── task/                  # タスク実装
│   │   └── cnn_task.py             # CNN 訓練・評価・推論
│   └── util/                  # ユーティリティ
│       ├── communication_cost.py    # 通信コスト計算
│       └── model_util.py            # モデルユーティリティ
├── outputs/                   # 実験結果出力
└── scripts/                   # スクリプト
    ├── evaluate_model_accuracy.py
    └── visualize_data_distribution.py
```

## 各モジュールの役割

### 1. fed/algorithms/ - 学習アルゴリズム

連合学習で使用する各種アルゴリズムの実装。

- **moon.py**: MOON (Model-Contrastive Federated Learning) の実装
  - グローバルモデルと前回のローカルモデルを用いたコントラスト学習
  - MoonContrastiveLearning: 対比損失の計算
  - MoonTrainer: MOON による訓練ループ

- **csd_based_moon.py**: CSD（Class Prototype Similarity Distillation）を統合した MOON
  - クラスプロトタイプとの類似度に基づく蒸留損失を追加
  - CsdBasedMoonContrastiveLearning: CSD 損失を含む対比学習
  - CsdBasedMoonTrainer: CSD + MOON の訓練ループ

- **logit_calibration_moon.py**: FedLC（ロジット校正）を統合した MOON
  - クラス不均衡に対するロジット校正を適用
  - LogitCalibrationMoonContrastiveLearning: FedLC 対応
  - LogitCalibrationMoonTrainer: 校正付き訓練

- **distillation.py**: 標準的な知識蒸留
  - サーバ側でのモデル蒸留
  - Distillation: KL divergence ベースの蒸留

- **loca_based_distillation.py**: LoCa（Logit Calibration）ベースの蒸留
  - 非正解クラスの確率を縮小し、正解クラスを強調
  - 温度スケーリングと組み合わせた蒸留

### 2. fed/communication/ - 通信層

クライアントとサーバ間の通信とデータ交換を管理。

#### client/ - クライアント側

- **client_app.py**: クライアントのルーティングハブ
  - client_name に基づいて適切なクライアントアプリを起動
  - 対応クライアント: fed-kd-client, fed-kd-params-share-client, fed-moon-client, fed-moon-params-share-client, fed-moon-params-share-csd-client

- **apps/**: 各クライアント実装
  - **fed_kd_params_share_client.py**: パラメータ共有 + ロジット送信クライアント
  - **fed_moon_params_share_client.py**: MOON + パラメータ共有クライアント
  - **fed_moon_params_share_csd_client.py**: CSD-MOON クライアント

#### server/ - サーバ側

- **server_app.py**: サーバのルーティングハブ
  - server_name に基づいて適切なサーバアプリと戦略を起動
  - 対応サーバ: fed-avg-server, fed-kd-params-share-server, fed-kd-params-share-csd-server

- **strategy/**: 集約戦略
  - **fed_avg.py**: FedAvg（標準的な連合平均化）
  - **fed_kd_params_share.py**: パラメータ配布 + ロジット集約 + 蒸留
  - **fed_kd_params_share_csd.py**: CSD 統合版（クラスプロトタイプ生成）
  - **fed_moon_params_share.py**: MOON パラメータ共有戦略

- **apps/**: サーバアプリケーション
  - 各戦略を初期化し、ServerAppComponents を返す

### 3. fed/data/ - データ管理

連合学習のためのデータ分割とローディング。

- **data_loader_config.py**: データローダーの設定
  - DataLoaderConfig: データセット名、バッチサイズ、分割方法などを定義
  - train_max_samples: 訓練データの最大サンプル数制限

- **data_loader_manager.py**: データローダーの管理
  - create_federated_data_loaders(): クライアント用の分割データローダー作成
  - create_public_test_data_loader(): サーバ用の公開テストデータ作成
  - IID / Non-IID 分割対応

### 4. fed/models/ - モデル定義

ニューラルネットワークモデルの実装。

- **base_model.py**: 基底クラス
  - BaseModel: 全モデルの抽象基底クラス
  - predict(), forward() メソッドを定義

- **mini_cnn.py**: シンプルな CNN
  - MiniCNN: 軽量な畳み込みニューラルネットワーク
  - CIFAR-10 などの小規模データセット用

- **moon_model.py**: MOON 用モデル
  - ModelFedCon: プロジェクションヘッド付き CNN
  - features → l1 → l2 (projection head) と features → l3 (classifier) の二つの出力

- **resnet.py**: ResNet ベースモデル
  - ResNet-18, ResNet-50 など

### 5. fed/task/ - タスク実装

モデルの訓練・評価・推論タスク。

- **cnn_task.py**: CNN タスクの実装
  - train(): 標準的な訓練
  - test(): テスト評価
  - inference(): ロジット生成（補正なし）
  - inference_with_label_correction(): ラベル補正付き推論
  - inference_with_loca(): LoCa 補正付き推論

### 6. fed/util/ - ユーティリティ

共通のヘルパー関数。

- **communication_cost.py**: 通信コスト計算
  - パラメータやロジットのサイズを計算
  - MB 単位での通信量測定

- **model_util.py**: モデルユーティリティ
  - Base64 エンコード/デコード
  - バッチリストの変換
  - 実行ディレクトリの作成

## 実行方法

### 基本実行
```bash
flwr run .
```

### 設定のカスタマイズ

`pyproject.toml` の `[tool.flwr.app.config]` セクションで設定を変更：

```toml
[tool.flwr.app.config]
model_name = "mini-cnn"              # モデル選択
dataset_name = "uoft-cs/cifar10"     # データセット選択
client_name = "fed-moon-params-share-client"  # クライアント選択
server_name = "fed-kd-params-share-server"    # サーバ選択
num_clients = 10                     # クライアント数
num_rounds = 50                      # ラウンド数
```

## 主要な連合学習戦略

### 1. FedAvg (Federated Averaging)

- 標準的な連合学習
- 各クライアントがローカルで訓練 → サーバが重み付き平均化

### 2. FedKD with Parameter Sharing

- サーバがモデルパラメータをクライアントに配布
- クライアントが訓練後にロジットを送信
- サーバがロジットを集約して知識蒸留

### 3. FedMoon with Parameter Sharing

- MOON コントラスト学習を使用
- グローバルモデルと前回モデルを用いた対比損失
- パラメータ共有とロジット集約を組み合わせ

### 4. CSD-based FedMoon

- クラスプロトタイプを生成
- プロトタイプとの類似度に基づく蒸留損失
- クラス不均衡データに対応

### 5. LoCa-based Distillation

- 非正解クラスの確率を縮小
- 正解クラスの確率を強調
- 温度スケーリングと組み合わせた蒸留

## 実験管理

### Weights & Biases (W&B) 連携

環境変数で W&B プロジェクトを設定：

```bash
export WANDB_PROJECT_NAME="your-project-name"
```

### 出力ディレクトリ

実験結果は `outputs/YYYY-MM-DD/HH-MM-SS/` に保存されます：

```
outputs/
└── 2025-11-30/
    └── 14-30-00/
        ├── results.json          # 評価結果
        ├── best_model.pth        # ベストモデル（戦略による）
        └── logs/                 # ログファイル
```

## トラブルシューティング

### よくある問題

1. **モジュールが見つからない**
   ```bash
   pip install -e .
   ```

2. **CUDA メモリ不足**
   - バッチサイズを小さくする
   - モデルサイズを縮小

3. **データセットのダウンロードエラー**
   - ネットワーク接続を確認
   - Hugging Face Datasets のキャッシュをクリア

## 参考文献

- **Flower**: [https://flower.ai/](https://flower.ai/)
- **MOON**: Li et al. "Model-Contrastive Federated Learning" (CVPR 2021)
- **FedLC**: Zhang et al. "FedLC: Federated Learning with Logit Calibration" (NeurIPS 2022)
- **Knowledge Distillation**: Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)

## ライセンス

Apache License 2.0
